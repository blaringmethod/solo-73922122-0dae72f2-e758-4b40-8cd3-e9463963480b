"""Background task orchestration.

Each upload is processed by a daemon worker thread with its own SQLite
connection. Cancellation is cooperative: the cancel endpoint flips an event
and returns immediately, while the worker observes the event at every batch
boundary and aborts before anything can be published.
"""

from __future__ import annotations

import shutil
import sqlite3
import threading
from pathlib import Path

from . import config, db
from .fastq import (
    FastqFormatError,
    TaskCancelled,
    parse_file,
    promote_results,
    publish,
    validate_pairing,
)

# task_id -> threading.Event set when the task is asked to cancel.
_cancel_events: dict[str, threading.Event] = {}
_registry_lock = threading.Lock()
# Keep worker threads reachable (mostly for tests / diagnostics).
_workers: dict[str, threading.Thread] = {}
# task_id -> release reaper thread currently running for that task.
_reapers: dict[str, threading.Thread] = {}


class DownloadTracker:
    """Counts in-flight result downloads per task.

    The release protocol relies on two orderings:
      * the download endpoint acquires a permit BEFORE it re-checks the
        task's release status, and holds it until streaming ends;
      * the release endpoint commits 'releasing' BEFORE the reaper looks at
        the counter.
    A download admitted before the release is therefore always visible to
    the reaper, which waits for it; a download that arrives after the
    release sees 'releasing' in the database and is rejected with 409.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._counts: dict[str, int] = {}

    def acquire(self, task_id: str) -> None:
        with self._cond:
            self._counts[task_id] = self._counts.get(task_id, 0) + 1

    def release(self, task_id: str) -> None:
        with self._cond:
            remaining = self._counts.get(task_id, 0) - 1
            if remaining > 0:
                self._counts[task_id] = remaining
            else:
                self._counts.pop(task_id, None)
            self._cond.notify_all()

    def active(self, task_id: str) -> int:
        with self._cond:
            return self._counts.get(task_id, 0)

    def wait_until_idle(self, task_id: str) -> None:
        with self._cond:
            self._cond.wait_for(lambda: self._counts.get(task_id, 0) == 0)


# In-flight downloads of published results, keyed by task id.
DOWNLOADS = DownloadTracker()


class UploadTooLarge(Exception):
    def __init__(self, side: str, limit: int) -> None:
        self.side = side
        self.limit = limit
        super().__init__(f"{side} exceeds the {limit} byte upload limit")


def register_cancel_event(task_id: str) -> threading.Event:
    with _registry_lock:
        event = _cancel_events.get(task_id)
        if event is None:
            event = threading.Event()
            _cancel_events[task_id] = event
        return event


def request_cancel(task_id: str) -> None:
    register_cancel_event(task_id).set()


def task_dir(task_id: str) -> Path:
    return config.TASKS_DIR / task_id


def cleanup_task_dir(task_id: str) -> None:
    """Remove only this task's own directory; UUID names isolate tasks."""
    path = task_dir(task_id)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _fail(
    conn: sqlite3.Connection,
    task_id: str,
    message: str,
    *,
    cancelled: bool = False,
) -> None:
    # Remove files BEFORE publishing the terminal status, so any observer of
    # 'failed'/'cancelled' is guaranteed the staging directory is already gone.
    cleanup_task_dir(task_id)
    db.delete_records(conn, task_id)
    db.update_task(
        conn,
        task_id,
        status="cancelled" if cancelled else "failed",
        stage=config.STAGE_DONE,
        progress=1.0 if cancelled else 0.0,
        message=message,
        result_r1="",
        result_r2="",
    )
    with _registry_lock:
        _cancel_events.pop(task_id, None)


def run_task(task_id: str) -> None:
    """Worker entry point: parse, validate, pair, publish or fail atomically."""
    event = register_cancel_event(task_id)
    conn = db.connect()
    base = task_dir(task_id)
    r1_path = base / "input_R1.fastq"
    r2_path = base / "input_R2.fastq"
    total_bytes = max(r1_path.stat().st_size + r2_path.stat().st_size, 1)

    def parse_progress(side: int, byte_offset: int) -> None:
        other = 0 if side == 1 else r1_path.stat().st_size
        current = other + byte_offset
        db.update_task(
            conn,
            task_id,
            stage=config.STAGE_PARSE,
            progress=min(0.9 * current / total_bytes, 0.9),
        )

    try:
        if event.is_set():
            raise TaskCancelled()
        db.update_task(conn, task_id, status="processing", stage=config.STAGE_PARSE)

        n1 = parse_file(
            conn, task_id, r1_path, 1, event.is_set,
            lambda off: parse_progress(1, off),
        )
        n2 = parse_file(
            conn, task_id, r2_path, 2, event.is_set,
            lambda off: parse_progress(2, off),
        )
        if event.is_set():
            raise TaskCancelled()

        db.update_task(conn, task_id, stage=config.STAGE_VALIDATE, progress=0.92)
        validate_pairing(conn, task_id)
        if n1 != n2:
            # Defensive: orphan validation should already have caught this.
            raise FastqFormatError(
                f"record count mismatch: R1 has {n1} records, R2 has {n2}"
            )
        if event.is_set():
            raise TaskCancelled()

        db.update_task(conn, task_id, stage=config.STAGE_PUBLISH, progress=0.95)
        out_dir = base / "output"
        tmp1, tmp2 = publish(
            conn,
            task_id,
            out_dir,
            event.is_set,
            lambda fraction: db.update_task(
                conn, task_id, stage=config.STAGE_PUBLISH,
                progress=0.95 + 0.05 * fraction,
            ),
        )

        # Arbitrate success vs. cancellation in the database BEFORE any result
        # file gets its final name. If the cancel endpoint flipped the status
        # to 'cancelling', this UPDATE matches no row: we keep the temp files
        # unpublished and fall through to cancellation cleanup.
        cur = conn.execute(
            "UPDATE tasks SET status = 'succeeded', stage = ?, progress = 1.0, "
            "message = ?, result_r1 = ?, result_r2 = ?, updated_at = ? "
            "WHERE id = ? AND status != 'cancelling'",
            (
                config.STAGE_DONE,
                f"paired {n1} reads",
                "output/R1.fastq",
                "output/R2.fastq",
                db.utcnow(),
                task_id,
            ),
        )
        if cur.rowcount == 0:
            raise TaskCancelled()

        # We have won the arbitration; the temp files are complete and durable.
        promote_results(out_dir, tmp1, tmp2)
        db.delete_records(conn, task_id)
        # Inputs are no longer needed once results are durable.
        for stale in (r1_path, r2_path):
            try:
                stale.unlink(missing_ok=True)
            except OSError:
                pass
    except TaskCancelled:
        _fail(conn, task_id, "task cancelled before completion", cancelled=True)
    except FastqFormatError as exc:
        if event.is_set():
            _fail(conn, task_id, "task cancelled before completion", cancelled=True)
        else:
            _fail(conn, task_id, f"rejected: {exc}")
    except Exception as exc:  # noqa: BLE001 - any failure fails the whole task
        if event.is_set():
            _fail(conn, task_id, "task cancelled before completion", cancelled=True)
        else:
            _fail(conn, task_id, f"unexpected error: {exc!r}")
    finally:
        conn.close()
        with _registry_lock:
            _workers.pop(task_id, None)


def start_worker(task_id: str) -> None:
    thread = threading.Thread(
        target=run_task, args=(task_id,), name=f"fastq-task-{task_id[:8]}", daemon=True
    )
    with _registry_lock:
        _workers[task_id] = thread
    thread.start()


def recover_on_startup() -> dict[str, int]:
    """Mark unfinished tasks failed and remove their staging files.

    Succeeded tasks keep their result files and remain downloadable. Tasks
    whose release was interrupted mid-cleanup (release_status='releasing')
    are cleaned up again here: a fresh process has no in-flight downloads,
    so the retry can run inline.
    """
    conn = db.connect()
    recovered = 0
    releasing: list[str] = []
    try:
        ids = db.non_terminal_task_ids(conn)
        for task_id in ids:
            db.delete_records(conn, task_id)
            db.update_task(
                conn,
                task_id,
                status="failed",
                stage=config.STAGE_DONE,
                progress=0.0,
                message="interrupted by service restart",
                result_r1="",
                result_r2="",
            )
            cleanup_task_dir(task_id)
            recovered += 1
        releasing = db.releasing_task_ids(conn)
    finally:
        conn.close()
    # Cancellation registry is empty in a fresh process.
    resumed = 0
    for task_id in releasing:
        if attempt_release_cleanup(task_id):
            resumed += 1
    return {"recovered": recovered, "release_resumed": resumed}


def attempt_release_cleanup(task_id: str) -> bool:
    """Try to delete both published result files of a releasing task.

    Returns True when the task is fully released afterwards (both ends
    gone). Deleting either end may fail; in that case the task stays in
    'releasing' with the error recorded, downloads remain blocked, and a
    later release request or restart retries the cleanup. Only this task's
    own directory is ever touched.
    """
    conn = db.connect()
    try:
        row = db.get_task(conn, task_id)
        if row is None or row["release_status"] != db.RELEASE_RELEASING:
            return row is not None and row["release_status"] == db.RELEASE_RELEASED
        base = task_dir(task_id).resolve()
        errors: list[str] = []
        for rel in (row["result_r1"], row["result_r2"]):
            if not rel:
                continue
            path = (base / rel).resolve()
            # Never follow a stored path outside this task's own directory.
            if base not in path.parents:
                errors.append(f"{rel}: path escapes task directory")
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                errors.append(f"{rel}: {exc.strerror or exc}")
        if errors:
            db.update_task(conn, task_id, release_error="; ".join(errors))
            return False
        db.mark_released(conn, task_id)
    finally:
        conn.close()
    # Both ends are gone; drop the remaining directory shell best-effort.
    cleanup_task_dir(task_id)
    return True


def _reaper_main(task_id: str) -> None:
    """Wait for in-flight downloads to drain, then clean up the results."""
    try:
        DOWNLOADS.wait_until_idle(task_id)
        attempt_release_cleanup(task_id)
    finally:
        with _registry_lock:
            _reapers.pop(task_id, None)


def request_release_cleanup(task_id: str) -> None:
    """Ensure a reaper thread is working on this task's release.

    Idempotent: if a reaper is already waiting for downloads to drain (or
    cleaning up), the call is a no-op; if a previous cleanup attempt failed
    and its reaper exited, a new one is started — this is how repeated
    release requests retry an unfinished cleanup.
    """
    with _registry_lock:
        existing = _reapers.get(task_id)
        if existing is not None and existing.is_alive():
            return
        thread = threading.Thread(
            target=_reaper_main,
            args=(task_id,),
            name=f"fastq-release-{task_id[:8]}",
            daemon=True,
        )
        _reapers[task_id] = thread
        thread.start()
