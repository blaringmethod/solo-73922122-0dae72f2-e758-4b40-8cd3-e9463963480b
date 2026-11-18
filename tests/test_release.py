"""Self-tests for the disk-release lifecycle of succeeded tasks.

Covers: accepting a release, blocking new downloads while in-flight ones
finish (or disconnect), retrying failed cleanups, restart recovery, and
isolation between tasks. Run with:
    .venv/bin/python -I tests/test_release.py
"""

from __future__ import annotations

import asyncio
import errno
import io
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from app import config, db, worker  # noqa: E402
from app.main import app  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def make_record(rid: str, seq: bytes, suffix: str = "") -> bytes:
    return (b"@" + rid.encode() + suffix.encode() + b"\n" + seq + b"\n+\n"
            + b"~" * len(seq) + b"\n")


def fastq(records: list[bytes]) -> io.BytesIO:
    return io.BytesIO(b"".join(records))


async def upload(client: httpx.AsyncClient, r1: io.BytesIO,
                 r2: io.BytesIO) -> httpx.Response:
    r1.seek(0)
    r2.seek(0)
    return await client.post(
        "/tasks",
        files={"r1": ("r1.fastq", r1, "application/octet-stream"),
               "r2": ("r2.fastq", r2, "application/octet-stream")},
    )


async def make_succeeded(client: httpx.AsyncClient, n: int = 50) -> str:
    ids = [f"read-{i}" for i in range(n)]
    r1 = fastq([make_record(rid, b"ACGT" * 10, "/1") for rid in ids])
    r2 = fastq([make_record(rid, b"TGCA" * 10, "/2") for rid in reversed(ids)])
    resp = await upload(client, r1, r2)
    assert resp.status_code == 202, resp.text
    task_id = resp.json()["id"]
    deadline = time.time() + 30
    while time.time() < deadline:
        body = (await client.get(f"/tasks/{task_id}")).json()
        if body["status"] == "succeeded":
            return task_id
        if body["status"] in ("failed", "cancelled"):
            raise AssertionError(f"task ended as {body['status']}: {body['message']}")
        await asyncio.sleep(0.05)
    raise AssertionError("task did not succeed in time")


async def wait_release(client: httpx.AsyncClient, task_id: str, want: str,
                       timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = (await client.get(f"/tasks/{task_id}")).json()
        if body["release_status"] == want:
            return body
        await asyncio.sleep(0.05)
    raise AssertionError(f"task {task_id} did not reach release_status={want}")


async def wait_release_error(client: httpx.AsyncClient, task_id: str,
                             timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = (await client.get(f"/tasks/{task_id}")).json()
        if body["release_status"] == "releasing" and body["release_error"]:
            return body
        await asyncio.sleep(0.05)
    raise AssertionError(f"task {task_id} did not record a release error")


def raw_download(task_id: str, side: int, on_body):
    """Drive the ASGI app directly so the test controls response pacing.

    ``on_body`` is awaited for every http.response.body message; it may block
    (pausing the download) or raise (simulating a client disconnect).
    Returns a coroutine resolving to (status, body_bytes).
    """
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "GET",
        "headers": [],
        "scheme": "http",
        "path": f"/tasks/{task_id}/download/r{side}",
        "raw_path": f"/tasks/{task_id}/download/r{side}".encode(),
        "query_string": b"",
        "server": ("test", 80),
        "client": ("test", 123),
        "root_path": "",
    }
    parts: list[bytes] = []
    status: dict[str, int] = {}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            status["code"] = message["status"]
        elif message["type"] == "http.response.body":
            parts.append(message.get("body", b""))
            await on_body()

    async def run():
        await app(scope, receive, send)
        return status.get("code"), b"".join(parts)

    return run()


async def main_async() -> int:
    tmp_root = Path(tempfile.mkdtemp(prefix="fastq-release-",
                                     dir=str(ROOT / "tests" / "output")))
    data_dir = tmp_root / "data"
    os.environ["FASTQ_DATA_DIR"] = str(data_dir)
    config.DATA_DIR = data_dir
    config.DB_PATH = data_dir / "tasks.db"
    config.TASKS_DIR = data_dir / "tasks"

    db._initialized = False
    db.init_db()
    worker.recover_on_startup()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test",
                                 timeout=60) as client:
        # ---- 1. happy path: release reclaims the disk of a succeeded task
        tid = await make_succeeded(client)
        out1 = config.TASKS_DIR / tid / "output" / "R1.fastq"
        out2 = config.TASKS_DIR / tid / "output" / "R2.fastq"
        record("results exist before release", out1.is_file() and out2.is_file())
        r = await client.post(f"/tasks/{tid}/release")
        record("release accepted with 202", r.status_code == 202,
               str(r.status_code))
        record("acceptance reads as releasing",
               r.json()["release_status"] == "releasing",
               r.json().get("release_status", ""))
        body = await wait_release(client, tid, "released")
        record("release completes to released",
               body["release_status"] == "released")
        record("released task keeps status succeeded",
               body["status"] == "succeeded", body["status"])
        record("result files deleted", not out1.exists() and not out2.exists())
        record("task directory removed", not (config.TASKS_DIR / tid).exists())
        record("released task exposes no download links",
               body["download_r1"] is None and body["download_r2"] is None)
        d1 = await client.get(f"/tasks/{tid}/download/r1")
        d2 = await client.get(f"/tasks/{tid}/download/r2")
        record("downloads rejected after release",
               d1.status_code == 409 and d2.status_code == 409,
               f"{d1.status_code}/{d2.status_code}")
        r = await client.post(f"/tasks/{tid}/release")
        record("re-release of released task is idempotent (200)",
               r.status_code == 200 and r.json()["release_status"] == "released",
               str(r.status_code))

        # ---- 2. in-flight download finishes before cleanup; new ones blocked
        tid = await make_succeeded(client, n=5000)
        out1 = config.TASKS_DIR / tid / "output" / "R1.fastq"
        out2 = config.TASKS_DIR / tid / "output" / "R2.fastq"
        expected = out1.read_bytes()
        first_chunk = asyncio.Event()
        allow_continue = asyncio.Event()

        async def hold_after_first_chunk():
            if not first_chunk.is_set():
                first_chunk.set()
                await allow_continue.wait()

        download = asyncio.create_task(raw_download(tid, 1, hold_after_first_chunk))
        await asyncio.wait_for(first_chunk.wait(), timeout=10)
        record("in-flight download holds a permit",
               worker.DOWNLOADS.active(tid) == 1,
               str(worker.DOWNLOADS.active(tid)))
        r = await client.post(f"/tasks/{tid}/release")
        record("release accepted while download in flight",
               r.status_code == 202 and r.json()["release_status"] == "releasing")
        blocked = await client.get(f"/tasks/{tid}/download/r2")
        record("new download rejected while releasing", blocked.status_code == 409,
               str(blocked.status_code))
        record("rejected download leaves no permit behind",
               worker.DOWNLOADS.active(tid) == 1,
               str(worker.DOWNLOADS.active(tid)))
        await asyncio.sleep(0.5)
        record("cleanup waits for the in-flight download",
               out1.is_file() and out2.is_file())
        body = (await client.get(f"/tasks/{tid}")).json()
        record("query distinguishes releasing from released",
               body["release_status"] == "releasing"
               and body["download_r1"] is None)
        allow_continue.set()
        code, data = await asyncio.wait_for(download, timeout=15)
        record("in-flight download completed intact",
               code == 200 and data == expected,
               f"{code}, {len(data)}/{len(expected)} bytes")
        body = await wait_release(client, tid, "released")
        record("cleanup ran after the download finished",
               not out1.exists() and not out2.exists())
        record("permit drained after streaming", worker.DOWNLOADS.active(tid) == 0)

        # ---- 3. a disconnected download also unblocks the cleanup
        tid = await make_succeeded(client, n=5000)
        out1 = config.TASKS_DIR / tid / "output" / "R1.fastq"
        first_chunk = asyncio.Event()
        release_seen = asyncio.Event()

        async def disconnect_after_release():
            if not first_chunk.is_set():
                first_chunk.set()
                await release_seen.wait()
                raise ConnectionResetError("client went away")

        download = asyncio.create_task(
            raw_download(tid, 1, disconnect_after_release))
        await asyncio.wait_for(first_chunk.wait(), timeout=10)
        r = await client.post(f"/tasks/{tid}/release")
        record("release accepted before disconnect", r.status_code == 202)
        release_seen.set()
        try:
            await asyncio.wait_for(download, timeout=15)
            record("disconnected download raises out of the app", False,
                   "no exception")
        except (ConnectionResetError, asyncio.TimeoutError) as exc:
            record("disconnected download raises out of the app",
                   isinstance(exc, ConnectionResetError))
        await wait_release(client, tid, "released")
        record("cleanup ran after the disconnect", not out1.exists())
        record("permit drained after disconnect",
               worker.DOWNLOADS.active(tid) == 0)

        # ---- 4. failing to delete one end keeps 'releasing'; retry works
        tid = await make_succeeded(client)
        out1 = config.TASKS_DIR / tid / "output" / "R1.fastq"
        out2 = config.TASKS_DIR / tid / "output" / "R2.fastq"
        original_unlink = Path.unlink

        def failing_unlink(self, *args, **kwargs):
            if self == out2:
                raise PermissionError(errno.EACCES, "simulated deletion failure")
            return original_unlink(self, *args, **kwargs)

        Path.unlink = failing_unlink
        try:
            r = await client.post(f"/tasks/{tid}/release")
            record("release accepted before failure", r.status_code == 202)
            body = await wait_release_error(client, tid)
            record("one-end deletion failure stays releasing",
                   body["release_status"] == "releasing")
            record("failure is reported in release_error",
                   "R2" in body["release_error"], body["release_error"])
            record("undeleted end is still on disk", out2.is_file())
            record("other end was deleted", not out1.exists())
            d = await client.get(f"/tasks/{tid}/download/r2")
            record("downloads stay blocked after failed cleanup",
                   d.status_code == 409, str(d.status_code))
        finally:
            Path.unlink = original_unlink
        r = await client.post(f"/tasks/{tid}/release")
        record("repeated release retries the cleanup", r.status_code == 202,
               str(r.status_code))
        body = await wait_release(client, tid, "released")
        record("retry finishes the release",
               not out2.exists() and body["release_error"] == "")

        # ---- 5. a restart resumes an interrupted cleanup
        tid = await make_succeeded(client)
        out1 = config.TASKS_DIR / tid / "output" / "R1.fastq"
        out2 = config.TASKS_DIR / tid / "output" / "R2.fastq"

        def restart_failing_unlink(self, *args, **kwargs):
            if self in (out1, out2):
                raise PermissionError(errno.EACCES, "simulated deletion failure")
            return original_unlink(self, *args, **kwargs)

        Path.unlink = restart_failing_unlink
        try:
            await client.post(f"/tasks/{tid}/release")
            await wait_release_error(client, tid)
            worker.recover_on_startup()  # simulated restart, failure persists
            body = (await client.get(f"/tasks/{tid}")).json()
            record("restart with failing deletion stays releasing",
                   body["release_status"] == "releasing"
                   and out1.is_file() and out2.is_file())
        finally:
            Path.unlink = original_unlink
        worker.recover_on_startup()  # next restart: deletion now works
        body = (await client.get(f"/tasks/{tid}")).json()
        record("restart resumes and finishes the cleanup",
               body["release_status"] == "released"
               and not out1.exists() and not out2.exists(),
               body["release_status"])

        # ---- 6. other tasks are not affected by a release
        keep = await make_succeeded(client)
        drop = await make_succeeded(client)
        keep_r1 = (config.TASKS_DIR / keep / "output" / "R1.fastq").read_bytes()
        await client.post(f"/tasks/{drop}/release")
        await wait_release(client, drop, "released")
        body = (await client.get(f"/tasks/{keep}")).json()
        record("unrelated task still offers downloads",
               body["release_status"] == "none"
               and body["download_r1"] is not None)
        d = await client.get(f"/tasks/{keep}/download/r1")
        record("unrelated task downloads intact",
               d.status_code == 200 and d.content == keep_r1,
               str(d.status_code))

        # ---- 7. only succeeded tasks can be released
        bad = fastq([make_record("x", b"ACGT")])
        mismatch = io.BytesIO(b"@x\nACGT\n+\n~~\n")
        resp = await upload(client, bad, mismatch)
        failed_id = resp.json()["id"]
        deadline = time.time() + 30
        while time.time() < deadline:
            body = (await client.get(f"/tasks/{failed_id}")).json()
            if body["status"] == "failed":
                break
            await asyncio.sleep(0.05)
        r = await client.post(f"/tasks/{failed_id}/release")
        record("release of failed task -> 409", r.status_code == 409,
               str(r.status_code))

        conn = db.connect()
        pending_id = uuid.uuid4().hex
        db.create_task(conn, pending_id, "a", "b", 1, 1)
        cancelled_id = uuid.uuid4().hex
        db.create_task(conn, cancelled_id, "a", "b", 1, 1)
        db.update_task(conn, cancelled_id, status="cancelled",
                       stage=config.STAGE_DONE)
        conn.close()
        r = await client.post(f"/tasks/{pending_id}/release")
        record("release of pending task -> 409", r.status_code == 409,
               str(r.status_code))
        r = await client.post(f"/tasks/{cancelled_id}/release")
        record("release of cancelled task -> 409", r.status_code == 409,
               str(r.status_code))
        r = await client.post(f"/tasks/{uuid.uuid4().hex}/release")
        record("release of unknown task -> 404", r.status_code == 404,
               str(r.status_code))

        # ---- 8. repeated release while releasing is idempotent
        tid = await make_succeeded(client)
        r1 = await client.post(f"/tasks/{tid}/release")
        r2 = await client.post(f"/tasks/{tid}/release")
        record("double release both accepted",
               r1.status_code == 202 and r2.status_code == 202,
               f"{r1.status_code}/{r2.status_code}")
        body = await wait_release(client, tid, "released")
        record("double release settles as released",
               body["release_status"] == "released")

    leaked = [t for t in ("*",) if worker.DOWNLOADS._counts]
    record("no download permits leaked", not leaked,
           str(worker.DOWNLOADS._counts))

    shutil.rmtree(tmp_root, ignore_errors=True)
    failed = [name for name, ok, _ in RESULTS if not ok]
    print("\n%d/%d checks passed" % (len(RESULTS) - len(failed), len(RESULTS)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
