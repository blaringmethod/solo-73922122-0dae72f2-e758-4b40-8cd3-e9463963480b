"""FastAPI application: bounded uploads, status, cancellation, downloads,
and disk-release of succeeded results."""

from __future__ import annotations

import sqlite3
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Response, UploadFile, status

from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import config, db, worker
from .worker import UploadTooLarge, start_worker, task_dir


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    worker.recover_on_startup()
    yield


app = FastAPI(
    title="Paired FASTQ Service",
    version="1.1.0",
    description=(
        "Upload shuffled paired-end FASTQ files (R1/R2, uncompressed, "
        "max 256 MiB each). The service re-pairs them by read id, delivers "
        "both mates in R1 order and preserves the original bytes. Succeeded "
        "results can be downloaded until POST /tasks/{id}/release reclaims "
        "their disk space: acceptance blocks new downloads, in-flight "
        "downloads finish first, and the task then reads as released."
    ),
    lifespan=lifespan,
)


class TaskOut(BaseModel):
    id: str
    status: str
    stage: str
    progress: float
    message: str
    r1_name: str
    r2_name: str
    r1_bytes: int
    r2_bytes: int
    release_status: str
    release_error: str
    created_at: str
    updated_at: str
    download_r1: str | None = None
    download_r2: str | None = None


def _serialize(row: sqlite3.Row) -> TaskOut:
    download_r1 = download_r2 = None
    if (
        row["status"] == "succeeded"
        and row["release_status"] == db.RELEASE_NONE
        and row["result_r1"]
    ):
        download_r1 = f"/tasks/{row['id']}/download/r1"
        download_r2 = f"/tasks/{row['id']}/download/r2"
    return TaskOut(
        id=row["id"],
        status=row["status"],
        stage=row["stage"],
        progress=round(row["progress"], 4),
        message=row["message"],
        r1_name=row["r1_name"],
        r2_name=row["r2_name"],
        r1_bytes=row["r1_bytes"],
        r2_bytes=row["r2_bytes"],
        release_status=row["release_status"],
        release_error=row["release_error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        download_r1=download_r1,
        download_r2=download_r2,
    )


async def _spool_bounded(upload: UploadFile, side: str, dest: Path) -> int:
    """Stream an upload to ``dest`` without holding it in memory.

    SpooledTemporaryFile rolls over to a real file after
    ``UPLOAD_SPOOL_THRESHOLD`` and the copy proceeds in fixed chunks; the
    per-file 256 MiB limit is enforced while streaming.
    """
    spool = tempfile.SpooledTemporaryFile(max_size=config.UPLOAD_SPOOL_THRESHOLD)
    total = 0
    try:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > config.MAX_FILE_BYTES:
                raise UploadTooLarge(side, config.MAX_FILE_BYTES)
            spool.write(chunk)
        spool.rollover()
        spool.seek(0)
        with dest.open("wb") as out:
            while True:
                chunk = spool.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    finally:
        spool.close()
    return total


@app.post("/tasks", response_model=TaskOut, status_code=status.HTTP_202_ACCEPTED)
async def create_task(
    r1: UploadFile = File(..., alias="r1", description="R1 FASTQ, uncompressed, <= 256 MiB"),
    r2: UploadFile = File(..., alias="r2", description="R2 FASTQ, uncompressed, <= 256 MiB"),
) -> TaskOut:
    task_id = uuid.uuid4().hex
    base = task_dir(task_id)
    base.mkdir(parents=True, exist_ok=False)
    r1_dest = base / "input_R1.fastq"
    r2_dest = base / "input_R2.fastq"
    try:
        size1 = await _spool_bounded(r1, "R1", r1_dest)
        size2 = await _spool_bounded(r2, "R2", r2_dest)
    except UploadTooLarge as exc:
        worker.cleanup_task_dir(task_id)
        raise HTTPException(
            status_code=getattr(
                status, "HTTP_413_CONTENT_TOO_LARGE",
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            ),
            detail=str(exc),
        )
    except Exception:
        worker.cleanup_task_dir(task_id)
        raise
    finally:
        await r1.close()
        await r2.close()

    conn = db.connect()
    try:
        db.create_task(
            conn,
            task_id,
            r1_name=Path(r1.filename or "r1.fastq").name,
            r2_name=Path(r2.filename or "r2.fastq").name,
            r1_bytes=size1,
            r2_bytes=size2,
        )
        row = db.get_task(conn, task_id)
    finally:
        conn.close()

    worker.register_cancel_event(task_id)
    start_worker(task_id)
    return _serialize(row)


@app.get("/tasks/{task_id}", response_model=TaskOut)
def get_task(task_id: str) -> TaskOut:
    conn = db.connect()
    try:
        row = db.get_task(conn, task_id)
        if row is None:
            raise HTTPException(status_code=404, detail="task not found")
        return _serialize(row)
    finally:
        conn.close()


@app.post("/tasks/{task_id}/cancel", response_model=TaskOut)
def cancel_task(task_id: str) -> TaskOut:
    conn = db.connect()
    try:
        previous = db.try_set_cancelling(conn, task_id)
        if previous is None:
            raise HTTPException(status_code=404, detail="task not found")
        row = db.get_task(conn, task_id)
        if row["status"] in ("succeeded", "failed", "cancelled"):
            # The task reached a terminal state first; tell the caller plainly
            # that it cannot be cancelled.
            raise HTTPException(
                status_code=409,
                detail=f"task is already {row['status']} and cannot be cancelled",
            )
        # Flip the cooperative flag and return without waiting for the worker.
        worker.request_cancel(task_id)
        return _serialize(row)
    finally:
        conn.close()


def _result_path(task_id: str, side: int) -> Path:
    conn = db.connect()
    try:
        row = db.get_task(conn, task_id)
        if row is None:
            raise HTTPException(status_code=404, detail="task not found")
        if row["status"] != "succeeded":
            raise HTTPException(
                status_code=409,
                detail=f"task is {row['status']}; results are not published",
            )
        if row["release_status"] == db.RELEASE_RELEASING:
            raise HTTPException(
                status_code=409,
                detail="release in progress; downloads are disabled",
            )
        if row["release_status"] == db.RELEASE_RELEASED:
            raise HTTPException(
                status_code=409,
                detail="results have been released and are no longer downloadable",
            )
        rel = row["result_r1"] if side == 1 else row["result_r2"]
        if not rel:
            raise HTTPException(status_code=404, detail="result file missing")
    finally:
        conn.close()

    base = task_dir(task_id).resolve()
    path = (base / rel).resolve()
    # Guard against path traversal even though the rel value is server-generated.
    if base not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="result file missing")
    return path


class _TrackedFileResponse(FileResponse):
    """FileResponse that holds a download permit until streaming ends.

    The permit is acquired by the endpoint before the release status is
    checked; it is released here, after the body has been fully sent or the
    client has disconnected — whichever happens first. Only then may the
    release reaper delete the underlying files.
    """

    def __init__(self, *args, task_id: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._task_id = task_id

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            worker.DOWNLOADS.release(self._task_id)


def _download(task_id: str, side: int) -> FileResponse:
    # Acquire the permit BEFORE checking the release status: a release that
    # is committed after this point is guaranteed to see this download and
    # wait for it; a release committed before is seen in _result_path and
    # rejected with 409.
    worker.DOWNLOADS.acquire(task_id)
    try:
        path = _result_path(task_id, side)
        response = _TrackedFileResponse(
            path,
            task_id=task_id,
            media_type="text/plain",
            filename=f"R{side}.fastq",
            headers={"Content-Length": str(path.stat().st_size)},
        )
    except BaseException:
        worker.DOWNLOADS.release(task_id)
        raise
    return response


@app.get("/tasks/{task_id}/download/r1")
def download_r1(task_id: str) -> FileResponse:
    return _download(task_id, 1)


@app.get("/tasks/{task_id}/download/r2")
def download_r2(task_id: str) -> FileResponse:
    return _download(task_id, 2)


@app.post(
    "/tasks/{task_id}/release",
    response_model=TaskOut,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        200: {
            "model": TaskOut,
            "description": "Results were already released; nothing left to do.",
        },
        404: {"description": "Task not found."},
        409: {"description": "Task is not in a succeeded state."},
    },
)
def release_task(task_id: str, response: Response) -> TaskOut:
    """Reclaim the disk space of a succeeded task's paired results.

    Once accepted (202), new R1/R2 downloads are rejected; downloads already
    in progress finish (or disconnect) before both result files are deleted.
    The task then reads as ``release_status='released'``. If deleting either
    end fails, the task stays ``'releasing'`` (see ``release_error``) and a
    repeated POST retries the cleanup; cleanup also resumes automatically
    after a service restart.
    """
    conn = db.connect()
    try:
        row = db.get_task(conn, task_id)
        if row is None:
            raise HTTPException(status_code=404, detail="task not found")
        if row["status"] != "succeeded":
            raise HTTPException(
                status_code=409,
                detail=f"task is {row['status']}; only succeeded tasks can be released",
            )
        outcome = db.try_start_release(conn, task_id)
        row = db.get_task(conn, task_id)
    finally:
        conn.close()

    if outcome == db.RELEASE_RELEASED:
        # Already fully reclaimed; nothing left to do.
        response.status_code = status.HTTP_200_OK
        return _serialize(row)

    worker.request_release_cleanup(task_id)
    return _serialize(row)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})
