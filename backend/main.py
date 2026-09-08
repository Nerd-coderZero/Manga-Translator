import os
import secrets
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import runner
from batch_store import STATUS_DONE, store

SUPPORTED_LANGS = ("zh", "ja")
ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

app = FastAPI(title="Manga Translator API", version="1.0.0")

# CORS
#
# The frontend is served from this same origin in the Hugging Face Space
# deployment, where CORS is not involved at all. It matters for the two
# cases that do cross origins: local development, where the page is opened
# from a file:// URL or a separate static server on another port, and any
# future separately hosted frontend.
#
# The allowlist comes from an environment variable rather than being
# hardcoded so the deployed Space does not have to carry development
# origins. "*" is accepted for local work but is rejected in combination
# with credentials, which is why allow_credentials stays False: this API
# has no cookies or sessions, so nothing needs it, and leaving it off means
# a wildcard origin cannot be turned into a credentialed cross-site read.
_origins_env = os.environ.get("CORS_ALLOW_ORIGINS", "*").strip()
ALLOWED_ORIGINS = ["*"] if _origins_env == "*" else [
    origin.strip() for origin in _origins_env.split(",") if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)


def _validate_lang(source_lang):
    if source_lang not in SUPPORTED_LANGS:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported source_lang '{source_lang}'. supported: {list(SUPPORTED_LANGS)}",
        )


def _validate_upload(upload, data):
    extension = os.path.splitext(upload.filename or "")[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported file type '{extension or upload.filename}'. "
                   f"supported: {list(ALLOWED_EXTENSIONS)}",
        )
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"'{upload.filename}' exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )
    if not data:
        raise HTTPException(status_code=422, detail=f"'{upload.filename}' is empty")


@app.get("/api/health")
def health():
    # reports configuration rather than probing the models. loading
    # paddleocr and manga-ocr takes tens of seconds and allocates most of
    # the container's memory, so a health check that triggered it would be
    # a denial of service against the thing it is meant to be checking.
    return {
        "status": "ok",
        "pipeline_mode": runner.pipeline_mode(),
        "translation_api_key_configured": bool(os.environ.get("NVIDIA_API_KEY")),
        "supported_langs": list(SUPPORTED_LANGS),
        "cors_allow_origins": ALLOWED_ORIGINS,
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "store": store.stats(),
    }


@app.post("/api/translate")
async def translate_single(file: UploadFile = File(...), source_lang: str = Form("ja")):
    _validate_lang(source_lang)
    data = await file.read()
    _validate_upload(file, data)

    work_dir = tempfile.mkdtemp(prefix="single-", dir=store.root)
    extension = os.path.splitext(file.filename)[1].lower()
    input_path = os.path.join(work_dir, f"input{extension}")
    output_path = os.path.join(work_dir, "output.png")
    with open(input_path, "wb") as handle:
        handle.write(data)

    try:
        result = runner.run_page(input_path, output_path, source_lang)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

    token = store.register_image(output_path)
    return {
        "source_lang": source_lang,
        "filename": file.filename,
        "image_token": token,
        "image_url": f"/api/image/{token}",
        "regions": result["regions"],
        "skipped_boxes": result["skipped_boxes"],
    }


@app.get("/api/image/{token}")
def get_image(token: str):
    path = store.image_path(token)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="image not found or expired")
    return FileResponse(path, media_type="image/png")


@app.post("/api/batch", status_code=202)
async def create_batch(files: list[UploadFile] = File(...), source_lang: str = Form("ja")):
    _validate_lang(source_lang)
    if not files:
        raise HTTPException(status_code=422, detail="no files supplied")

    payloads = []
    for upload in files:
        data = await upload.read()
        _validate_upload(upload, data)
        payloads.append((upload.filename, data))

    try:
        batch = store.create_batch(payloads, source_lang)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # 202: the pages have been accepted and queued, none of them are
    # translated yet. the client is expected to poll the status endpoint.
    return {
        "batch_id": batch.id,
        "status_url": f"/api/batch/{batch.id}",
        "total": len(batch.pages),
    }


@app.get("/api/batch/{batch_id}")
def get_batch(batch_id: str):
    batch = store.get_batch(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="batch not found or expired")
    return batch.to_dict()


@app.get("/api/batch/{batch_id}/page/{page_index}")
def get_batch_page(batch_id: str, page_index: int):
    batch = store.get_batch(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="batch not found or expired")
    if page_index < 0 or page_index >= len(batch.pages):
        raise HTTPException(status_code=404, detail="page index out of range")

    page = batch.pages[page_index]
    if page.status != STATUS_DONE:
        raise HTTPException(status_code=409, detail=f"page is not ready (status: {page.status})")

    return {
        "index": page.index,
        "filename": page.filename,
        "image_url": f"/api/image/{page.image_token}",
        "regions": page.regions,
        "duration_seconds": page.duration_seconds,
    }


@app.get("/api/batch/{batch_id}/download")
def download_batch(batch_id: str):
    batch = store.get_batch(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="batch not found or expired")

    buffer = store.build_zip(batch_id)
    if buffer is None:
        raise HTTPException(status_code=409, detail="no completed pages to download")

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="translated_{batch_id}.zip"'},
    )


@app.delete("/api/batch/{batch_id}")
def delete_batch(batch_id: str):
    if not store.delete_batch(batch_id):
        raise HTTPException(status_code=404, detail="batch not found or expired")
    return {"deleted": batch_id}


# static mount is registered last so it cannot shadow any /api route
_frontend_dir = os.environ.get("FRONTEND_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend"
)
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
