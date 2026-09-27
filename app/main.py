"""FastAPI service: POST /tiles validates, classifies, and stores one PNG."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from PIL import Image, UnidentifiedImageError

from app import db
from app.classifier import classifier
from app.schemas import TileResult

ROOT = Path(__file__).resolve().parents[1]
TILES_DIR = ROOT / "storage" / "tiles"
TILES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Satellite tile classifier")


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_response(row: dict, status_code: int = 200) -> JSONResponse:
    payload = TileResult(**row).model_dump()
    return JSONResponse(content=payload, status_code=status_code)


@app.post("/tiles")
async def post_tile(file: UploadFile = File(...)) -> JSONResponse:
    raw = await file.read()
    original_filename = file.filename or "upload.bin"
    content_sha256 = hashlib.sha256(raw).hexdigest()

    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.verify()
        with Image.open(io.BytesIO(raw)) as img:
            img.load()
            decoded_ok = True
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        decoded_ok = False
        decode_error = str(exc)

    if not decoded_ok:
        existing = db.get_by_hash(content_sha256)
        if existing is not None:
            return _row_response(existing, status_code=400)
        row = {
            "content_sha256": content_sha256,
            "original_filename": original_filename,
            "tile_path": None,
            "status": "rejected",
            "predicted_class": None,
            "confidence": None,
            "probabilities_json": None,
            "model_name": classifier.model_name,
            "preprocess_version": classifier.preprocess,
            "error": decode_error,
            "created_at": _now(),
        }
        stored = db.insert(row)
        return _row_response(stored, status_code=400)

    existing = db.get_by_hash(content_sha256)
    if existing is not None:
        return _row_response(existing, status_code=200)

    predicted_class, confidence, probabilities = classifier.predict(raw)
    status = "accepted" if confidence >= classifier.threshold else "uncertain"

    tile_path = TILES_DIR / f"{content_sha256}.png"
    with Image.open(io.BytesIO(raw)) as img:
        img.convert("RGB").save(tile_path, format="PNG")

    row = {
        "content_sha256": content_sha256,
        "original_filename": original_filename,
        "tile_path": str(tile_path.relative_to(ROOT)).replace("\\", "/"),
        "status": status,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities_json": json.dumps(probabilities),
        "model_name": classifier.model_name,
        "preprocess_version": classifier.preprocess,
        "error": None,
        "created_at": _now(),
    }
    stored = db.insert(row)
    return _row_response(stored, status_code=200)
