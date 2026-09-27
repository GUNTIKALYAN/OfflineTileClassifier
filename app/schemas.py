"""Pydantic response models for the tile API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class TileResult(BaseModel):
    content_sha256: str
    original_filename: Optional[str] = None
    tile_path: Optional[str] = None
    status: str
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    probabilities_json: Optional[str] = None
    model_name: Optional[str] = None
    preprocess_version: Optional[str] = None
    error: Optional[str] = None
    created_at: str
