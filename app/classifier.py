"""Load the frozen tile CNN and run inference on CPU."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Union

import torch
from PIL import Image

from app.model import TileCNN, image_to_tensor, pil_to_tensor

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.pt"
META_PATH = ROOT / "artifacts" / "meta.json"


class TileClassifier:
    def __init__(self) -> None:
        if not MODEL_PATH.is_file():
            raise SystemExit(f"missing model file: {MODEL_PATH}")
        if not META_PATH.is_file():
            raise SystemExit(f"missing meta file: {META_PATH}")

        self.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        self.classes: list[str] = list(self.meta["classes"])
        self.image_size = int(self.meta["image_size"])
        self.threshold = float(self.meta["threshold"])
        self.preprocess = str(self.meta["preprocess"])
        self.model_name = MODEL_PATH.name

        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        if list(checkpoint["classes"]) != self.classes:
            raise SystemExit("classes in model.pt do not match meta.json")

        self.model = TileCNN(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(torch.device("cpu"))
        self.model.eval()

    def predict(
        self, image_bytes_or_path: Union[bytes, str, Path]
    ) -> tuple[str, float, list[float]]:
        if isinstance(image_bytes_or_path, (str, Path)):
            tensor = image_to_tensor(image_bytes_or_path)
        else:
            with Image.open(io.BytesIO(image_bytes_or_path)) as img:
                tensor = pil_to_tensor(img)

        batch = tensor.unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.model(batch), dim=1)[0]
        values = [float(v) for v in probs.tolist()]
        pred_index = int(probs.argmax().item())
        return self.classes[pred_index], values[pred_index], values


classifier = TileClassifier()
