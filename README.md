# Satellite tile classifier

One air-gapped machine. Train a small PyTorch CNN on `candidate_tiles`, freeze the weights, and serve them with FastAPI + SQLite. The coded slice is only `POST /tiles`.

## Setup

A virtualenv already exists at `.\venv`. Activate it and install dependencies:

```bash
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Train once

From the project root:

```bash
python scripts/train.py
```

This writes `artifacts/model.pt` and `artifacts/meta.json`. After `model.pt` exists, inference needs no internet.

## Run the API

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

`POST /tiles` with multipart field `file`. Pillow must decode the upload. Content SHA-256 is the identity; a repeat hash returns the stored row without classifying again.

## Offline eval (not part of the API)

```bash
python scripts/score_eval.py
```

Scores `be-mlsys-assignment-dataset/eval_set` against `eval_labels.csv`. The API never imports or opens that CSV.
