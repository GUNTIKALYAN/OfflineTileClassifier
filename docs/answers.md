# Answers

## A model that is wrong about 30% of the time

I would still ship it. Around 30% wrong on top-1 is ugly if every row is treated as firm. With the cutoff, low-confidence rows stay `uncertain` and firm queries can filter to `accepted` only. This holdout is about 26% wrong. That number is a report, not a bar. I would not keep retraining to chase it.

## How I would notice a month later

I would freeze a small labelled holdout from `candidate_tiles` at deploy time and, on a local timer, rescore it into a health file: holdout accuracy, counts by status, free disk, and the model filename. If the score drops past a margin written down at deploy, the file says degraded. `eval_labels.csv` stays out of that loop. The timer is design only in this slice.

## How I would trace one bad stored row

Check `status` first. Hash the file at `tile_path` and confirm it matches `content_sha256`. Confirm `model_name` is the deployed `artifacts/model.pt`. Compare `probabilities_json` to one fresh run of that same file. Same output means the model is wrong on that tile. Different output means preprocess or weights drifted.

## Weakest part of this one-machine design

One CPU worker and one SQLite file. A burst of uploads waits behind a single inference path. A full disk stops ingest first, before it stops reads of rows already stored.
