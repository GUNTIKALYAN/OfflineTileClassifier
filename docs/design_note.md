# Design note

One air-gapped machine. It accepts a satellite tile, labels the land type with a model that runs on that machine, stores the outcome, and lets an analyst read the stored outcome later. After install, the machine does not need a network.

The land types are Forest, River, Residential, Industrial, AnnualCrop, SeaLake, and Highway. The tiles we have are 64×64 RGB PNGs.

## Components

Five pieces, all local:

1. **Ingest.** `POST /tiles` accepts one PNG. A drop folder can sit in front of the same function later. The request returns when the row is written.
2. **Validator.** Pillow must decode the file. The SHA-256 of the bytes is the identity. The filename is only a name.
3. **Classifier.** A small PyTorch CNN, trained from scratch on `candidate_tiles`, loaded from `artifacts/model.pt` at startup. It runs on CPU and returns seven probabilities. One worker. The model is loaded once.
4. **Store.** The PNG goes under `storage/tiles/<sha256>.png`. One SQLite row goes in `storage/tiles.db`. SQLite is a single file, so there is no database server to keep up on an isolated box.
5. **Query.** A read of finished rows. The coded slice stores rows and returns the row for the tile just submitted. Filters by class, status, confidence, and time are the shape of the query API. They are specified here and not built yet.

`candidate_tiles/` is the only training data. `eval_set/` is scored once, offline, by a script that compares predictions to `eval_labels.csv`. That script is not part of the service. The API never opens the CSV. Training loss is computed on the `candidate_tiles` split. The CSV is an accuracy check, not a loss.

## How a tile moves

1. The file arrives.
2. If Pillow cannot decode it, write status `rejected`, store the error, and stop. The model does not run.
3. Hash the bytes. If that hash already has a row for this model file, return the existing row.
4. Run the CNN. Read the seven probabilities.
5. Compare the top probability with the cutoff in `artifacts/meta.json`.
6. Write the file and the row.
7. Return the row. A later query reads that row and does not run the model again.

Startup loads `model.pt` and `meta.json`. If either file is missing, or the checksum we recorded does not match, the process exits. A changed weights file is a new model version. Old rows keep the model name they were scored with.

## Low confidence

Three options:

- Always store the top class. Simple for the analyst, and a guess of 0.31 looks like a guess of 0.94.
- Drop anything under the cutoff. The analyst never sees a weak call, and the tile disappears, so we cannot tell "unsure" from "never arrived".
- Store the top class, all seven probabilities, and status `uncertain` when the top score is under the cutoff. Firm queries skip those rows unless the analyst asks for them.

I would ship the third. The cutoff starts at 0.55 and is then set from the confidence histogram on a 20% holdout of `candidate_tiles`, then frozen in `meta.json`. There is no eighth class. A scene that is none of the seven should come out uncertain. I would not add an Other label unless we agree the analyst wants one.

A wrong firm label is worse than an uncertain row. Uncertain is a queue for a person. A firm wrong label pollutes whatever map or count they build from the table.

## What to store

Three options:

- The class name only. Small, and later we cannot explain the label or change the cutoff without rescoring.
- Class plus one confidence number. Enough to filter, and not enough to see that River and Highway split the remaining probability.
- The raw file, the hash, the path, the status, the top class, the confidence, all seven probabilities, the model filename, the preprocess name, the error, and timestamps.

I would store the third. Disk cost is trivial at 64×64. The probabilities are what make a cutoff change and a bad-row investigation possible without running the model again.

One row per content hash and model file. Columns: `content_sha256`, `original_filename`, `tile_path`, `status` (`accepted`, `uncertain`, `rejected`), `predicted_class`, `confidence`, `probabilities_json`, `model_name`, `preprocess_version`, `error`, `created_at`.

## What a query means

Two options:

- Query means "classify this tile now". Every read burns CPU, and the answer changes when the weights change.
- Query means "filter rows already stored". Classification happened at ingest. The analyst asks for a class, a status, a minimum confidence, and a time range.

I would use the second. Pending work is a count in a health file, not a row mixed into results. The same pixels submitted twice return the same row.

The endpoint we will actually code is `POST /tiles`: validate, classify, store, return the row. `GET` filters stay in this note.

## When nobody is watching

A timer rescores a small frozen holdout taken from `candidate_tiles` and writes a local health file: model filename, counts by status, free disk, and that holdout accuracy. If the holdout score drops past a margin written down at deploy time, the file says degraded. This timer is design only. It is not in the first coded slice.

`eval_labels.csv` stays out of that loop. Using it as the live monitor would spend the only answer key we have.

If a stored label looks wrong, the order is: status is `accepted`, the file on disk hashes to `content_sha256`, `model_name` is the file we think is deployed, `probabilities_json` matches the stored class, then run that same file once more. The same output means the model is wrong. A different output means the preprocess or the weights drifted. Many tiles failing the same way points at the model. One tile points at that write.

## What the code includes

The running slice is one FastAPI process: `POST /tiles`, the CNN, and SQLite.

Left in this note, on purpose: a durable job queue, pausing ingest when the disk is low, retry then dead-letter, query filters, and the golden-set timer. Those are the first things that break under a burst of uploads or a full disk. The weakest part of what we would code first is the single CPU worker and the single SQLite file. A full disk stops ingest before it stops queries of rows already stored.

## Assumptions

- One machine, CPU only, one inference worker.
- Tiles stay small RGB PNGs, 64×64 unless a file says otherwise. A file that is not a decodable image is rejected.
- The analyst is a person on that machine, filtering stored rows.
- A new model does not overwrite old rows. Rescore is a separate job we have not built.
- The confidence cutoff is ours to choose. The assignment does not give one.
- Accuracy is a reported number, not a bar we have to clear.

## Questions I would ask

- Is an uncertain row useful to the analyst, or do they only want firm labels?
- What volume of tiles per day, and how long must the files be kept?
- Should a model update rescore history, or only new tiles?
- Will input always be 64×64 RGB PNG, or will band count and size vary?
- Who acts on the health file if the box has no network?
