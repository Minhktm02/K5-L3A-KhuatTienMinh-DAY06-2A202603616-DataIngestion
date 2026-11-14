# Day 06 — Implemented Qdrant pipeline

This workspace implements PDF ingestion, chunking, local embeddings, persistent vector storage, metadata filtering, and A/B/C evaluation. The original PostgreSQL + pgvector files are starter material; the active pipeline uses **Qdrant server in Docker**, not PostgreSQL.

## Requirements and setup

The verified environment is Python 3.14, Docker Engine/Compose, and the dependencies in `requirements.txt`. Run these PowerShell commands from the repository root. Create `.venv314` only if it does not already exist:

```powershell
if (-not (Test-Path .venv314)) { py -3.14 -m venv .venv314 }
& .\.venv314\Scripts\python.exe -m pip install -r starter\requirements.txt
if (-not (Test-Path starter\.env)) { Copy-Item starter\.env.example starter\.env }
docker compose -f starter\docker-compose.yml -p day06_qdrant up -d --wait
```

`.env.example` contains only the local Qdrant URL and local embedding model; no API key is required. Keep any local `.env` private.

## Corpus and benchmark

- `data/corpus/world_bank_eca_spring_2021_pages.jsonl` is the normalized snapshot of physical PDF pages 19–48 (30 content pages).
- `data/manifest.json` records the source, CC BY 3.0 IGO attribution, PDF and snapshot hashes, page selection, and section metadata.
- `data/benchmark.json` contains 15 dev and 5 replacement test questions with evidence quotes. It records that the replacement test was prepared after a preliminary run was inspected; see its revision note before interpreting the holdout.
- The pipeline reads the snapshot and does not need the original PDF. To recreate it, provide the licensed source PDF through `WORLD_BANK_PDF` and run `& .\.venv314\Scripts\python.exe -m starter.scripts.prepare_corpus` from the repository root. The script refuses conflicting existing outputs; do not delete corpus files to force a rerun.

## Run tests and A/B/C

```powershell
& .\.venv314\Scripts\python.exe -m unittest discover -s starter\tests -v
```

Start Qdrant as shown above, then direct each run to a **new** results directory to preserve earlier evidence:

```powershell
$env:DAY6_RESULTS_DIR = Join-Path (Get-Location) 'starter\results\my-run'
if (Test-Path $env:DAY6_RESULTS_DIR) { throw 'Choose a new results directory' }
& .\.venv314\Scripts\python.exe -m starter.src.run
```

The run validates benchmark evidence before retrieval, ingests all three strategies, and writes metrics, query latency, top-3 traces, resource statistics, and audit candidates. Current latency measures the Qdrant search call only; query embedding time is not included.

## Storage safety

Qdrant is bound to `127.0.0.1:6333` and persists data in Docker volume `day06_qdrant_data`. To stop it without deleting data:

```powershell
docker compose -f starter\docker-compose.yml -p day06_qdrant stop
```

Do not use `down -v` unless you explicitly intend to delete the persistent vector data. The verified final run is in `results/final_v3/`; preliminary runs are retained separately. The audit file clearly marks subjective labels as AI-assisted and requiring student confirmation.
