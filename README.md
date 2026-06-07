# google-patent-kb

Build a semantic search knowledge base over US patents using Google Patents Public Datasets + turbovec.

Query the Google Patents Public Datasets, chunk patents into searchable pieces,
embed on GPU, and serve fast semantic search from a CPU-only machine.

## Architecture

```
BigQuery (patents-public-data)
    │
    │  Query patent full text (free tier)
    ▼
vast.ai GPU instance (temporary, ~$0.30/hr)
    │
    │  1. Stream patent rows from BigQuery
    │  2. Chunk each patent (title, claims, description)
    │  3. Embed all chunks with bge-large-en-v1.5
    │  4. Build turbovec index
    │  5. Save: patent_index.tvim + patent_meta.db
    │
    ▼
Argus (125 GB RAM, Xeon, AVX-512, no GPU)
    │
    │  FastAPI search service
    │  Query: embed on CPU (~50ms) + turbovec search (~ms)
    ▼
    curl /search?q=neural+network+transformer
```

## Usage

```bash
# Test with 1000 patents
python pipeline.py --limit 1000 --output ./data/test

# Search engine patents + ML/NLP overlap
python pipeline.py \
  --cpc G06F16 --cpc G06N --cpc G06F40 \
  --output ./data/search_plus

# If a run is interrupted, start again with a clean output directory.
# --resume currently fails loudly until duplicate-free resume is implemented.

# By date range
python pipeline.py \
  --cpc G06F16 \
  --years 2020 2026 \
  --output ./data/recent

# CPC classes for search technology
#   G06F16  — Information retrieval, search, indexing, ranking
#   G06N    — Machine learning, AI
#   G06F40  — Natural language processing
```

## What you get

| File | Contents |
|---|---|
| `patent_index.tvim` | turbovec vector index with all chunk embeddings |
| `patent_meta.db` | SQLite DB with patent metadata + chunk text |
| `pipeline_checkpoint.json` | Advisory progress checkpoint; safe resume is not implemented yet |

### Metadata schema

```
patents
├── patent_number    TEXT PRIMARY KEY  — "US7603345B2"
├── title           TEXT
├── assignee        TEXT               — "Google LLC"
├── filing_date     TEXT
├── publication_date TEXT
├── ipc_codes       TEXT               — JSON array
├── cpc_codes       TEXT               — JSON array
└── ...

chunks
├── chunk_id        INTEGER PRIMARY KEY
├── patent_number   TEXT → patents
├── chunk_type      TEXT  — "title_abstract", "claim", "claims_all", "description"
├── chunk_label     TEXT  — "Claim 1", "Description §3"
└── chunk_text      TEXT
```

## Query service (runs on argus)

```bash
pip install fastapi uvicorn turbovec sentence-transformers numpy
python query_service.py \
  --index ./data/patent_index.tvim \
  --meta ./data/patent_meta.db

curl 'http://localhost:8080/search?q=spam+detection+documents&k=5'
curl 'http://localhost:8080/search?q=ranking+algorithm&assignee=Google&cpc=G06F16'
curl 'http://localhost:8080/patent/US7603345B2'
curl 'http://localhost:8080/stats'
```

## Deployment

See [DEPLOY.md](DEPLOY.md) for the full guide.

### Quick vast.ai start

```bash
vastai create instance <offer_id> \
  --image pytorch/pytorch:latest \
  --disk 100 \
  --ssh --direct \
  --onstart vast-init.sh

# After boot, upload GCP key and run:
python pipeline.py --cpc G06F16 --cpc G06N --cpc G06F40 --output ./data/search_plus
```

## Requirements

- Python 3.9+
- GCP project with BigQuery API enabled
- ~$5 for vast.ai GPU rental (RTX 4090, ~10 hrs)
- 125 GB RAM server for full US corpus; ~3 GB for search-engine subset

## License

MIT
