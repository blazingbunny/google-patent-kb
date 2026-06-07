# Patent Knowledge Base — Deployment Guide

## Architecture

```
BigQuery (Google Patents Public Datasets)
  │
  │  Step 1: Query → stream patent data
  │  Cost: free (within 1 TB/month tier)
  ▼
vast.ai GPU instance (temporary, ~$5-10 total)
  │  Step 2: Chunk patents
  │  Step 3: Embed all chunks on GPU
  │  Step 4: Build turbovec index
  │  Output: patent_index.tvim (~3 GB) + patent_meta.db (~10-20 GB)
  │
  │  Step 5: scp both files to argus
  ▼
Argus (production, CPU-only)
  ├─ Load index + metadata
  ├─ Serve FastAPI on port 8080
  └─ Query: embed on CPU (~50ms) + turbovec search (~ms)
```

## Step-by-Step

### 0. Prerequisites on your laptop

```bash
# GCP service account with BigQuery access
# 1. Go to https://console.cloud.google.com/apis/credentials
# 2. Create service account → "BigQuery User" role
# 3. Create JSON key → download as bq-key.json

# Vast.ai CLI
pip install vastai
vastai set api-key <your-api-key>
```

### 1. Spin up a vast.ai GPU instance

```bash
# Find a cheap RTX 4090 or RTX 3090 with lots of RAM
vastai search offers "reliability > 0.99 num_gpus=1 gpu_name=RTX_4090 inet_down>200 inet_up>50 disk_space>100"

# Create instance with a PyTorch image
vastai create instance <offer_id> \
  --image pytorch/pytorch:latest \
  --disk 100 \
  --ssh --direct

# Note the SSH command — it looks like:
# ssh -p 12345 root@<instance-ip>
```

### 2. Install dependencies on vast.ai

```bash
ssh -p <port> root@<ip>

# System
apt-get update && apt-get install -y python3-pip git

# Python packages
pip install google-cloud-bigquery turbovec sentence-transformers tqdm numpy

# Upload the pipeline script and GCP key
# From your laptop:
scp -P <port> pipeline.py root@<ip>:~
scp -P <port> bq-key.json root@<ip>:~

# Set auth
export GOOGLE_APPLICATION_CREDENTIALS=/root/bq-key.json
```

### 3. Run the pipeline

```bash
# Test with 1000 patents first
python pipeline.py --limit 1000 --output ./data/test

# First production corpus: search + ML/NLP overlap
# Screen session so it survives disconnection
screen -S patent-pipeline

# Estimate: measure with the pipeline dry run before starting the real job
python pipeline.py \
  --cpc G06F16 --cpc G06N --cpc G06F40 \
  --output ./data/search_plus \
  2>&1 | tee pipeline.log

# Detach: Ctrl+A, D
# Reattach: screen -r patent-pipeline
```

**Estimated runtimes on RTX 4090:**

| Scale | Patents | Chunks | GPU time | Index size | Metadata size |
|---|---|---|---|---|---|
| Test | 1,000 | ~15K | ~3 sec | 12 MB | 5 MB |
| By class (G06F) | ~500K | ~7.5M | ~25 min | 3.6 GB | 1 GB |
| Last 5 years | ~1.5M | ~22M | ~1.2 hr | 11 GB | 3 GB |
| Full US | ~11M | ~165M | ~9 hr | 84 GB | 23 GB |

The index file fits in argus's 125 GB RAM at 4-bit quantization. If you need
2-bit instead, halve the index RAM and speed up ingest by ~2x.

**If interrupted:** start again with a clean output directory unless safe resume
has been implemented and tested. The checkpoint file is advisory progress state;
it is not currently a duplicate-free resume mechanism.

### 4. Deploy to argus

```bash
# On your laptop, once pipeline finishes on vast.ai:
scp -P <port> root@<ip>:./data/search_plus/patent_index.tvim .
scp -P <port> root@<ip>:./data/search_plus/patent_meta.db .

# Upload to argus
scp patent_index.tvim adrian@argus:~/patent_kb/
scp patent_meta.db adrian@argus:~/patent_kb/
scp query_service.py adrian@argus:~/patent_kb/

# Install on argus (CPU inference)
ssh adrian@argus
cd ~/patent_kb
pip install fastapi uvicorn turbovec sentence-transformers numpy

# Test
python query_service.py \
  --index ./patent_index.tvim \
  --meta ./patent_meta.db \
  --model BAAI/bge-large-en-v1.5

# Verify with a search
curl http://localhost:8080/health
curl 'http://localhost:8080/search?q=neural+network+transformer&k=5'
```

### 5. Production service on argus (systemd)

```bash
# Create service file
sudo tee /etc/systemd/system/patent-search.service << 'EOF'
[Unit]
Description=Patent Knowledge Base Search
After=network.target

[Service]
Type=simple
User=adrian
WorkingDirectory=/home/adrian/patent_kb
ExecStart=/home/adrian/.local/bin/uv run python query_service.py \
  --index /home/adrian/patent_kb/patent_index.tvim \
  --meta /home/adrian/patent_kb/patent_meta.db \
  --port 8080
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now patent-search
```

## Usage

```bash
# Search by meaning
curl 'http://argus:8080/search?q=wireless+charging+electric+vehicle&k=5'

# Filter by CPC class + date range
curl 'http://argus:8080/search?q=battery+thermal+management&cpc=H01M&year_start=2020&year_end=2024&k=10'

# Filter to claims only
curl 'http://argus:8080/search?q=neural+network+accelerator&chunk_type=claim&k=20'

# Get full patent details
curl 'http://argus:8080/patent/US7603345B2'

# Search by assignee
curl 'http://argus:8080/search?q=attention+mechanism&assignee=Google&k=5'

# Index stats
curl 'http://argus:8080/stats'
```

## Performance on Argus (Xeon E-2388G, no GPU)

| Operation | Time |
|---|---|
| Embed query (bge-large 1024-dim) | ~50 ms |
| Search 84 GB index | ~2-5 ms |
| Fetch metadata + format response | ~1 ms |
| **End-to-end search** | **~55 ms** |
| Throughput | ~18 req/sec (single process) |

turbovec's AVX-512 kernel runs on the E-2388G (Rocket Lake). The SIMD
search is the fastest part of the pipeline — the CPU embedding is the
bottleneck. For higher throughput, you could:

- Use a smaller embedding model (`bge-base` → 768-dim, ~25 ms per query)
- Pre-embed frequent queries and cache results
- Run multiple uvicorn workers behind nginx

## Cost Summary

| Item | Cost |
|---|---|
| BigQuery query (within 1TB free tier) | $0 |
| vast.ai GPU (RTX 4090, ~10 hrs) | ~$5 |
| Index file transfer to argus (~3 GB) | ~$0 |
| argus runtime (already running) | $0 |
| **Total one-time build** | **~$5** |
| **Ongoing per-query** | **~$0.000001** (negligible CPU) |
