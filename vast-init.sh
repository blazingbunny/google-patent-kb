#!/usr/bin/env bash
# vast.ai startup script — runs once on first boot
#
# Usage:
#   vastai create instance <offer_id> \
#     --image pytorch/pytorch:latest \
#     --disk 100 \
#     --ssh --direct \
#     --onstart vast-init.sh
#
# Or upload this as the "onstart" script in vast.ai web UI.
# The repo is cloned automatically, then you SSH in and run the pipeline.

set -euo pipefail

echo "[vast-init] Starting google-patent-kb setup..."

REPO_URL="https://github.com/blazingbunny/google-patent-kb.git"
WORK_DIR="/root/google-patent-kb"
BQ_KEY_PATH="/root/bq-key.json"

# ------------------------------------------------------------------
# 1. System dependencies
# ------------------------------------------------------------------
apt-get update -qq
apt-get install -y -qq \
    python3-pip \
    python3-dev \
    git \
    screen \
    wget \
    curl

# ------------------------------------------------------------------
# 2. Clone the repo
# ------------------------------------------------------------------
if [ ! -d "$WORK_DIR" ]; then
    git clone "$REPO_URL" "$WORK_DIR"
fi
cd "$WORK_DIR"

# ------------------------------------------------------------------
# 3. Python dependencies
# ------------------------------------------------------------------
pip install --quiet --upgrade pip
pip install --quiet \
    google-cloud-bigquery \
    turbovec \
    sentence-transformers \
    tqdm \
    numpy

echo "Python packages installed."

# ------------------------------------------------------------------
# 4. Verify GPU is available for sentence-transformers
# ------------------------------------------------------------------
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# ------------------------------------------------------------------
# 5. Print instructions
# ------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  Patent KB — Ready"
echo "================================================================"
echo ""
echo "  Next steps:"
echo "  1. Upload your GCP service account key:"
echo "     scp -P \$PORT bq-key.json root@\$HOST:/root/"
echo ""
echo "  2. Run the pipeline:"
echo "     export GOOGLE_APPLICATION_CREDENTIALS=/root/bq-key.json"
echo ""
echo "     # Test (1000 patents)"
echo "     python pipeline.py --limit 1000 --output /root/data/test"
echo ""
echo "     # Full search engine patent corpus"
echo "     screen -S patent-pipeline"
echo "     python pipeline.py --output /root/data/final \\"
echo "       --cpc G06F16 --cpc G06N --cpc G06F40 \\"
echo "       2>&1 | tee pipeline.log"
echo ""
echo "  3. When done, scp the index + metadata to argus:"
echo "     scp /root/data/final/patent_index.tvim adrian@argus:~/patent_kb/"
echo "     scp /root/data/final/patent_meta.db adrian@argus:~/patent_kb/"
echo ""
echo "================================================================"
