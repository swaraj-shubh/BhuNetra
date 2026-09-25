#!/usr/bin/env bash
# One-shot setup on the AWS GPU box (Ubuntu, "Deep Learning Base OSS Nvidia Driver GPU AMI"). Idempotent.
#   bash scripts/setup_aws.sh            # from inside the cloned repo
# Then (long-running, resumable; survives SSH disconnects via tmux):
#   tmux new -d -s stage 'uv run python scripts/stage_data.py --workers 8 all 2>&1 | tee -a data/manifests/staging.log'
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "== system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq make tmux git >/dev/null

echo "== uv + Python 3.11"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
grep -q '.local/bin' ~/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
uv python install 3.11
uv sync --frozen            # Linux resolves torch from the cu126 index (pyproject.toml)

echo "== checks"
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible to torch'; print('torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"
uv run ruff check .
uv run pytest -q
df -h . | tail -1
echo "== setup OK. Next: stage data (see header), then: make bench-clip"
