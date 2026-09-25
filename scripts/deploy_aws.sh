#!/usr/bin/env bash
# Laptop -> AWS box deploy (docs/AWS.md). Staging on the box rewrites manifests/*.yaml (tracked files), which
# blocks push-to-deploy. So: pull the box's manifests into a commit, reset them on the box, then push.
# Staging can rewrite a manifest between those steps (race), so retry a few times.
#   bash scripts/deploy_aws.sh            # from Git Bash on the laptop
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=${HOST:-sih-aws}

git push -q origin main
for attempt in 1 2 3 4 5; do
  for f in manifests/datasets.yaml manifests/models.yaml; do
    scp -q "$HOST:sih_ps2/$f" "$f"
  done
  if ! git diff --quiet -- manifests/; then
    git add manifests/
    git commit -q -m "manifests: sync from AWS box staging"
  fi
  ssh "$HOST" 'cd ~/sih_ps2 && git checkout -- manifests/'
  if git push -q aws main 2>/dev/null; then
    git push -q origin main
    ssh "$HOST" 'cd ~/sih_ps2 && git log --oneline -1'
    exit 0
  fi
  echo "deploy attempt $attempt raced with staging; retrying" >&2
  sleep 3
done
echo "deploy failed after 5 attempts" >&2
exit 1
