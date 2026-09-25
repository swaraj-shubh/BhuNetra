# AWS GPU box: configuration and runbook

## 0. The box we actually run (2026-09-25)
| | |
|---|---|
| Instance | `i-09ea6dd470482c3b3` **g4dn.xlarge**: Tesla T4 16 GB, 4 vCPU, 15 GB RAM · **ap-south-1 (Mumbai)** |
| AMI | Deep Learning Base AMI with Single CUDA (Ubuntu 24.04) 20260922 · driver 595.91 · Docker 29.8 |
| Disk | 290 GB root (gp3) |
| Access | laptop `~/.ssh/config` host **`sih-aws`** (key `~/.ssh/sih.pem`, user `ubuntu`). **The public IP changes on every stop/start**: update `HostName` (or attach an Elastic IP). |
| Code | push-to-deploy: laptop remote **`aws`** → `~/sih_ps2` on the box (`receive.denyCurrentBranch updateInstead`). `git push aws main` updates the box's working tree. GitHub `origin` stays the source of truth. |
| Jobs | `tmux` sessions on the box: `stage` (data), `bench` (M1). `ssh sih-aws tmux ls` · attach: `ssh -t sih-aws tmux attach -t stage` |

**Results produced on the box** (e.g. `eval/reports/*`) come back to the laptop with `scp sih-aws:sih_ps2/<path> <path>`, get committed there, and the box copy is moved aside **before** the next `git push aws main`. Otherwise the push is refused because an untracked file would be overwritten.

Mumbai vs Oregon: Sentinel-2 COG reads cross regions (a bit slower, no cost to us because the bucket is open data). Everything else is unchanged.

Why AWS: the dev laptop (8 GB RAM, no CUDA) kept killing long jobs. All staging, benchmarks, training and the offline demo stack run on this box.

## 1. Instance configuration

| Setting | Value | Why |
|---|---|---|
| **Region** | **us-west-2 (Oregon)** | The Sentinel-2 COGs (Earth Search, `sentinel-cogs` bucket) live in us-west-2, so reads are fastest there and in-region transfer is free. S1/Landsat come from Planetary Computer (Azure, internet), which works from any region. |
| **Instance type** | **g6.2xlarge**: 1× NVIDIA L4 24 GB, 8 vCPU, 32 GB RAM | 24 GB VRAM covers Prithvi-EO-2.0-300M, the CLIP models and change-model fine-tuning (DESIGN.md compute tier "training workstation"). 32 GB RAM avoids the OOM kills. |
| Alternatives | g5.2xlarge (A10G 24 GB, same RAM); budget: g4dn.xlarge (T4 16 GB, 4 vCPU, 16 GB RAM), fine for inference/benchmarks, tight for training | Prices change. Check the current on-demand price for your region on the EC2 pricing page before launching. |
| **AMI** | *Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*, latest | NVIDIA driver, Docker and nvidia-container-toolkit preinstalled. We install our own Python 3.11 + CUDA torch via `uv`. Pick the newest dated version in the AMI catalog (**# VERIFY** the exact name at launch). |
| **Storage** | root volume **gp3, 300 GB** (default 3000 IOPS / 125 MB/s) | 3 AOIs × S2+S1+Landsat 2019–2026 ≈ 25–35 GB, models ≈ 7 GB, Docker images + venv ≈ 20 GB, plus room for ARD/composites. |
| **Key pair** | New key pair `sih-gpu`, type **ED25519**, format **.pem** → save as `C:\Users\kesha\.ssh\sih-gpu.pem` | Referenced from `~/.ssh/config`. |
| **Security group** | Inbound: **SSH (22) from "My IP" only**. Nothing else. | The API/UI are reached through an SSH tunnel (`ssh -L 8000:localhost:8000 ...`), never exposed publicly. |
| IAM role | none needed | All data sources are public/anonymous. |

**Before launching: GPU quota.** New AWS accounts often have **0** vCPUs allowed for GPU instances. Go to *Service Quotas → Amazon EC2 → "Running On-Demand G and VT instances"* and request **≥ 8** (g6.2xlarge = 8 vCPU) in us-west-2. Approval can take from minutes to a day.

**Cost control:** *Stop* the instance whenever it's idle (you then pay only for the EBS volume). Set an AWS Budgets alert. Don't *terminate* the instance unless the data is no longer needed; that deletes the volume.

## 2. What you do (console, about 10 minutes)
1. Request the GPU quota (above), region us-west-2.
2. EC2 → Launch instance: settings from the table. Download `sih-gpu.pem` into `C:\Users\kesha\.ssh\`.
3. Note the instance's **Public IPv4 address** (username is `ubuntu`).

## 3. Box setup (over SSH from the laptop)
1. `ssh -i ~/.ssh/sih-gpu.pem ubuntu@<ip>`, checks `nvidia-smi`.
2. Generates a **read-only deploy key** on the box and shows its public key. You add it at GitHub → `Klath123/sih_ps2` → Settings → Deploy keys (leave "Allow write access" unchecked).
3. `git clone git@github.com:Klath123/sih_ps2.git` → `bash scripts/setup_aws.sh` (uv, Python 3.11, CUDA torch, lint + tests).
4. Staging in `tmux` (survives disconnects): `uv run python scripts/stage_data.py --workers 8 all`.
5. `make bench-clip` on the GPU → M1 table + ADR. Then M0 proof (`make up-offline`, Docker is on the AMI) and M2 onward.

## 4. Everyday commands (from the laptop)
```bash
ssh -i ~/.ssh/sih-gpu.pem ubuntu@<ip>                       # shell
ssh -i ~/.ssh/sih-gpu.pem ubuntu@<ip> 'tmux ls'             # running jobs
ssh -i ~/.ssh/sih-gpu.pem -L 8000:localhost:8000 ubuntu@<ip> # then open http://localhost:8000 locally
```
