# Deployment on Google Cloud

Same layout as [AWS.md](AWS.md): a GPU VM builds the index once, a CPU VM serves it. Prices are approximate on-demand rates; check the console for your region.

## Hosts

| Role | Machine | ≈ cost | Notes |
|---|---|---|---|
| Build | `n1-standard-4` + 1× `nvidia-tesla-t4` (4 vCPU, 15 GB RAM, T4 16 GB), 300 GB `pd-balanced` | ~$0.55/h + disk | Staging and full index build (3.9 h on a T4). Deep Learning VM image (driver, Docker, NVIDIA container toolkit). Delete it once the index has moved. |
| Serve | `e2-standard-4` (4 vCPU, 16 GB RAM), 100 GB `pd-balanced`, Ubuntu 24.04 | ~$100/mo + ~$10 disk + ~$4 IP | Search and review only; query encoding runs on CPU. |

Region: `asia-south1` (Mumbai) is closest to the AOIs and users. Not every zone has T4s; check with
`gcloud compute accelerator-types list --filter="name:nvidia-tesla-t4 AND zone~asia-south1"`.

## One-time project setup

1. Install the CLI and log in: https://cloud.google.com/sdk/docs/install, then `gcloud init`.
2. Billing must be a **paid** account. Free-trial accounts cannot attach GPUs.
3. GPU quota is 0 on new projects. Request it in IAM & Admin → Quotas: `GPUs (all regions)` = 1, and `NVIDIA T4 GPUs` in your region = 1. Approval usually takes minutes to a day.

```bash
PROJECT=<your-project-id>; REGION=asia-south1; ZONE=asia-south1-a
gcloud config set project $PROJECT
gcloud config set compute/zone $ZONE
gcloud services enable compute.googleapis.com
```

## Build VM (GPU)

```bash
# pick a current Deep Learning VM family, e.g. common-cu128-ubuntu-2204-nvidia-570
gcloud compute images list --project deeplearning-platform-release --no-standard-images \
  --filter="family~^common-cu12" --format="value(family)" | sort -u

gcloud compute instances create bhunetra-build \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 --maintenance-policy=TERMINATE \
  --image-family=<family-from-above> --image-project=deeplearning-platform-release \
  --boot-disk-size=300GB --boot-disk-type=pd-balanced \
  --metadata=install-nvidia-driver=True

gcloud compute ssh bhunetra-build
```

On the VM (the first login waits for the driver install):

```bash
nvidia-smi && docker compose version
docker run --rm --gpus all ubuntu nvidia-smi          # container sees the GPU
sudo usermod -aG docker $USER && exit                  # log in again for the group to apply

git clone https://github.com/swaraj-shubh/BhuNetra.git && cd BhuNetra
bash scripts/setup_aws.sh                              # works on any Ubuntu GPU host
cp docker/.env.example docker/.env                     # set POSTGRES_PASSWORD
tmux new -d -s stage 'uv run python scripts/stage_data.py --workers 8 all'
make build && make up && make build-index              # run inside tmux too; ~4 h
```

`--provisioning-model=SPOT` makes the build VM about 60–70% cheaper, but it can be preempted mid-build. Staging resumes, but the index build does not.

## Serve VM (CPU)

```bash
gcloud compute addresses create bhunetra-ip --region=$REGION
gcloud compute firewall-rules create bhunetra-web --network=default \
  --allow=tcp:80,tcp:443 --target-tags=web --source-ranges=0.0.0.0/0

gcloud compute instances create bhunetra \
  --machine-type=e2-standard-4 \
  --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB --boot-disk-type=pd-balanced \
  --address=bhunetra-ip --tags=web

gcloud compute ssh bhunetra
```

On the VM:

```bash
curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER
sudo apt-get install -y make git caddy
git clone https://github.com/swaraj-shubh/BhuNetra.git && cd BhuNetra
cp <build VM's docker/.env> docker/.env                # same password as the dump
```

The `default` network allows SSH (22) from anywhere. To narrow it: `gcloud compute firewall-rules update default-allow-ssh --source-ranges=<your-ip>/32`.

## Moving the index from the build VM

Use the recipe in [AWS.md](AWS.md#moving-a-built-index-to-another-host). Two changes: the repo directory is `BhuNetra`, not `sih_ps2`, and `new` is the serve VM's **internal** IP (`gcloud compute instances list`). Forward your gcloud key so the build VM can reach the serve VM:

```bash
ssh-add ~/.ssh/google_compute_engine
gcloud compute ssh bhunetra-build -- -A
```

If the index already exists on the AWS box, run the same recipe from there (`ssh -A` into the AWS box, target the serve VM's external IP) and skip the build VM entirely.

Then delete the build VM. The disk is billed until you do:
`gcloud compute instances delete bhunetra-build`.

## Public HTTPS

Same Caddy block as [AWS.md](AWS.md#public-https). Point the DNS `A` record at `gcloud compute addresses describe bhunetra-ip --region=$REGION --format='value(address)'`.

## Cost

A stopped VM is not billed for CPU or RAM, but its disk and static IP are. Set a budget alert under Billing → Budgets & alerts.
