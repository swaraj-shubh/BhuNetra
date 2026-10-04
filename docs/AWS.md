# Deployment

The stack runs on a single Linux host. Two setups are used: a GPU box for building the index, and a CPU box for serving.

## Hosts

| Role | Instance | Notes |
|---|---|---|
| Build | g4dn.xlarge (T4 16 GB, 4 vCPU, 16 GB RAM), 300 GB gp3 | Staging, benchmarks, full index build (3.9 h). Deep Learning Base AMI (NVIDIA driver, Docker, nvidia-container-toolkit). |
| Serve | m7i.xlarge (4 vCPU, 16 GB RAM), 100 GB gp3, Ubuntu 24.04 | Search and review only. No GPU needed: query-time encoding runs on CPU. |

On a host without `nvidia-smi` the Makefile adds `docker/compose.cpu.yaml`, which drops the worker's GPU reservation.

Attach an Elastic IP; the public address otherwise changes on every stop/start.

## Fresh host

```bash
git clone https://github.com/Klath123/sih_ps2.git && cd sih_ps2
bash scripts/setup_aws.sh              # uv, Python 3.11, lint + tests (GPU host)
cp docker/.env.example docker/.env     # set POSTGRES_PASSWORD
tmux new -d -s stage 'uv run python scripts/stage_data.py --workers 8 all'
make build && make up && make build-index
```

## Moving a built index to another host

The index does not have to be rebuilt. Copy the image, the data and a database dump:

```bash
C="docker compose --env-file docker/.env -f docker/compose.yaml"
docker save sih26227-api | gzip | ssh new 'gunzip | docker load'
rsync -a --exclude data/benchmarks data models eval docker/.env new:sih_ps2/
$C exec -T db pg_dump -U sih -Fc sih | ssh new 'cat > sih.dump'
# on the new host
make up && $C exec -T db pg_restore -U sih -d sih --clean < ~/sih.dump
$C exec api python -m index.reconcile && make warm    # rebuild Qdrant from PostGIS
```

The new host creates its own export signing key in `data/keys/`, so exports signed on the old host verify only against the old public key.

## Public HTTPS

The app binds to `127.0.0.1:8000`. Caddy terminates TLS (Let's Encrypt) and proxies to it:

```
bhunetra.swarajz.xyz {
	encode zstd gzip
	header /assets/* Cache-Control "public, max-age=31536000, immutable"
	reverse_proxy 127.0.0.1:8000 {
		transport http {
			read_timeout 300s
		}
	}
	request_body {
		max_size 200MB
	}
}
```

1. DNS: `A` record for the host name pointing at the Elastic IP (no proxying, so the HTTP challenge reaches Caddy).
2. Security group: inbound 22 from your IP, 80 and 443 from anywhere.
3. `sudo apt install caddy`, put the block above in `/etc/caddy/Caddyfile`, `sudo systemctl reload caddy`.

For a private deployment add a `basic_auth` block, or enable API keys (`python -m governance.rbac add <name> <role>`). Without either, anyone with the URL can review and export.

Over SSH only, with no public ports:

```bash
ssh -N -L 8000:localhost:8000 <host>   # then open http://localhost:8000
```

## Cost

Stop instances when idle; the EBS volume is still billed. An Elastic IP costs about $3.60 a month.
