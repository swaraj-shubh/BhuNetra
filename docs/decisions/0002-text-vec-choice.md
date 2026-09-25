# 0002 text_vec choice: GeoRSCLIP ViT-B-32 (provisional)

**Date:** 2026-09-25 · **Status:** provisional. Final after M1 part 2 (P@10 on 200 own tiles) **and** the licence check.

## Evidence (M1 part 1)
Zero-shot EuroSAT RGB, torchgeo test split (5,400 chips, 64 px = 640 m = one tile, so this exercises the deployed 64 → 224 px resampling). Run on the AWS T4, fully offline. Report: `eval/reports/2026-09-25_bench_clip.{json,md}`.

| model | top-1 (deploy templates) | top-1 (CLIP EuroSAT templates) | text→img P@10 | text→img mAP | img/s (T4) |
|---|---|---|---|---|---|
| **georsclip-vit-b-32-quickgelu** | **0.670** | **0.675** | **0.780** | **0.635** | 250 |
| georsclip-vit-b-32 (GELU, first run) | 0.525 | 0.536 | 0.750 | 0.603 | 264 |
| openai-clip-vit-b-32 (generic baseline) | 0.493 | 0.455 | 0.640 | 0.420 | 257 |
| remoteclip-vit-b-32-quickgelu | 0.378 | 0.337 | 0.550 | 0.354 | 250 |
| remoteclip-vit-b-32 (GELU) | 0.367 | 0.357 | 0.590 | 0.364 | 204 |
| rs-m-clip | 0.301 | 0.277 | 0.330 | 0.210 | 252 |

Chance top-1 = 0.10. Rerun 2026-09-25 with all rows. The first run loaded the OpenAI-derived checkpoints with open_clip's plain `ViT-B-32` (GELU). OpenAI CLIP uses QuickGELU, and with the correct activation GeoRSCLIP's top-1 rises from 0.53 to 0.67. RemoteCLIP is roughly unchanged under either activation and **scores below plain OpenAI CLIP**. RS-specific training helps only for GeoRSCLIP (P@10 0.78 vs 0.64 for the generic model).

## Decision
`configs/models.yaml → text_vec = georsclip-vit-b-32`, arch `ViT-B-32-quickgelu` (`ckpt/RS5M_ViT-B-32.pt`). `features/embed_clip.py` loads it with the same `open_clip` loader as the benchmark.

## Caveats / open
- **Licence: unresolved; ask the authors.** See "Licence audit" below.
- EuroSAT is Europe (2017–18), so India's appearance may differ. **Part 2** (P@10 on 200 labelled 640 m composite tiles from our AOIs) must confirm or overturn this. It needs M2 composites.
- Absolute zero-shot top-1 is 0.67. A small adapter may be justified later; per DESIGN.md §5.5 that happens only if part 2 shows a gap.

## Licence audit (2026-09-25, public sources only; not legal advice)
| model | weights licence as published | code repo | training data | base weights |
|---|---|---|---|---|
| georsclip-vit-b-32 | HF `Zilun/GeoRSCLIP`: `license: cc` with no variant, and no terms in the README | [om-ai-lab/RS5M](https://github.com/om-ai-lab/RS5M): MIT (code) | HF `Zilun/RS5M`: **CC-BY-NC-4.0** | OpenAI CLIP ViT-B-32 |
| remoteclip-vit-b-32 | HF `chendelong/RemoteCLIP`: no licence field, no README | [ChenDelong1999/RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP): Apache-2.0 (code); README silent on the weights | mixed RS datasets | OpenAI CLIP ViT-B-32 |
| rs-m-clip | HF card: MIT | - | - | CLIP + XLM-R |
| openai-clip-vit-b-32 | timm repo: Apache-2.0; [openai/CLIP](https://github.com/openai/CLIP): MIT | | | - |

Findings:
- GeoRSCLIP's weights were trained on a **NonCommercial** dataset, and the `cc` tag probably means CC-BY-NC as well (unconfirmed). SIH is non-commercial, but deployment to MoD may not be. **Email the RS5M authors for written terms** before we call the model deployable.
- RemoteCLIP's weights have **no stated licence**. Under default copyright, that means no grant of rights. Ask the authors too.
- All of these models descend from OpenAI CLIP. OpenAI's [model card](https://github.com/openai/CLIP/blob/main/model-card.md) says "Any deployed use case of the model - whether commercial or not - is currently out of scope". That is usage guidance, not the MIT licence, but switching to a "cleaner" CLIP fine-tune does not escape it. Judges may raise it.
- Fallback if the GeoRSCLIP terms don't allow our use: **plain OpenAI CLIP** (it beats RemoteCLIP and RS-M-CLIP here), unless part 2 shows otherwise. The model is chosen by one line, `configs/models.yaml → text_vec`, plus re-running the embedding step.
