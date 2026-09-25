# 0002 text_vec choice: GeoRSCLIP ViT-B-32 (QuickGELU)

**Date:** 2026-09-25, part 2 added 2026-09-26 · **Status:** technically final (both benchmark parts done). **Deployment eligibility pending the licence answer** (see audit). Part 2 labels still need a human spot-check.

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

## Evidence (M1 part 2): 200 of our own 640 m tiles
Sample: seeded (26227), stratified 67/67/66 over dholera/hasdeo/navimumbai, non-monsoon composites (DJF/MAM/ON) only,
no two tiles overlapping, composite `valid_frac` >= VALID_FRAC_EMB. Chips are made exactly as the ingest worker makes them
(`rgb_quicklook` stretch, then each model's own preprocess). 12 multi-label classes, one query each through the deployed
templates. Labels: `eval/labels/m1_own_tiles.csv`, **labelled visually with a vision-language model, blind to model scores**, and committed
(1b4b617) before the benchmark ran. **Not yet reviewed by a human.** Run on CPU because another project on the box held
the GPU. Report: `eval/reports/2026-09-25_bench_own_tiles.{json,md}`.

| model | mAP | mean P@10 |
|---|---|---|
| georsclip-vit-b-32 (GELU) | **0.488** | **0.508** |
| **georsclip-vit-b-32-quickgelu** (deployed) | 0.479 | 0.483 |
| remoteclip-vit-b-32-quickgelu | 0.404 | 0.500 |
| remoteclip-vit-b-32 (GELU) | 0.391 | 0.475 |
| openai-clip-vit-b-32 | 0.372 | 0.458 |
| rs-m-clip | 0.294 | 0.392 |

Reading it:
- **mAP is the headline, not mean P@10.** Four classes have fewer than 10 positives (mining 3, salt_pan 5, industrial 6,
  river 7), which caps their P@10 below 1 for every model.
- **GeoRSCLIP stays first** on our Indian tiles (mAP about 0.48 vs 0.37 for generic OpenAI CLIP), which confirms part 1.
- **GELU vs QuickGELU is a tie here** (0.488 vs 0.479). With 200 tiles that gap is inside the noise (no CI computed). We keep
  **QuickGELU**: it is the architecture the weights were trained with, and part 1 (5,400 chips) shows +0.15 top-1.
- **Every model is weak on water_body, salt_pan, industrial and mining** (P@10 0.0 to 0.4). These are small classes, but they
  matter for the PS example queries. If M4/M5 queries show the same gap, that is the evidence DESIGN.md asks for before
  adding a small adapter (no full fine-tune, §15).
- Of the licence-clean models, **OpenAI CLIP (0.372) beats RS-M-CLIP (0.294)**, so it stays the fallback.

## Decision
`configs/models.yaml → text_vec = georsclip-vit-b-32`, arch `ViT-B-32-quickgelu` (`ckpt/RS5M_ViT-B-32.pt`). `features/embed_clip.py` loads it with the same `open_clip` loader as the benchmark.

## Caveats / open
- **Licence: unresolved; ask the authors.** See "Licence audit" below.
- Part 2 labels are model-generated: a team member should review at least a random 50 (the contact sheets come from `python -m eval.own_tiles sheets`) before any number goes into the report.
- Adapter: part 2 shows a class-specific gap (water_body, salt_pan, industrial, mining), not a general one. Revisit at M4/M5 with real queries.

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
