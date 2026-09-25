# 0002 text_vec choice: GeoRSCLIP ViT-B-32 (provisional)

**Date:** 2026-09-25 · **Status:** provisional. Final after M1 part 2 (P@10 on 200 own tiles) **and** the licence check.

## Evidence (M1 part 1)
Zero-shot EuroSAT RGB, torchgeo test split (5,400 chips, 64 px = 640 m = one tile, so this exercises the deployed 64 → 224 px resampling). Run on the AWS T4, fully offline. Report: `eval/reports/2026-09-25_bench_clip.{json,md}`.

| model | top-1 (deploy templates) | top-1 (CLIP EuroSAT templates) | text→img P@10 | text→img mAP | img/s (T4) |
|---|---|---|---|---|---|
| **georsclip-vit-b-32** | **0.525** | **0.536** | **0.750** | **0.603** | 192 |
| remoteclip-vit-b-32 | 0.367 | 0.357 | 0.590 | 0.364 | 156 |
| rs-m-clip | 0.301 | 0.277 | 0.330 | 0.210 | 182 |

Chance top-1 = 0.10.

## Decision
`configs/models.yaml → text_vec = georsclip-vit-b-32` (`ckpt/RS5M_ViT-B-32.pt`). `features/embed_clip.py` loads it with the same `open_clip` loader as the benchmark.

## Caveats / open
- **Licence: VERIFY.** The HF card declares only `license: cc` (no variant). If it is NonCommercial, confirm that SIH/MoD use is acceptable, or fall back to RS-M-CLIP (MIT, weakest here) or RemoteCLIP (licence also undeclared).
- EuroSAT is Europe (2017–18), so India's appearance may differ. **Part 2** (P@10 on 200 labelled 640 m composite tiles from our AOIs) must confirm or overturn this. It needs M2 composites.
- Absolute zero-shot accuracy is modest (0.53). A small adapter may be justified later; per DESIGN.md §5.5 that happens only if part 2 shows a gap.
- No generic (non-RS) CLIP baseline was run. Add OpenAI ViT-B-32 if judges ask "does RS-specific help?".
