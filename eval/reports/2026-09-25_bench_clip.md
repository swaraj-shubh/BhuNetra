# M1 zero-shot EuroSAT (5400 test images)

| model | top1_deploy_templates | top1_clip_eurosat_templates | text2img_P@10 | text2img_mAP | img_per_s |
|---|---|---|---|---|---|
| remoteclip-vit-b-32 | 0.367 | 0.357 | 0.590 | 0.364 | 203.917 |
| georsclip-vit-b-32 | 0.525 | 0.536 | 0.750 | 0.603 | 263.765 |
| openai-clip-vit-b-32 | 0.493 | 0.455 | 0.640 | 0.420 | 256.976 |
| georsclip-vit-b-32-quickgelu | 0.670 | 0.675 | 0.780 | 0.635 | 250.346 |
| remoteclip-vit-b-32-quickgelu | 0.378 | 0.337 | 0.550 | 0.354 | 250.262 |
| rs-m-clip | 0.301 | 0.277 | 0.330 | 0.210 | 251.631 |
