# M1 part 2: text->image on 200 own 640 m tiles

Labellers: vlm-assisted (visual, blind to model scores). Classes with < 3 positives are excluded from the means.

| model | P@10 | mAP | river P@10 (n=7) | water_body P@10 (n=11) | sea P@10 (n=11) | wetland P@10 (n=22) | salt_pan P@10 (n=5) | farmland P@10 (n=46) | forest P@10 (n=68) | barren P@10 (n=62) | built_up P@10 (n=31) | industrial P@10 (n=6) | road P@10 (n=46) | mining P@10 (n=3) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| remoteclip-vit-b-32 | 0.475 | 0.391 | 0.3 | 0.0 | 0.3 | 0.5 | 0.1 | 0.9 | 0.6 | 0.7 | 1.0 | 0.2 | 0.8 | 0.3 |
| georsclip-vit-b-32 | 0.508 | 0.488 | 0.4 | 0.1 | 0.4 | 0.5 | 0.1 | 0.8 | 1.0 | 0.6 | 1.0 | 0.4 | 0.6 | 0.2 |
| openai-clip-vit-b-32 | 0.458 | 0.372 | 0.2 | 0.3 | 0.1 | 0.4 | 0.0 | 0.8 | 1.0 | 0.9 | 0.8 | 0.2 | 0.6 | 0.2 |
| georsclip-vit-b-32-quickgelu | 0.483 | 0.479 | 0.4 | 0.0 | 0.3 | 0.5 | 0.1 | 0.8 | 1.0 | 0.9 | 0.9 | 0.2 | 0.5 | 0.2 |
| remoteclip-vit-b-32-quickgelu | 0.500 | 0.404 | 0.3 | 0.1 | 0.5 | 0.6 | 0.0 | 0.8 | 0.7 | 0.8 | 1.0 | 0.2 | 0.8 | 0.2 |
| rs-m-clip | 0.392 | 0.294 | 0.2 | 0.0 | 0.2 | 0.3 | 0.0 | 0.8 | 0.8 | 1.0 | 0.7 | 0.1 | 0.5 | 0.1 |
