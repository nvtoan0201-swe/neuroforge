# Run: s32_gap_w32b3

- timestamp: 2026-09-23T10:20:43+00:00
- data root: `data/Durian_Leaf_Diseases`
- classes (6): ['Leaf_Algal', 'Leaf_Blight', 'Leaf_Colletotrichum', 'Leaf_Healthy', 'Leaf_Phomopsis', 'Leaf_Rhizoctonia']
- sizes: train 1814 | val 387 | test 394
- image size: 32 | batch size: 32 | seeds: [0]
- architecture: width 32, blocks 3, pool gap, batchnorm True, dropout 0
- optimizer: Adam (lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
- lr schedule: cosine (min_lr=0, step_size=10, gamma=0.1)
- augment: hflip + crop(pad=4)
- early stop: loss, patience=6, min_delta=0.0 | restore best: True
- tta: True

## Model

```
Layer                 Param shape                #Params
Conv2D                (32, 3, 3, 3), (32,)           896
BatchNorm2D           (32,), (32,)                    64
ReLU                  -                                0
MaxPool2D             -                                0
Conv2D                (64, 32, 3, 3), (64,)        18496
BatchNorm2D           (64,), (64,)                   128
ReLU                  -                                0
MaxPool2D             -                                0
Conv2D                (128, 64, 3, 3), (128,)      73856
BatchNorm2D           (128,), (128,)                 256
ReLU                  -                                0
MaxPool2D             -                                0
GlobalAvgPool2D       -                                0
Dense                 (128, 128), (128,)           16512
ReLU                  -                                0
Dense                 (128, 6), (6,)                 774
Total                                             110982
```

## Per-seed results

### seed 0

- best val acc: 0.7494
- best val loss: 0.7205 (epoch 23)
- stopped epoch: 25
- test loss: 0.6777
- test acc: 0.7513

| epoch | train loss | train acc | val loss | val acc | lr |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.5617 | 0.3809 | 1.4628 | 0.4057 | 1.00e-03 |
| 2 | 1.2741 | 0.5226 | 1.2885 | 0.5065 | 9.96e-04 |
| 3 | 1.1418 | 0.5579 | 1.1747 | 0.5401 | 9.84e-04 |
| 4 | 1.0691 | 0.5970 | 1.0868 | 0.5556 | 9.65e-04 |
| 5 | 1.0225 | 0.6213 | 1.0576 | 0.5995 | 9.38e-04 |
| 6 | 0.9747 | 0.6284 | 1.2086 | 0.5426 | 9.05e-04 |
| 7 | 0.9438 | 0.6384 | 1.1279 | 0.5685 | 8.64e-04 |
| 8 | 0.8961 | 0.6659 | 0.9383 | 0.6641 | 8.19e-04 |
| 9 | 0.8478 | 0.6781 | 1.2321 | 0.5065 | 7.68e-04 |
| 10 | 0.8292 | 0.6924 | 0.9702 | 0.6512 | 7.13e-04 |
| 11 | 0.7943 | 0.7172 | 0.9121 | 0.6848 | 6.55e-04 |
| 12 | 0.8096 | 0.7067 | 0.8367 | 0.6848 | 5.94e-04 |
| 13 | 0.7370 | 0.7183 | 0.7924 | 0.7287 | 5.31e-04 |
| 14 | 0.7228 | 0.7310 | 0.7694 | 0.7287 | 4.69e-04 |
| 15 | 0.6841 | 0.7607 | 0.8164 | 0.7183 | 4.06e-04 |
| 16 | 0.6749 | 0.7547 | 0.8194 | 0.6693 | 3.45e-04 |
| 17 | 0.6413 | 0.7574 | 0.7545 | 0.7468 | 2.87e-04 |
| 18 | 0.6440 | 0.7839 | 0.7732 | 0.7390 | 2.32e-04 |
| 19 | 0.6421 | 0.7701 | 0.7467 | 0.7261 | 1.81e-04 |
| 20 | 0.6094 | 0.7839 | 0.7218 | 0.7416 | 1.36e-04 |
| 21 | 0.6148 | 0.7800 | 0.7267 | 0.7364 | 9.55e-05 |
| 22 | 0.5864 | 0.7933 | 0.7263 | 0.7390 | 6.18e-05 |
| 23 | 0.5935 | 0.7938 | 0.7205 | 0.7390 | 3.51e-05 |
| 24 | 0.5788 | 0.7999 | 0.7205 | 0.7468 | 1.57e-05 |
| 25 | 0.5912 | 0.7894 | 0.7205 | 0.7494 | 3.94e-06 |
