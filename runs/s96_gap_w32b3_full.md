# Run: s96_gap_w32b3_full

- timestamp: 2026-09-24T08:55:55+00:00
- data root: `data/Durian_Leaf_Diseases`
- classes (6): ['Leaf_Algal', 'Leaf_Blight', 'Leaf_Colletotrichum', 'Leaf_Healthy', 'Leaf_Phomopsis', 'Leaf_Rhizoctonia']
- sizes: train 1814 | val 387 | test 394
- image size: 96 | batch size: 32 | seeds: [0]
- architecture: width 32, blocks 3, pool gap, batchnorm True, dropout 0.3
- optimizer: Adam (lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
- lr schedule: cosine (min_lr=0, step_size=10, gamma=0.1)
- augment: hflip + crop(pad=4) + rotate90 + brightness(0.2)
- early stop: loss, patience=8, min_delta=0.0 | restore best: True
- tta: True

## Model

```
Layer                 Param shape                #Params
Conv2D                (32, 3, 3, 3), (32,)           896
BatchNorm2D           (32,), (32,)                    64
ReLU                  -                                0
MaxPool2D             -                                0
Dropout               -                                0
Conv2D                (64, 32, 3, 3), (64,)        18496
BatchNorm2D           (64,), (64,)                   128
ReLU                  -                                0
MaxPool2D             -                                0
Dropout               -                                0
Conv2D                (128, 64, 3, 3), (128,)      73856
BatchNorm2D           (128,), (128,)                 256
ReLU                  -                                0
MaxPool2D             -                                0
Dropout               -                                0
GlobalAvgPool2D       -                                0
Dense                 (128, 128), (128,)           16512
ReLU                  -                                0
Dropout               -                                0
Dense                 (128, 6), (6,)                 774
Total                                             110982
```

## Per-seed results

### seed 0

- best val acc: 0.6408
- best val loss: 1.0133 (epoch 20)
- stopped epoch: 28
- test loss: 0.9569
- test acc: 0.6294

| epoch | train loss | train acc | val loss | val acc | lr |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.7500 | 0.3418 | 1.7012 | 0.2739 | 1.00e-03 |
| 2 | 1.4497 | 0.4498 | 1.6641 | 0.3204 | 9.97e-04 |
| 3 | 1.2902 | 0.5160 | 1.5479 | 0.4031 | 9.89e-04 |
| 4 | 1.2139 | 0.5496 | 1.5104 | 0.3850 | 9.76e-04 |
| 5 | 1.1868 | 0.5535 | 1.6094 | 0.3747 | 9.57e-04 |
| 6 | 1.1407 | 0.5772 | 1.3997 | 0.4858 | 9.33e-04 |
| 7 | 1.0592 | 0.6114 | 1.2895 | 0.5323 | 9.05e-04 |
| 8 | 1.0302 | 0.6246 | 1.4050 | 0.4884 | 8.72e-04 |
| 9 | 1.0161 | 0.6406 | 1.2426 | 0.5090 | 8.35e-04 |
| 10 | 0.9894 | 0.6262 | 1.2840 | 0.5375 | 7.94e-04 |
| 11 | 0.9575 | 0.6533 | 1.1966 | 0.5530 | 7.50e-04 |
| 12 | 0.9347 | 0.6566 | 1.0745 | 0.5840 | 7.03e-04 |
| 13 | 0.9256 | 0.6621 | 1.4135 | 0.5220 | 6.55e-04 |
| 14 | 0.8843 | 0.6786 | 1.1902 | 0.5581 | 6.04e-04 |
| 15 | 0.8659 | 0.6786 | 1.1641 | 0.5840 | 5.52e-04 |
| 16 | 0.8657 | 0.6781 | 1.0502 | 0.6176 | 5.00e-04 |
| 17 | 0.8207 | 0.6940 | 1.0717 | 0.6176 | 4.48e-04 |
| 18 | 0.8282 | 0.7106 | 1.0501 | 0.6176 | 3.96e-04 |
| 19 | 0.8151 | 0.7178 | 1.1525 | 0.5917 | 3.45e-04 |
| 20 | 0.7878 | 0.7111 | 1.0133 | 0.6253 | 2.97e-04 |
| 21 | 0.7844 | 0.7244 | 1.1521 | 0.5995 | 2.50e-04 |
| 22 | 0.7761 | 0.7233 | 1.1739 | 0.5762 | 2.06e-04 |
| 23 | 0.7616 | 0.7321 | 1.1292 | 0.5917 | 1.65e-04 |
| 24 | 0.7763 | 0.7260 | 1.0321 | 0.6408 | 1.28e-04 |
| 25 | 0.7659 | 0.7337 | 1.1417 | 0.6072 | 9.55e-05 |
| 26 | 0.7506 | 0.7321 | 1.1267 | 0.6098 | 6.70e-05 |
| 27 | 0.7542 | 0.7343 | 1.1424 | 0.5943 | 4.32e-05 |
| 28 | 0.7367 | 0.7404 | 1.1932 | 0.5866 | 2.45e-05 |
