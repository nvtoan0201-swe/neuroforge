# Run: s32_gap_w16b2

- timestamp: 2026-09-23T10:05:43+00:00
- data root: `data/Durian_Leaf_Diseases`
- classes (6): ['Leaf_Algal', 'Leaf_Blight', 'Leaf_Colletotrichum', 'Leaf_Healthy', 'Leaf_Phomopsis', 'Leaf_Rhizoctonia']
- sizes: train 1814 | val 387 | test 394
- image size: 32 | batch size: 32 | seeds: [0]
- architecture: width 16, blocks 2, pool gap, batchnorm True, dropout 0
- optimizer: Adam (lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
- lr schedule: cosine (min_lr=0, step_size=10, gamma=0.1)
- augment: hflip + crop(pad=4)
- early stop: loss, patience=6, min_delta=0.0 | restore best: True
- tta: True

## Model

```
Layer                 Param shape                #Params
Conv2D                (16, 3, 3, 3), (16,)           448
BatchNorm2D           (16,), (16,)                    32
ReLU                  -                                0
MaxPool2D             -                                0
Conv2D                (32, 16, 3, 3), (32,)         4640
BatchNorm2D           (32,), (32,)                    64
ReLU                  -                                0
MaxPool2D             -                                0
GlobalAvgPool2D       -                                0
Dense                 (32, 128), (128,)             4224
ReLU                  -                                0
Dense                 (128, 6), (6,)                 774
Total                                              10182
```

## Per-seed results

### seed 0

- best val acc: 0.5607
- best val loss: 1.1341 (epoch 24)
- stopped epoch: 25
- test loss: 1.0472
- test acc: 0.5812

| epoch | train loss | train acc | val loss | val acc | lr |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.6887 | 0.2883 | 1.6325 | 0.3411 | 1.00e-03 |
| 2 | 1.5357 | 0.4002 | 1.5332 | 0.4057 | 9.96e-04 |
| 3 | 1.4537 | 0.4399 | 1.5056 | 0.3953 | 9.84e-04 |
| 4 | 1.3896 | 0.4642 | 1.4339 | 0.4367 | 9.65e-04 |
| 5 | 1.3574 | 0.4923 | 1.3651 | 0.4703 | 9.38e-04 |
| 6 | 1.2999 | 0.4972 | 1.3267 | 0.4755 | 9.05e-04 |
| 7 | 1.2884 | 0.5050 | 1.2844 | 0.5065 | 8.64e-04 |
| 8 | 1.2315 | 0.5298 | 1.2530 | 0.5116 | 8.19e-04 |
| 9 | 1.2154 | 0.5436 | 1.2417 | 0.4935 | 7.68e-04 |
| 10 | 1.1856 | 0.5535 | 1.2192 | 0.5116 | 7.13e-04 |
| 11 | 1.1797 | 0.5469 | 1.2109 | 0.5271 | 6.55e-04 |
| 12 | 1.1830 | 0.5628 | 1.2178 | 0.5194 | 5.94e-04 |
| 13 | 1.1545 | 0.5678 | 1.1738 | 0.5323 | 5.31e-04 |
| 14 | 1.1419 | 0.5662 | 1.1749 | 0.5245 | 4.69e-04 |
| 15 | 1.1337 | 0.5678 | 1.1709 | 0.5426 | 4.06e-04 |
| 16 | 1.1272 | 0.5772 | 1.1707 | 0.5426 | 3.45e-04 |
| 17 | 1.1033 | 0.5854 | 1.1598 | 0.5452 | 2.87e-04 |
| 18 | 1.1044 | 0.5788 | 1.1497 | 0.5607 | 2.32e-04 |
| 19 | 1.0935 | 0.5926 | 1.1491 | 0.5530 | 1.81e-04 |
| 20 | 1.0781 | 0.6053 | 1.1410 | 0.5426 | 1.36e-04 |
| 21 | 1.0958 | 0.5915 | 1.1400 | 0.5401 | 9.55e-05 |
| 22 | 1.0820 | 0.6075 | 1.1347 | 0.5452 | 6.18e-05 |
| 23 | 1.0872 | 0.6058 | 1.1342 | 0.5452 | 3.51e-05 |
| 24 | 1.0750 | 0.6047 | 1.1341 | 0.5426 | 1.57e-05 |
| 25 | 1.0901 | 0.5888 | 1.1368 | 0.5323 | 3.94e-06 |
