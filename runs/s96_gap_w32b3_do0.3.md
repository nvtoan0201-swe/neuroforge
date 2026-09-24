# Run: s96_gap_w32b3_do0.3

- timestamp: 2026-09-24T08:49:32+00:00
- data root: `data/Durian_Leaf_Diseases`
- classes (6): ['Leaf_Algal', 'Leaf_Blight', 'Leaf_Colletotrichum', 'Leaf_Healthy', 'Leaf_Phomopsis', 'Leaf_Rhizoctonia']
- sizes: train 1814 | val 387 | test 394
- image size: 96 | batch size: 32 | seeds: [0]
- architecture: width 32, blocks 3, pool gap, batchnorm True, dropout 0.3
- optimizer: Adam (lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
- lr schedule: cosine (min_lr=0, step_size=10, gamma=0.1)
- augment: hflip + crop(pad=4)
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

- best val acc: 0.6641
- best val loss: 0.9064 (epoch 18)
- stopped epoch: 26
- test loss: 0.8317
- test acc: 0.6827

| epoch | train loss | train acc | val loss | val acc | lr |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.7389 | 0.3451 | 1.6990 | 0.2972 | 1.00e-03 |
| 2 | 1.4204 | 0.4642 | 1.5994 | 0.3618 | 9.97e-04 |
| 3 | 1.2683 | 0.5342 | 1.4493 | 0.4341 | 9.89e-04 |
| 4 | 1.1737 | 0.5700 | 1.6142 | 0.3618 | 9.76e-04 |
| 5 | 1.1605 | 0.5700 | 1.4253 | 0.4393 | 9.57e-04 |
| 6 | 1.0733 | 0.6075 | 1.3222 | 0.4755 | 9.33e-04 |
| 7 | 1.0397 | 0.6224 | 1.3012 | 0.5065 | 9.05e-04 |
| 8 | 1.0120 | 0.6290 | 1.1682 | 0.5504 | 8.72e-04 |
| 9 | 0.9643 | 0.6560 | 1.5967 | 0.4755 | 8.35e-04 |
| 10 | 0.9398 | 0.6643 | 1.2836 | 0.5504 | 7.94e-04 |
| 11 | 0.9003 | 0.6703 | 1.0488 | 0.6098 | 7.50e-04 |
| 12 | 0.9007 | 0.6748 | 1.2494 | 0.5194 | 7.03e-04 |
| 13 | 0.8773 | 0.6797 | 1.1206 | 0.5840 | 6.55e-04 |
| 14 | 0.8253 | 0.7106 | 1.0260 | 0.6227 | 6.04e-04 |
| 15 | 0.8052 | 0.7166 | 1.1683 | 0.5866 | 5.52e-04 |
| 16 | 0.8007 | 0.7023 | 1.0128 | 0.6305 | 5.00e-04 |
| 17 | 0.7660 | 0.7326 | 0.9858 | 0.6589 | 4.48e-04 |
| 18 | 0.7769 | 0.7299 | 0.9064 | 0.6589 | 3.96e-04 |
| 19 | 0.7561 | 0.7420 | 0.9974 | 0.6486 | 3.45e-04 |
| 20 | 0.7297 | 0.7365 | 1.0895 | 0.6176 | 2.97e-04 |
| 21 | 0.7335 | 0.7426 | 1.0111 | 0.6486 | 2.50e-04 |
| 22 | 0.7071 | 0.7552 | 1.0417 | 0.6486 | 2.06e-04 |
| 23 | 0.6838 | 0.7602 | 1.0694 | 0.6227 | 1.65e-04 |
| 24 | 0.6921 | 0.7459 | 1.0528 | 0.6434 | 1.28e-04 |
| 25 | 0.6920 | 0.7607 | 1.0347 | 0.6434 | 9.55e-05 |
| 26 | 0.6720 | 0.7690 | 0.9791 | 0.6641 | 6.70e-05 |
