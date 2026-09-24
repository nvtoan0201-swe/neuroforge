# Run: drop0.3_aug_rot_es

- timestamp: 2026-09-23T09:29:12+00:00
- data root: `data/Durian_Leaf_Diseases`
- classes (6): ['Leaf_Algal', 'Leaf_Blight', 'Leaf_Colletotrichum', 'Leaf_Healthy', 'Leaf_Phomopsis', 'Leaf_Rhizoctonia']
- sizes: train 1814 | val 387 | test 394
- image size: 32 | batch size: 32 | seed: 0
- optimizer: Adam (lr=0.001, betas=(0.9, 0.999), eps=1e-08)
- dropout: 0.3
- augment: hflip + crop(pad=4) + rotate90
- early stop: patience=5, min_delta=0.0
- restore best: True

## Model

```
Layer                 Param shape                #Params
Conv2D                (16, 3, 3, 3), (16,)           448
ReLU                  -                                0
MaxPool2D             -                                0
Dropout               -                                0
Conv2D                (32, 16, 3, 3), (32,)         4640
ReLU                  -                                0
MaxPool2D             -                                0
Dropout               -                                0
Flatten               -                                0
Dense                 (2048, 128), (128,)         262272
ReLU                  -                                0
Dropout               -                                0
Dense                 (128, 6), (6,)                 774
Total                                             268134
```

## Epoch history

| epoch | train loss | train acc | val loss | val acc |
|---:|---:|---:|---:|---:|
| 1 | 2.2971 | 0.2073 | 1.7674 | 0.2377 |
| 2 | 1.7526 | 0.2486 | 1.7068 | 0.3049 |
| 3 | 1.7270 | 0.2574 | 1.6724 | 0.3127 |
| 4 | 1.6859 | 0.2850 | 1.6517 | 0.3127 |
| 5 | 1.6516 | 0.2999 | 1.6316 | 0.3643 |
| 6 | 1.5975 | 0.3302 | 1.5843 | 0.3928 |
| 7 | 1.5987 | 0.3412 | 1.5577 | 0.4212 |
| 8 | 1.5530 | 0.3721 | 1.5139 | 0.4625 |
| 9 | 1.5297 | 0.3771 | 1.4930 | 0.4341 |
| 10 | 1.4804 | 0.4068 | 1.4901 | 0.4496 |
| 11 | 1.4803 | 0.4146 | 1.4567 | 0.4806 |
| 12 | 1.4636 | 0.4162 | 1.4389 | 0.4160 |
| 13 | 1.4760 | 0.4201 | 1.4112 | 0.4651 |
| 14 | 1.4480 | 0.4311 | 1.4145 | 0.4599 |
| 15 | 1.4195 | 0.4432 | 1.4179 | 0.4548 |
| 16 | 1.3769 | 0.4603 | 1.3938 | 0.4910 |
| 17 | 1.3802 | 0.4493 | 1.3239 | 0.5013 |
| 18 | 1.3668 | 0.4724 | 1.3379 | 0.5039 |
| 19 | 1.3299 | 0.4757 | 1.3073 | 0.5168 |
| 20 | 1.3343 | 0.4868 | 1.2765 | 0.5659 |

## Result

- best val acc: 0.5659
- best val loss: 1.2765 (epoch 20)
- stopped epoch: 20
- test loss: 1.2014
- test acc: 0.5939
