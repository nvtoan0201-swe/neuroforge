# Run: drop0.5_aug_es

- timestamp: 2026-09-23T09:32:48+00:00
- data root: `data/Durian_Leaf_Diseases`
- classes (6): ['Leaf_Algal', 'Leaf_Blight', 'Leaf_Colletotrichum', 'Leaf_Healthy', 'Leaf_Phomopsis', 'Leaf_Rhizoctonia']
- sizes: train 1814 | val 387 | test 394
- image size: 32 | batch size: 32 | seed: 0
- optimizer: Adam (lr=0.001, betas=(0.9, 0.999), eps=1e-08)
- dropout: 0.5
- augment: hflip + crop(pad=4)
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
| 1 | 3.0267 | 0.1797 | 1.7880 | 0.2067 |
| 2 | 1.7958 | 0.1797 | 1.7845 | 0.2145 |
| 3 | 1.7938 | 0.1759 | 1.7856 | 0.1990 |
| 4 | 1.7953 | 0.1852 | 1.7848 | 0.1860 |
| 5 | 1.7825 | 0.1974 | 1.7726 | 0.2687 |
| 6 | 1.7727 | 0.2106 | 1.7499 | 0.2972 |
| 7 | 1.7656 | 0.2117 | 1.7455 | 0.2997 |
| 8 | 1.7415 | 0.2343 | 1.7466 | 0.3256 |
| 9 | 1.7489 | 0.2260 | 1.7345 | 0.3152 |
| 10 | 1.7400 | 0.2404 | 1.7086 | 0.3359 |
| 11 | 1.7396 | 0.2255 | 1.7254 | 0.2972 |
| 12 | 1.7396 | 0.2464 | 1.7184 | 0.3514 |
| 13 | 1.7353 | 0.2255 | 1.6964 | 0.3540 |
| 14 | 1.7409 | 0.2381 | 1.7109 | 0.3204 |
| 15 | 1.7126 | 0.2470 | 1.6726 | 0.3773 |
| 16 | 1.7044 | 0.2525 | 1.6731 | 0.3695 |
| 17 | 1.6737 | 0.2872 | 1.6402 | 0.3928 |
| 18 | 1.6532 | 0.2999 | 1.6325 | 0.3876 |
| 19 | 1.6178 | 0.3043 | 1.5942 | 0.4264 |
| 20 | 1.6359 | 0.3197 | 1.6016 | 0.4109 |

## Result

- best val acc: 0.4264
- best val loss: 1.5942 (epoch 19)
- stopped epoch: 20
- test loss: 1.5717
- test acc: 0.4137
