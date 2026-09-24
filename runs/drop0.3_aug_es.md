# Run: drop0.3_aug_es

- timestamp: 2026-09-23T09:25:35+00:00
- data root: `data/Durian_Leaf_Diseases`
- classes (6): ['Leaf_Algal', 'Leaf_Blight', 'Leaf_Colletotrichum', 'Leaf_Healthy', 'Leaf_Phomopsis', 'Leaf_Rhizoctonia']
- sizes: train 1814 | val 387 | test 394
- image size: 32 | batch size: 32 | seed: 0
- optimizer: Adam (lr=0.001, betas=(0.9, 0.999), eps=1e-08)
- dropout: 0.3
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
| 1 | 2.2563 | 0.2095 | 1.7392 | 0.2894 |
| 2 | 1.7249 | 0.2563 | 1.6575 | 0.3902 |
| 3 | 1.6385 | 0.3203 | 1.6063 | 0.3540 |
| 4 | 1.6031 | 0.3423 | 1.5228 | 0.4134 |
| 5 | 1.5656 | 0.3578 | 1.5363 | 0.4238 |
| 6 | 1.5241 | 0.3969 | 1.4835 | 0.4884 |
| 7 | 1.4845 | 0.4041 | 1.4453 | 0.4858 |
| 8 | 1.4485 | 0.4327 | 1.4234 | 0.5556 |
| 9 | 1.4066 | 0.4592 | 1.3433 | 0.5401 |
| 10 | 1.3818 | 0.4603 | 1.3202 | 0.5504 |
| 11 | 1.3233 | 0.4967 | 1.2886 | 0.5194 |
| 12 | 1.3164 | 0.4912 | 1.2733 | 0.5297 |
| 13 | 1.2956 | 0.5022 | 1.2146 | 0.5711 |
| 14 | 1.2575 | 0.5105 | 1.1993 | 0.5891 |
| 15 | 1.2756 | 0.5154 | 1.1886 | 0.5504 |
| 16 | 1.2303 | 0.5309 | 1.1421 | 0.5762 |
| 17 | 1.2092 | 0.5369 | 1.1826 | 0.5943 |
| 18 | 1.1916 | 0.5314 | 1.1176 | 0.5840 |
| 19 | 1.1485 | 0.5673 | 1.1050 | 0.5840 |
| 20 | 1.1525 | 0.5546 | 1.1322 | 0.5866 |

## Result

- best val acc: 0.5943
- best val loss: 1.1050 (epoch 19)
- stopped epoch: 20
- test loss: 1.1151
- test acc: 0.6015
