# Run: aug_es

- timestamp: 2026-09-23T09:21:59+00:00
- data root: `data/Durian_Leaf_Diseases`
- classes (6): ['Leaf_Algal', 'Leaf_Blight', 'Leaf_Colletotrichum', 'Leaf_Healthy', 'Leaf_Phomopsis', 'Leaf_Rhizoctonia']
- sizes: train 1814 | val 387 | test 394
- image size: 32 | batch size: 32 | seed: 0
- optimizer: Adam (lr=0.001, betas=(0.9, 0.999), eps=1e-08)
- dropout: 0
- augment: hflip + crop(pad=4)
- early stop: patience=5, min_delta=0.0
- restore best: True

## Model

```
Layer                 Param shape                #Params
Conv2D                (16, 3, 3, 3), (16,)           448
ReLU                  -                                0
MaxPool2D             -                                0
Conv2D                (32, 16, 3, 3), (32,)         4640
ReLU                  -                                0
MaxPool2D             -                                0
Flatten               -                                0
Dense                 (2048, 128), (128,)         262272
ReLU                  -                                0
Dense                 (128, 6), (6,)                 774
Total                                             268134
```

## Epoch history

| epoch | train loss | train acc | val loss | val acc |
|---:|---:|---:|---:|---:|
| 1 | 2.0719 | 0.2911 | 1.5451 | 0.3876 |
| 2 | 1.4802 | 0.4184 | 1.4038 | 0.4315 |
| 3 | 1.3577 | 0.4653 | 1.2715 | 0.4987 |
| 4 | 1.2376 | 0.5160 | 1.2162 | 0.5323 |
| 5 | 1.1910 | 0.5524 | 1.1812 | 0.5426 |
| 6 | 1.1426 | 0.5617 | 1.0610 | 0.5969 |
| 7 | 1.0581 | 0.6119 | 1.0511 | 0.5917 |
| 8 | 1.0202 | 0.6273 | 1.0608 | 0.5969 |
| 9 | 0.9925 | 0.6356 | 1.0383 | 0.5891 |
| 10 | 0.9331 | 0.6566 | 0.9862 | 0.6279 |
| 11 | 0.9023 | 0.6582 | 0.9608 | 0.6253 |
| 12 | 0.9122 | 0.6582 | 1.0251 | 0.6047 |
| 13 | 0.8597 | 0.6736 | 0.9552 | 0.6331 |
| 14 | 0.8545 | 0.6830 | 0.9516 | 0.6331 |
| 15 | 0.8231 | 0.6891 | 0.9432 | 0.6434 |
| 16 | 0.7944 | 0.6951 | 0.9335 | 0.6641 |
| 17 | 0.7649 | 0.7194 | 0.9507 | 0.6770 |
| 18 | 0.7443 | 0.7166 | 0.9697 | 0.6512 |
| 19 | 0.7585 | 0.7106 | 0.8964 | 0.6512 |
| 20 | 0.6957 | 0.7354 | 0.9073 | 0.6873 |

## Result

- best val acc: 0.6873
- best val loss: 0.8964 (epoch 19)
- stopped epoch: 20
- test loss: 0.8913
- test acc: 0.6777
