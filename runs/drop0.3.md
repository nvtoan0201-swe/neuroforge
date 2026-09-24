# Run: drop0.3

- timestamp: 2026-09-23T09:18:27+00:00
- data root: `data/Durian_Leaf_Diseases`
- classes (6): ['Leaf_Algal', 'Leaf_Blight', 'Leaf_Colletotrichum', 'Leaf_Healthy', 'Leaf_Phomopsis', 'Leaf_Rhizoctonia']
- sizes: train 1814 | val 387 | test 394
- image size: 32 | batch size: 32 | seed: 0
- optimizer: Adam (lr=0.001, betas=(0.9, 0.999), eps=1e-08)
- dropout: 0.3
- augment: none
- early stop: off
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
| 1 | 2.2946 | 0.2266 | 1.6826 | 0.3437 |
| 2 | 1.6801 | 0.3076 | 1.5883 | 0.3979 |
| 3 | 1.5555 | 0.3660 | 1.5065 | 0.4496 |
| 4 | 1.4629 | 0.4173 | 1.4879 | 0.4470 |
| 5 | 1.4090 | 0.4531 | 1.3894 | 0.5220 |
| 6 | 1.3140 | 0.4884 | 1.3298 | 0.5194 |
| 7 | 1.2134 | 0.5375 | 1.2916 | 0.5375 |
| 8 | 1.1865 | 0.5540 | 1.2815 | 0.5401 |
| 9 | 1.1129 | 0.5827 | 1.2349 | 0.5452 |
| 10 | 1.0645 | 0.5888 | 1.1545 | 0.5814 |
| 11 | 1.0033 | 0.6279 | 1.1208 | 0.5788 |
| 12 | 0.9801 | 0.6279 | 1.1138 | 0.5659 |
| 13 | 0.9390 | 0.6428 | 1.0961 | 0.5917 |
| 14 | 0.9024 | 0.6654 | 1.1009 | 0.5814 |
| 15 | 0.8785 | 0.6731 | 1.0580 | 0.5891 |
| 16 | 0.8390 | 0.6863 | 1.0517 | 0.5711 |
| 17 | 0.8111 | 0.6924 | 1.0380 | 0.6047 |
| 18 | 0.7635 | 0.7056 | 1.0318 | 0.6124 |
| 19 | 0.7322 | 0.7205 | 1.0287 | 0.6124 |
| 20 | 0.7329 | 0.7244 | 1.0285 | 0.6021 |

## Result

- best val acc: 0.6124
- best val loss: 1.0285 (epoch 20)
- stopped epoch: 20
- test loss: 0.9442
- test acc: 0.6523
