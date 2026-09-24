# Run: base_adam

- timestamp: 2026-09-23T09:14:51+00:00
- data root: `data/Durian_Leaf_Diseases`
- classes (6): ['Leaf_Algal', 'Leaf_Blight', 'Leaf_Colletotrichum', 'Leaf_Healthy', 'Leaf_Phomopsis', 'Leaf_Rhizoctonia']
- sizes: train 1814 | val 387 | test 394
- image size: 32 | batch size: 32 | seed: 0
- optimizer: Adam (lr=0.001, betas=(0.9, 0.999), eps=1e-08)
- dropout: 0
- augment: none
- early stop: off
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
| 1 | 2.0512 | 0.3319 | 1.4773 | 0.4289 |
| 2 | 1.2454 | 0.5391 | 1.3116 | 0.4935 |
| 3 | 0.9335 | 0.6725 | 1.2385 | 0.5375 |
| 4 | 0.7057 | 0.7630 | 1.3778 | 0.5142 |
| 5 | 0.5221 | 0.8302 | 1.2174 | 0.5788 |
| 6 | 0.3494 | 0.9068 | 1.2754 | 0.5788 |
| 7 | 0.2450 | 0.9383 | 1.3173 | 0.5762 |
| 8 | 0.1310 | 0.9824 | 1.5211 | 0.5659 |
| 9 | 0.0852 | 0.9928 | 1.5068 | 0.5840 |
| 10 | 0.0465 | 0.9972 | 1.5564 | 0.5891 |
| 11 | 0.0245 | 0.9994 | 1.6484 | 0.5762 |
| 12 | 0.0182 | 0.9994 | 1.7635 | 0.5426 |
| 13 | 0.0196 | 0.9989 | 1.7370 | 0.6124 |
| 14 | 0.0173 | 0.9994 | 1.7683 | 0.5659 |
| 15 | 0.0231 | 0.9978 | 1.8051 | 0.6072 |
| 16 | 0.0209 | 0.9989 | 1.8089 | 0.5866 |
| 17 | 0.0109 | 0.9994 | 1.8736 | 0.5814 |
| 18 | 0.0099 | 0.9994 | 1.9023 | 0.5711 |
| 19 | 0.0116 | 0.9989 | 1.9154 | 0.5917 |
| 20 | 0.0161 | 0.9989 | 2.1226 | 0.5685 |

## Result

- best val acc: 0.6124
- best val loss: 1.2174 (epoch 5)
- stopped epoch: 20
- test loss: 1.1334
- test acc: 0.6066
