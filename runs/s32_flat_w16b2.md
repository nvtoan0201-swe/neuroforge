# Run: s32_flat_w16b2

- timestamp: 2026-09-23T10:10:26+00:00
- data root: `data/Durian_Leaf_Diseases`
- classes (6): ['Leaf_Algal', 'Leaf_Blight', 'Leaf_Colletotrichum', 'Leaf_Healthy', 'Leaf_Phomopsis', 'Leaf_Rhizoctonia']
- sizes: train 1814 | val 387 | test 394
- image size: 32 | batch size: 32 | seeds: [0]
- architecture: width 16, blocks 2, pool flatten, batchnorm True, dropout 0
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
Flatten               -                                0
Dense                 (2048, 128), (128,)         262272
ReLU                  -                                0
Dense                 (128, 6), (6,)                 774
Total                                             268230
```

## Per-seed results

### seed 0

- best val acc: 0.6744
- best val loss: 0.8354 (epoch 21)
- stopped epoch: 25
- test loss: 0.8163
- test acc: 0.7030

| epoch | train loss | train acc | val loss | val acc | lr |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.8964 | 0.2762 | 1.5491 | 0.3747 | 1.00e-03 |
| 2 | 1.4918 | 0.4068 | 1.3922 | 0.4806 | 9.96e-04 |
| 3 | 1.3577 | 0.4664 | 1.2358 | 0.5220 | 9.84e-04 |
| 4 | 1.2433 | 0.5237 | 1.1701 | 0.5607 | 9.65e-04 |
| 5 | 1.1720 | 0.5634 | 1.0941 | 0.5788 | 9.38e-04 |
| 6 | 1.1226 | 0.5838 | 1.0156 | 0.6202 | 9.05e-04 |
| 7 | 1.0835 | 0.5921 | 1.0859 | 0.5478 | 8.64e-04 |
| 8 | 1.0230 | 0.6279 | 0.9577 | 0.6202 | 8.19e-04 |
| 9 | 1.0004 | 0.6273 | 0.9425 | 0.6382 | 7.68e-04 |
| 10 | 0.9574 | 0.6362 | 0.9127 | 0.6693 | 7.13e-04 |
| 11 | 0.9101 | 0.6577 | 0.8947 | 0.6460 | 6.55e-04 |
| 12 | 0.9291 | 0.6521 | 0.9009 | 0.6563 | 5.94e-04 |
| 13 | 0.8850 | 0.6670 | 0.8796 | 0.6460 | 5.31e-04 |
| 14 | 0.8761 | 0.6714 | 0.8475 | 0.6537 | 4.69e-04 |
| 15 | 0.8454 | 0.6951 | 0.8688 | 0.6460 | 4.06e-04 |
| 16 | 0.8426 | 0.6814 | 0.8421 | 0.6667 | 3.45e-04 |
| 17 | 0.8137 | 0.6946 | 0.8445 | 0.6641 | 2.87e-04 |
| 18 | 0.8073 | 0.6968 | 0.8643 | 0.6615 | 2.32e-04 |
| 19 | 0.7985 | 0.6963 | 0.8573 | 0.6537 | 1.81e-04 |
| 20 | 0.7835 | 0.7089 | 0.8517 | 0.6589 | 1.36e-04 |
| 21 | 0.8091 | 0.6968 | 0.8354 | 0.6744 | 9.55e-05 |
| 22 | 0.7786 | 0.7194 | 0.8432 | 0.6667 | 6.18e-05 |
| 23 | 0.7939 | 0.7073 | 0.8422 | 0.6693 | 3.51e-05 |
| 24 | 0.7811 | 0.7045 | 0.8401 | 0.6693 | 1.57e-05 |
| 25 | 0.7885 | 0.7051 | 0.8447 | 0.6589 | 3.94e-06 |
