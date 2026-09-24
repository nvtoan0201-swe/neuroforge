# Kết quả huấn luyện CNN (from scratch)

## Cấu hình chung

- Model: `Conv2D(3→16,k3) → ReLU → MaxPool(2) → Conv2D(16→32,k3) → ReLU → MaxPool(2) → Flatten → Dense(2048→128) → ReLU → Dense(128→6)`
- Tổng tham số: **268,134**
- Dataset `data/Durian_Leaf_Diseases`: train **1814** | val **387** | test **394**, 6 lớp
- Chuẩn hoá mean/std tính trên train; `image-size 32`, `batch-size 32`, `epochs 15`, `seed 0`
- Test set chỉ được đánh giá **1 lần** sau khi train xong (không dùng để cập nhật trọng số)

## Bảng tổng hợp

| Run | Optimizer | lr | momentum | weight-decay | Best val acc | Test loss | Test acc |
|---|---|---|---|---|---|---|---|
| adam_lr1e-3 | Adam | 1e-3 | - | - | **0.6124** (ep 13) | 1.8588 | **0.6066** |
| adam_lr1e-2 | Adam | 1e-2 | - | - | 0.3824 (ep 14) | 2.0405 | 0.3629 |
| sgd_lr1e-2_m0.9 | SGD | 1e-2 | 0.9 | 0.0 | 0.6047 (ep 12) | 2.4275 | 0.5863 |
| sgd_lr1e-2_m0.0 | SGD | 1e-2 | 0.0 | 0.0 | 0.5478 (ep 13) | **1.5190** | 0.5457 |
| sgd_lr1e-2_m0.9_wd1e-4 | SGD | 1e-2 | 0.9 | 1e-4 | 0.5788 (ep 14) | 2.5409 | 0.5736 |

Sắp xếp theo test acc: `adam_lr1e-3` > `sgd_m0.9` > `sgd_m0.9_wd1e-4` > `sgd_m0.0` > `adam_lr1e-2`.

## Nhận xét

- **Adam lr=1e-3 tốt nhất** (test acc 0.6066) và là run duy nhất giảm loss ổn định qua 15 epoch.
- **Adam lr=1e-2 quá cao**: học chậm, chưa hội tụ (train acc chỉ 0.635).
- **SGD momentum=0.9** (0.5863) nhỉnh hơn **momentum=0** (0.5457) về acc, nhưng loss val/test cao hơn và tăng dần → dấu hiệu **overfit** về mặt loss.
- **SGD momentum + weight-decay 1e-4** gần như không cải thiện so với không decay.
- Mọi run đều **overfit nặng** (train acc ~0.96–0.99, val ~0.55–0.61). Đây là giới hạn của CNN nhỏ from-scratch + chưa có augmentation/Dropout/early-stopping.

## Log chi tiết từng run

### adam_lr1e-3

```
optimizer: Adam
epoch   1/15  train loss 2.0512 acc 0.3319  val loss 1.4773 acc 0.4289
epoch   2/15  train loss 1.2454 acc 0.5391  val loss 1.3116 acc 0.4935
epoch   3/15  train loss 0.9335 acc 0.6725  val loss 1.2385 acc 0.5375
epoch   4/15  train loss 0.7057 acc 0.7630  val loss 1.3778 acc 0.5142
epoch   5/15  train loss 0.5221 acc 0.8302  val loss 1.2174 acc 0.5788
epoch   6/15  train loss 0.3494 acc 0.9068  val loss 1.2754 acc 0.5788
epoch   7/15  train loss 0.2450 acc 0.9383  val loss 1.3173 acc 0.5762
epoch   8/15  train loss 0.1310 acc 0.9824  val loss 1.5211 acc 0.5659
epoch   9/15  train loss 0.0852 acc 0.9928  val loss 1.5068 acc 0.5840
epoch  10/15  train loss 0.0465 acc 0.9972  val loss 1.5564 acc 0.5891
epoch  11/15  train loss 0.0245 acc 0.9994  val loss 1.6484 acc 0.5762
epoch  12/15  train loss 0.0182 acc 0.9994  val loss 1.7635 acc 0.5426
epoch  13/15  train loss 0.0196 acc 0.9989  val loss 1.7370 acc 0.6124
epoch  14/15  train loss 0.0173 acc 0.9994  val loss 1.7683 acc 0.5659
epoch  15/15  train loss 0.0231 acc 0.9978  val loss 1.8051 acc 0.6072
test loss 1.8588 acc 0.6066
```

### adam_lr1e-2

```
optimizer: Adam
epoch   1/15  train loss 2.5965 acc 0.1858  val loss 1.7716 acc 0.1912
epoch   2/15  train loss 1.7443 acc 0.2172  val loss 1.7333 acc 0.2119
epoch   3/15  train loss 1.6836 acc 0.2707  val loss 1.6970 acc 0.2687
epoch   4/15  train loss 1.6302 acc 0.3060  val loss 1.6420 acc 0.2791
epoch   5/15  train loss 1.5939 acc 0.3357  val loss 1.6620 acc 0.2713
epoch   6/15  train loss 1.5265 acc 0.3749  val loss 1.6868 acc 0.2791
epoch   7/15  train loss 1.4571 acc 0.4101  val loss 1.7191 acc 0.3127
epoch   8/15  train loss 1.4569 acc 0.4173  val loss 1.7056 acc 0.2946
epoch   9/15  train loss 1.3685 acc 0.4587  val loss 1.8504 acc 0.3463
epoch  10/15  train loss 1.2990 acc 0.4879  val loss 1.7886 acc 0.3618
epoch  11/15  train loss 1.2137 acc 0.5270  val loss 1.8300 acc 0.3359
epoch  12/15  train loss 1.2296 acc 0.5121  val loss 1.8069 acc 0.3643
epoch  13/15  train loss 1.1254 acc 0.5650  val loss 1.9278 acc 0.3669
epoch  14/15  train loss 1.0480 acc 0.5899  val loss 1.9446 acc 0.3824
epoch  15/15  train loss 0.9656 acc 0.6351  val loss 2.0606 acc 0.3618
test loss 2.0405 acc 0.3629
```

### sgd_lr1e-2_m0.9

```
optimizer: SGD
epoch   1/15  train loss 2.1051 acc 0.2602  val loss 1.5494 acc 0.3721
epoch   2/15  train loss 1.4430 acc 0.4355  val loss 1.4221 acc 0.4315
epoch   3/15  train loss 1.2329 acc 0.5364  val loss 1.3536 acc 0.4703
epoch   4/15  train loss 1.0641 acc 0.6125  val loss 1.3489 acc 0.4961
epoch   5/15  train loss 0.9321 acc 0.6588  val loss 1.2169 acc 0.5323
epoch   6/15  train loss 0.7412 acc 0.7277  val loss 1.2490 acc 0.5504
epoch   7/15  train loss 0.5289 acc 0.8093  val loss 1.3804 acc 0.5245
epoch   8/15  train loss 0.4203 acc 0.8495  val loss 1.3931 acc 0.5633
epoch   9/15  train loss 0.2632 acc 0.9096  val loss 1.4656 acc 0.5788
epoch  10/15  train loss 0.1585 acc 0.9487  val loss 1.6788 acc 0.5840
epoch  11/15  train loss 0.0962 acc 0.9746  val loss 2.0866 acc 0.5685
epoch  12/15  train loss 0.0827 acc 0.9757  val loss 2.0891 acc 0.6047
epoch  13/15  train loss 0.0956 acc 0.9708  val loss 2.1280 acc 0.5581
epoch  14/15  train loss 0.0387 acc 0.9923  val loss 2.2806 acc 0.5556
epoch  15/15  train loss 0.0217 acc 0.9972  val loss 2.3353 acc 0.5659
test loss 2.4275 acc 0.5863
```

### sgd_lr1e-2_m0.0

```
optimizer: SGD
epoch   1/15  train loss 1.9354 acc 0.3087  val loss 1.5707 acc 0.3540
epoch   2/15  train loss 1.3797 acc 0.4779  val loss 1.4313 acc 0.4005
epoch   3/15  train loss 1.2287 acc 0.5424  val loss 1.4110 acc 0.4367
epoch   4/15  train loss 1.0864 acc 0.6042  val loss 1.4400 acc 0.4703
epoch   5/15  train loss 0.9545 acc 0.6610  val loss 1.3500 acc 0.4884
epoch   6/15  train loss 0.8598 acc 0.7018  val loss 1.3275 acc 0.4935
epoch   7/15  train loss 0.7399 acc 0.7508  val loss 1.3397 acc 0.4910
epoch   8/15  train loss 0.6570 acc 0.7927  val loss 1.4163 acc 0.4703
epoch   9/15  train loss 0.5820 acc 0.8065  val loss 1.3440 acc 0.4884
epoch  10/15  train loss 0.5016 acc 0.8495  val loss 1.4458 acc 0.5090
epoch  11/15  train loss 0.4296 acc 0.8749  val loss 1.3648 acc 0.5039
epoch  12/15  train loss 0.3541 acc 0.9101  val loss 1.4874 acc 0.4910
epoch  13/15  train loss 0.3062 acc 0.9272  val loss 1.3601 acc 0.5478
epoch  14/15  train loss 0.2489 acc 0.9487  val loss 1.4576 acc 0.5142
epoch  15/15  train loss 0.2099 acc 0.9620  val loss 1.5317 acc 0.5116
test loss 1.5190 acc 0.5457
```

### sgd_lr1e-2_m0.9_wd1e-4

```
optimizer: SGD
epoch   1/15  train loss 2.1043 acc 0.2530  val loss 1.5410 acc 0.3824
epoch   2/15  train loss 1.4373 acc 0.4465  val loss 1.4047 acc 0.4444
epoch   3/15  train loss 1.2475 acc 0.5187  val loss 1.3436 acc 0.4780
epoch   4/15  train loss 1.0547 acc 0.6119  val loss 1.3413 acc 0.5013
epoch   5/15  train loss 0.8934 acc 0.6676  val loss 1.2223 acc 0.5530
epoch   6/15  train loss 0.7398 acc 0.7299  val loss 1.3767 acc 0.5142
epoch   7/15  train loss 0.5123 acc 0.8219  val loss 1.4939 acc 0.5245
epoch   8/15  train loss 0.4076 acc 0.8638  val loss 1.4642 acc 0.5581
epoch   9/15  train loss 0.2695 acc 0.9057  val loss 1.9162 acc 0.5220
epoch  10/15  train loss 0.1990 acc 0.9333  val loss 1.7901 acc 0.5530
epoch  11/15  train loss 0.1057 acc 0.9647  val loss 1.7922 acc 0.5711
epoch  12/15  train loss 0.0738 acc 0.9796  val loss 2.1832 acc 0.5452
epoch  13/15  train loss 0.0662 acc 0.9846  val loss 2.1520 acc 0.5711
epoch  14/15  train loss 0.0330 acc 0.9950  val loss 2.2520 acc 0.5788
epoch  15/15  train loss 0.0288 acc 0.9928  val loss 2.5328 acc 0.5426
test loss 2.5409 acc 0.5736
```
