| run | seed | optimizer | lr | schedule | pool | width | blocks | dropout | augment | tta | epochs run | best val acc | test loss | test acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s32_flat_w16b2 | 0 | adam | 0.001 | cosine | flatten | 16 | 2 | 0 | yes | yes | 25 | 0.6744 | 0.8163 | 0.7030 |
| s32_gap_w16b2 | 0 | adam | 0.001 | cosine | gap | 16 | 2 | 0 | yes | yes | 25 | 0.5607 | 1.0472 | 0.5812 |
| s32_gap_w32b3 | 0 | adam | 0.001 | cosine | gap | 32 | 3 | 0 | yes | yes | 25 | 0.7494 | 0.6777 | 0.7513 |
| s64_gap_w32b3 | 0 | adam | 0.001 | cosine | gap | 32 | 3 | 0 | yes | yes | 30 | 0.8165 | 0.4993 | 0.8249 |
| s96_gap_w32b3 | 0 | adam | 0.001 | cosine | gap | 32 | 3 | 0 | yes | yes | 30 | 0.8682 | 0.4303 | 0.8528 |
| s96_gap_w32b3_rot | 0 | adam | 0.001 | cosine | gap | 32 | 3 | 0 | yes | yes | 30 | 0.8398 | 0.4560 | 0.8426 |
| s96_gap_w32b3_bri | 0 | adam | 0.001 | cosine | gap | 32 | 3 | 0 | yes | yes | 30 | 0.8579 | 0.4317 | 0.8325 |
| s96_gap_w32b3_do0.3 | 0 | adam | 0.001 | cosine | gap | 32 | 3 | 0.3 | yes | yes | 26 | 0.6641 | 0.8317 | 0.6827 |
| s96_gap_w32b3_full | 0 | adam | 0.001 | cosine | gap | 32 | 3 | 0.3 | yes | yes | 28 | 0.6408 | 0.9569 | 0.6294 |
