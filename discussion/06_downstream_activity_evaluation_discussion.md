# 06 Downstream Activity Detection Discussion

## 1. Purpose

This notebook evaluates the downstream utility of the synthetic KoVAE data.

The main question is:

```text
Does adding synthetic KoVAE data improve human activity recognition on unseen real test subjects?
```

The classifier used in this final evaluation was:

```text
aeon MiniRocketClassifier
```

The modality used was:

```text
fused = ACC_x + ACC_y + ACC_z + BVP + EDA + TEMP
```

For fusion, all channels were represented with the same temporal length so that the classifier could use a single multi-channel time-series input.

---

## 2. Evaluation design

The final evaluation design contains four experiment types:

```text
1. train on 10 real subjects -> test on 3 real subjects
2. train on 10 real subjects -> test on 3 synthetic subjects
3. train on 10 synthetic subjects -> test on 3 real subjects
4. train on 10 real + 10 synthetic subjects -> test on 3 real subjects
```

The synthetic test setting used the first 3 synthetic subjects.

The two synthetic generation methods were:

```text
KoVAE-Rollout
KoVAE-Posterior
```

---

## 3. Dataset split summary

| dataset                                    |   num_windows | subjects                                                                                                                                                                                                          |   activity_1_windows |   activity_2_windows |   activity_3_windows |   activity_4_windows |   activity_5_windows |   activity_6_windows |   activity_7_windows |   activity_8_windows |
|:-------------------------------------------|--------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|
| real_train                                 |         30762 | S1,S2,S3,S4,S5,S6,S9,S11,S12,S13                                                                                                                                                                                  |                 3032 |                 2139 |                 1500 |                 2296 |                 4580 |                 8792 |                 2903 |                 5520 |
| real_val                                   |          6100 | S14,S15                                                                                                                                                                                                           |                  605 |                  432 |                  335 |                  449 |                  866 |                 1581 |                  632 |                 1200 |
| real_test                                  |         10063 | S7,S8,S10                                                                                                                                                                                                         |                  901 |                  635 |                  444 |                  697 |                 1363 |                 3147 |                 1128 |                 1748 |
| synthetic_train_all10_rollout_v1           |         30000 | synthetic_subject_01,synthetic_subject_02,synthetic_subject_03,synthetic_subject_04,synthetic_subject_05,synthetic_subject_06,synthetic_subject_07,synthetic_subject_08,synthetic_subject_09,synthetic_subject_10 |                 2900 |                 2051 |                 1397 |                 2190 |                 4616 |                 8628 |                 2756 |                 5462 |
| synthetic_train_all10_posterior_bank_v2    |         30000 | synthetic_subject_01,synthetic_subject_02,synthetic_subject_03,synthetic_subject_04,synthetic_subject_05,synthetic_subject_06,synthetic_subject_07,synthetic_subject_08,synthetic_subject_09,synthetic_subject_10 |                 2951 |                 2108 |                 1443 |                 2204 |                 4400 |                 8778 |                 2779 |                 5337 |
| synthetic_test_3subjects_rollout_v1        |          9000 | synthetic_subject_01,synthetic_subject_02,synthetic_subject_03                                                                                                                                                    |                  898 |                  589 |                  417 |                  677 |                 1437 |                 2488 |                  862 |                 1632 |
| synthetic_test_3subjects_posterior_bank_v2 |          9000 | synthetic_subject_01,synthetic_subject_02,synthetic_subject_03                                                                                                                                                    |                  851 |                  624 |                  420 |                  676 |                 1311 |                 2694 |                  863 |                 1561 |

The real test set contains the held-out real subjects. These subjects were not used during KoVAE training.

---

## 4. Important implementation note: near-flat windows

MiniRocket/Rocket failed when some individual case/channel pairs had almost zero standard deviation. This was handled by dropping only the near-flat windows from the temporary aeon classifier arrays.

This did not modify the saved real or synthetic data.

Dropped-window summary:

| experiment                                     |   train_windows_original_before_aeon_filter |   train_windows_dropped_by_aeon_filter |   test_windows_original_before_aeon_filter |   test_windows_dropped_by_aeon_filter |   train_windows |   test_windows | train_drop_percent   | test_drop_percent   |
|:-----------------------------------------------|--------------------------------------------:|---------------------------------------:|-------------------------------------------:|--------------------------------------:|----------------:|---------------:|:---------------------|:--------------------|
| real_to_real                                   |                                       30762 |                                    661 |                                      10063 |                                   125 |           30101 |           9938 | 2.15%                | 1.24%               |
| synthetic_to_real__rollout_v1                  |                                       30000 |                                      0 |                                      10063 |                                   125 |           30000 |           9938 | 0.00%                | 1.24%               |
| real_plus_synthetic_to_real__rollout_v1        |                                       60762 |                                    661 |                                      10063 |                                   125 |           60101 |           9938 | 1.09%                | 1.24%               |
| real_to_3synthetic__rollout_v1                 |                                       30762 |                                    661 |                                       9000 |                                     0 |           30101 |           9000 | 2.15%                | 0.00%               |
| synthetic_to_real__posterior_bank_v2           |                                       30000 |                                      0 |                                      10063 |                                   125 |           30000 |           9938 | 0.00%                | 1.24%               |
| real_plus_synthetic_to_real__posterior_bank_v2 |                                       60762 |                                    661 |                                      10063 |                                   125 |           60101 |           9938 | 1.09%                | 1.24%               |
| real_to_3synthetic__posterior_bank_v2          |                                       30762 |                                    661 |                                       9000 |                                     0 |           30101 |           9000 | 2.15%                | 0.00%               |

The most important point is that the filtering was small for the real data:

```text
Real training input: 661 of 30,762 windows dropped = 2.15%
Real test input:     125 of 10,063 windows dropped = 1.24%
```

No synthetic training windows were dropped for either KoVAE-Rollout or KoVAE-Posterior.

---

## 5. Overall downstream results

| framework   | modality   | experiment                                     | synthetic_method   | synthetic_method_display_name   |   accuracy |   macro_f1 |   weighted_f1 |   balanced_accuracy |
|:------------|:-----------|:-----------------------------------------------|:-------------------|:--------------------------------|-----------:|-----------:|--------------:|--------------------:|
| aeon        | fused      | real_to_real                                   | none               | none                            |     0.6772 |     0.7277 |        0.6747 |              0.7705 |
| aeon        | fused      | synthetic_to_real__rollout_v1                  | rollout_v1         | KoVAE-Rollout                   |     0.3026 |     0.295  |        0.2131 |              0.447  |
| aeon        | fused      | real_plus_synthetic_to_real__rollout_v1        | rollout_v1         | KoVAE-Rollout                   |     0.7004 |     0.7283 |        0.7024 |              0.7678 |
| aeon        | fused      | real_to_3synthetic__rollout_v1                 | rollout_v1         | KoVAE-Rollout                   |     0.2923 |     0.0773 |        0.1504 |              0.1377 |
| aeon        | fused      | synthetic_to_real__posterior_bank_v2           | posterior_bank_v2  | KoVAE-Posterior                 |     0.6506 |     0.6688 |        0.6299 |              0.7266 |
| aeon        | fused      | real_plus_synthetic_to_real__posterior_bank_v2 | posterior_bank_v2  | KoVAE-Posterior                 |     0.7076 |     0.7366 |        0.7123 |              0.7827 |
| aeon        | fused      | real_to_3synthetic__posterior_bank_v2          | posterior_bank_v2  | KoVAE-Posterior                 |     0.5101 |     0.4916 |        0.4484 |              0.4473 |

---

## 6. Result changes compared with real-only baseline

| experiment                                     | method          |   accuracy |   accuracy_delta_vs_real |   macro_f1 |   macro_f1_delta_vs_real |   weighted_f1 |   weighted_f1_delta_vs_real |   balanced_accuracy |   balanced_accuracy_delta_vs_real |
|:-----------------------------------------------|:----------------|-----------:|-------------------------:|-----------:|-------------------------:|--------------:|----------------------------:|--------------------:|----------------------------------:|
| real_to_real                                   | none            |     0.6772 |                   0      |     0.7277 |                   0      |        0.6747 |                      0      |              0.7705 |                            0      |
| synthetic_to_real__rollout_v1                  | KoVAE-Rollout   |     0.3026 |                  -0.3746 |     0.295  |                  -0.4327 |        0.2131 |                     -0.4616 |              0.447  |                           -0.3235 |
| real_plus_synthetic_to_real__rollout_v1        | KoVAE-Rollout   |     0.7004 |                   0.0232 |     0.7283 |                   0.0006 |        0.7024 |                      0.0278 |              0.7678 |                           -0.0027 |
| real_to_3synthetic__rollout_v1                 | KoVAE-Rollout   |     0.2923 |                  -0.3849 |     0.0773 |                  -0.6504 |        0.1504 |                     -0.5243 |              0.1377 |                           -0.6329 |
| synthetic_to_real__posterior_bank_v2           | KoVAE-Posterior |     0.6506 |                  -0.0266 |     0.6688 |                  -0.0589 |        0.6299 |                     -0.0448 |              0.7266 |                           -0.0439 |
| real_plus_synthetic_to_real__posterior_bank_v2 | KoVAE-Posterior |     0.7076 |                   0.0304 |     0.7366 |                   0.0089 |        0.7123 |                      0.0376 |              0.7827 |                            0.0122 |
| real_to_3synthetic__posterior_bank_v2          | KoVAE-Posterior |     0.5101 |                  -0.1671 |     0.4916 |                  -0.2361 |        0.4484 |                     -0.2263 |              0.4473 |                           -0.3232 |

The real-only baseline achieved:

```text
Accuracy:          0.6772
Macro-F1:          0.7277
Weighted-F1:       0.6747
Balanced accuracy: 0.7705
```

The best result was obtained by adding KoVAE-Posterior synthetic data to the real training data:

```text
Experiment:        real_plus_synthetic_to_real__posterior_bank_v2
Accuracy:          0.7076
Macro-F1:          0.7366
Weighted-F1:       0.7123
Balanced accuracy: 0.7827
```

Compared with real-only training, KoVAE-Posterior augmentation improved:

```text
Accuracy:          +0.0304
Macro-F1:          +0.0089
Weighted-F1:       +0.0376
Balanced accuracy: +0.0122
```

This is the strongest downstream result in the notebook.

---

## 7. Interpretation of each experiment

### 7.1 Real-only training: real_to_real

This is the baseline.

```text
Macro-F1 = 0.7277
Accuracy = 0.6772
```

This shows how well the MiniRocket classifier generalizes from the 10 real training subjects to the 3 unseen real test subjects.

### 7.2 Synthetic-only training: synthetic_to_real

This tests whether synthetic data alone can train a classifier that works on real test subjects.

KoVAE-Posterior synthetic-only achieved:

```text
Macro-F1 = 0.6688
Accuracy = 0.6506
```

KoVAE-Rollout synthetic-only achieved:

```text
Macro-F1 = 0.2950
Accuracy = 0.3026
```

This is an important result. KoVAE-Posterior synthetic-only is clearly much better than KoVAE-Rollout synthetic-only. It is still below the real-only baseline, but it is close enough to show that KoVAE-Posterior contains useful activity-discriminative information.

KoVAE-Rollout performs poorly as synthetic-only training data. This agrees with the previous realism/diversity results and the visual comparison, where rollout was too smooth and less realistic.

### 7.3 Real + synthetic training: real_plus_synthetic_to_real

This is the most important downstream utility test.

KoVAE-Posterior augmentation achieved the best overall result:

```text
Macro-F1 = 0.7366
Accuracy = 0.7076
```

KoVAE-Rollout augmentation achieved:

```text
Macro-F1 = 0.7283
Accuracy = 0.7004
```

Both augmentation settings improved accuracy compared with real-only training, but KoVAE-Posterior gave the strongest and most consistent result.

The key conclusion is:

```text
Adding KoVAE-Posterior synthetic data improves the downstream real-test performance.
```

The gain is not huge, but it is positive across accuracy, macro-F1, weighted-F1, and balanced accuracy.

### 7.4 Real-to-synthetic testing

This tests whether a classifier trained on real subjects recognizes synthetic subjects.

KoVAE-Posterior synthetic test performance:

```text
Macro-F1 = 0.4916
Accuracy = 0.5101
```

KoVAE-Rollout synthetic test performance:

```text
Macro-F1 = 0.0773
Accuracy = 0.2923
```

The real-trained classifier recognizes KoVAE-Posterior much better than KoVAE-Rollout. This supports the earlier conclusion that KoVAE-Posterior is more aligned with real activity patterns.

However, the real-to-synthetic score is still lower than real-to-real. This means the synthetic data is not identical to the real distribution, which is expected.

---

## 8. Per-activity analysis

|   activity |   real_only_f1 |   real_plus_posterior_f1 |   posterior_delta |   real_plus_rollout_f1 |   rollout_delta |   posterior_only_f1 |   rollout_only_f1 |
|-----------:|---------------:|-------------------------:|------------------:|-----------------------:|----------------:|--------------------:|------------------:|
|          1 |         0.6681 |                   0.7603 |            0.0922 |                 0.7535 |          0.0853 |              0.5928 |            0.3392 |
|          2 |         0.8765 |                   0.8692 |           -0.0074 |                 0.8542 |         -0.0223 |              0.866  |            0.6203 |
|          3 |         0.6278 |                   0.5187 |           -0.1091 |                 0.5231 |         -0.1048 |              0.5929 |            0.1967 |
|          4 |         0.9847 |                   0.9712 |           -0.0135 |                 0.9636 |         -0.0211 |              0.9477 |            0.4304 |
|          5 |         0.8766 |                   0.8116 |           -0.065  |                 0.8514 |         -0.0252 |              0.7539 |            0.0884 |
|          6 |         0.5426 |                   0.6456 |            0.103  |                 0.6286 |          0.086  |              0.644  |            0.0496 |
|          7 |         0.6727 |                   0.6653 |           -0.0073 |                 0.6143 |         -0.0583 |              0.638  |            0.5169 |
|          8 |         0.5726 |                   0.6512 |            0.0786 |                 0.6378 |          0.0652 |              0.3152 |            0.1187 |

KoVAE-Posterior augmentation improved these activities compared with real-only training:

```text
Activity 1: +0.0922
Activity 6: +0.1030
Activity 8: +0.0786
```

The largest improvement was activity 6:

```text
Activity 6 F1: 0.5426 -> 0.6456
```

This matters because activity 6 had weak real-only F1. Synthetic augmentation helped this difficult class.

KoVAE-Posterior augmentation reduced F1 most strongly for activity 3 and activity 5:

```text
Activity 3 F1: 0.6278 -> 0.5187
Activity 5 F1: 0.8766 -> 0.8116
```

So the benefit is not uniform across all activities. The synthetic data helps some classes but hurts others.

This should be discussed honestly in the report.

---

## 9. Main conclusion

The downstream evaluation supports the usefulness of KoVAE-Posterior synthetic data.

The strongest evidence is:

```text
Real-only macro-F1:               0.7277
Real + KoVAE-Posterior macro-F1:  0.7366

Real-only accuracy:               0.6772
Real + KoVAE-Posterior accuracy:  0.7076
```

The improvement is modest but meaningful because the test set contains held-out real subjects.

The final conclusion is:

```text
KoVAE-Posterior is the best synthetic generation method in this project.
It improves downstream activity detection when added to real training data.
KoVAE-Rollout is useful as a baseline but is much weaker as synthetic-only training data.
```

---

## 10. Report-ready paragraph

A report-ready paragraph is:

> Downstream activity-recognition experiments were performed using a MiniRocket classifier on fused multi-channel windows containing ACC, BVP, EDA, and TEMP signals. The real-only baseline, trained on 10 real subjects and tested on 3 held-out real subjects, achieved 0.678 accuracy and 0.728 macro-F1. Training on KoVAE-Posterior synthetic data alone achieved 0.651 accuracy and 0.669 macro-F1 on the real test subjects, showing that the generated data contains useful activity-discriminative structure. The strongest result was obtained when real training data was augmented with KoVAE-Posterior synthetic data, reaching 0.708 accuracy and 0.737 macro-F1. This improved over the real-only baseline by +0.030 accuracy and +0.009 macro-F1. KoVAE-Rollout was much weaker in the synthetic-only setting, achieving only 0.303 accuracy and 0.295 macro-F1. Overall, the downstream evaluation shows that KoVAE-Posterior provides useful synthetic data for activity detection, although the improvement is moderate and not uniform across all activity classes.

---

## 11. Limitations

The downstream results have some limitations.

First, the improvement from KoVAE-Posterior augmentation is modest. It supports usefulness, but it is not a dramatic improvement.

Second, the per-activity results show mixed effects. Some activities improve, especially activity 1, activity 6, and activity 8, but activity 3 and activity 5 become worse.

Third, the evaluation used one classifier family, MiniRocket. A stronger final study could also test a neural model such as CNN-BiLSTM or InceptionTime.

Fourth, near-flat windows had to be dropped only for aeon compatibility. The percentage was small, but it should be mentioned.

Fifth, the synthetic test subjects are artificial groupings, not true independent human subjects. Therefore, the most important result remains the test performance on real held-out subjects.

---

## 12. Final answer to the project question

The project question was:

```text
What is the impact of using synthetic data for human activity detection?
```

Based on the final downstream evaluation:

```text
Using KoVAE-Posterior synthetic data as augmentation improves real-subject activity detection slightly.
Using KoVAE-Posterior synthetic data alone gives reasonable but lower performance than real-only training.
Using KoVAE-Rollout synthetic data alone performs poorly.
```

Therefore:

```text
Synthetic data is useful when it is realistic enough and used as augmentation, not as a full replacement for real data.
```
