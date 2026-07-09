# Extended Downstream Activity Classification Discussion

## 1. Purpose of the extended evaluation

This extended evaluation was added after the main downstream notebook to make the activity-recognition analysis broader.  
The earlier downstream notebook focused mainly on the fused multi-channel setting. This extended version evaluates the same downstream question across separate sensor modalities as well:

```text
ACC
BVP
EDA
TEMP
Fused ACC + BVP + EDA + TEMP
```

The main question is:

```text
How useful are KoVAE-Rollout, KoVAE-Posterior, and TimeVAE-Prior synthetic data for activity recognition?
```

The evaluated synthetic methods were:

```text
KoVAE-Rollout      = rollout_v1
KoVAE-Posterior    = posterior_bank_v2
TimeVAE-Prior      = prior_v1
```

The classifier framework was:

```text
aeon MiniRocket/Rocket-style classifier
```

This notebook also used pretrained real-trained aeon models where possible. Therefore, the pretrained real model was reused for the real-to-real and real-to-synthetic tests. The synthetic-to-real and real-plus-synthetic-to-real settings still required training new classifiers because their training data changed.

---

## 2. Evaluation design

The extended notebook used four evaluation settings:

```text
1. real_to_real
   Train on 10 real subjects and test on 3 held-out real subjects.

2. real_to_3synthetic
   Use the real-trained classifier and test it on the first 3 synthetic subjects.

3. synthetic_to_real
   Train only on synthetic subjects and test on 3 held-out real subjects.

4. real_plus_synthetic_to_real
   Train on real + synthetic data and test on 3 held-out real subjects.
```

The most important setting for downstream utility is:

```text
real_plus_synthetic_to_real
```

because it directly tests whether synthetic data improves activity recognition on unseen real subjects.

The second most important setting is:

```text
synthetic_to_real
```

because it checks whether synthetic data alone contains activity-discriminative structure that transfers to real subjects.

---

## 3. Coverage and successful execution

All expected extended experiments completed successfully.

```text
KoVAE coverage:   {'ok': 35}
TimeVAE coverage: {'ok': 20}
```

This means the extended comparison is complete for the configured methods and modalities.

---

## 4. Real-only baseline by modality

The real-only baseline shows how much each modality contributes to activity classification before adding synthetic data.

| modality   |   accuracy |   macro_f1 |   weighted_f1 |   balanced_accuracy |
|:-----------|-----------:|-----------:|--------------:|--------------------:|
| acc        |     0.6879 |     0.7268 |        0.6924 |              0.7156 |
| bvp        |     0.4507 |     0.4658 |        0.4461 |              0.4652 |
| eda        |     0.303  |     0.2448 |        0.2848 |              0.2664 |
| temp       |     0.1827 |     0.1457 |        0.1886 |              0.1735 |
| fused      |     0.6859 |     0.7281 |        0.6875 |              0.7691 |

The strongest real-only performance came from ACC and fused signals. ACC alone already achieved high macro-F1, and the fused representation was similar. This suggests that body motion from the accelerometer is the dominant signal for activity recognition in this dataset.

BVP had moderate activity-discrimination ability, while EDA and TEMP were much weaker as individual modalities. This is expected because EDA and TEMP are slow physiological signals and do not directly encode movement patterns as strongly as acceleration.

---

## 5. Fused downstream results

The fused setting is the most important result because it combines all available physiological channels.

| experiment                  | synthetic_method_display_name   |   accuracy |   macro_f1 |   weighted_f1 |   balanced_accuracy |   used_pretrained_real_model |
|:----------------------------|:--------------------------------|-----------:|-----------:|--------------:|--------------------:|-----------------------------:|
| real_to_real                | none                            |     0.6859 |     0.7281 |        0.6875 |              0.7691 |                            1 |
| real_to_real                | none                            |     0.6859 |     0.7281 |        0.6875 |              0.7691 |                            1 |
| real_to_3synthetic          | KoVAE-Posterior                 |     0.5179 |     0.5151 |        0.4622 |              0.469  |                            1 |
| real_to_3synthetic          | KoVAE-Rollout                   |     0.1937 |     0.094  |        0.1512 |              0.131  |                            1 |
| real_to_3synthetic          | TimeVAE-Prior                   |     0.181  |     0.0628 |        0.0824 |              0.1314 |                            1 |
| synthetic_to_real           | KoVAE-Posterior                 |     0.6506 |     0.6688 |        0.6299 |              0.7266 |                            0 |
| synthetic_to_real           | KoVAE-Rollout                   |     0.3026 |     0.295  |        0.2131 |              0.447  |                            0 |
| synthetic_to_real           | TimeVAE-Prior                   |     0.3078 |     0.3148 |        0.2515 |              0.4138 |                            0 |
| real_plus_synthetic_to_real | KoVAE-Posterior                 |     0.7076 |     0.7366 |        0.7123 |              0.7827 |                            0 |
| real_plus_synthetic_to_real | KoVAE-Rollout                   |     0.7004 |     0.7283 |        0.7024 |              0.7678 |                            0 |
| real_plus_synthetic_to_real | TimeVAE-Prior                   |     0.7204 |     0.7455 |        0.7282 |              0.786  |                            0 |

### Interpretation

The fused real-only baseline achieved:

```text
Accuracy:    0.6859
Macro-F1:    0.7281
Weighted-F1: 0.6875
```

For synthetic-only training, KoVAE-Posterior was clearly the strongest method:

```text
KoVAE-Posterior synthetic-to-real macro-F1: 0.6688
KoVAE-Rollout synthetic-to-real macro-F1:   0.2950
TimeVAE-Prior synthetic-to-real macro-F1:   0.3148
```

This means KoVAE-Posterior produces synthetic data that transfers much better to real held-out subjects. KoVAE-Rollout and TimeVAE-Prior are much weaker when used as a full replacement for real training data.

For augmentation, all three methods were compared against the fused real-only baseline. The strongest fused augmentation result was TimeVAE-Prior:

```text
Real-only fused macro-F1:                0.7281
Real + KoVAE-Rollout fused macro-F1:     0.7283
Real + KoVAE-Posterior fused macro-F1:   0.7366
Real + TimeVAE-Prior fused macro-F1:     0.7455
```

TimeVAE-Prior gave the highest fused augmentation macro-F1 and accuracy. However, this does not mean TimeVAE is the most realistic generator overall. Its synthetic-only and real-to-synthetic scores are weak. Therefore, TimeVAE seems more useful as an augmentation/regularization source than as a realistic standalone replacement for real data.

---

## 6. Augmentation effect compared with real-only baseline

The following table shows how much each real-plus-synthetic setting changed performance compared with the corresponding real-only baseline for the same modality.

| modality   | synthetic_method_display_name   |   accuracy_delta_vs_real |   macro_f1_delta_vs_real |   weighted_f1_delta_vs_real |   balanced_accuracy_delta_vs_real |
|:-----------|:--------------------------------|-------------------------:|-------------------------:|----------------------------:|----------------------------------:|
| acc        | KoVAE-Posterior                 |                   0.016  |                   0.012  |                      0.0151 |                            0.0104 |
| acc        | KoVAE-Rollout                   |                   0.0128 |                   0.0017 |                      0.0115 |                           -0.002  |
| acc        | TimeVAE-Prior                   |                   0.0157 |                   0.0079 |                      0.0147 |                            0.0052 |
| bvp        | KoVAE-Posterior                 |                   0.0236 |                   0.0178 |                      0.0231 |                            0.0174 |
| bvp        | KoVAE-Rollout                   |                  -0.0154 |                  -0.0389 |                     -0.0175 |                           -0.0417 |
| bvp        | TimeVAE-Prior                   |                   0.0217 |                   0.0066 |                      0.0186 |                            0.0023 |
| eda        | KoVAE-Posterior                 |                  -0.003  |                  -0.0061 |                     -0.002  |                           -0.0013 |
| eda        | KoVAE-Rollout                   |                  -0.0116 |                  -0.0103 |                     -0.0144 |                           -0.0047 |
| eda        | TimeVAE-Prior                   |                  -0.0042 |                  -0.0126 |                     -0.0048 |                           -0.0071 |
| fused      | KoVAE-Posterior                 |                   0.0217 |                   0.0085 |                      0.0249 |                            0.0136 |
| fused      | KoVAE-Rollout                   |                   0.0146 |                   0.0002 |                      0.015  |                           -0.0013 |
| fused      | TimeVAE-Prior                   |                   0.0345 |                   0.0174 |                      0.0408 |                            0.0169 |
| temp       | KoVAE-Posterior                 |                   0.033  |                   0.0319 |                      0.0231 |                            0.0504 |
| temp       | KoVAE-Rollout                   |                   0.0269 |                   0.0319 |                      0.0204 |                            0.0428 |
| temp       | TimeVAE-Prior                   |                   0.0288 |                   0.0351 |                      0.0233 |                            0.0454 |

### Interpretation

The best augmentation behavior depends on the modality.

For ACC, KoVAE-Posterior was strongest:

```text
ACC real + KoVAE-Posterior macro-F1 gain: +0.0120
```

For BVP, KoVAE-Posterior also improved over the BVP real-only baseline:

```text
BVP real + KoVAE-Posterior macro-F1 gain: +0.0178
```

For fused data, TimeVAE-Prior gave the largest augmentation gain:

```text
Fused real + TimeVAE-Prior macro-F1 gain: +0.0174
```

KoVAE-Posterior also improved the fused setting:

```text
Fused real + KoVAE-Posterior macro-F1 gain: +0.0085
```

EDA was the weakest augmentation modality. Most EDA gains were negative or very small. This confirms that EDA alone is not a strong activity-recognition signal in the current window-level setup.

TEMP showed small positive augmentation gains, but its absolute real-only performance was low. Therefore, TEMP improvements should be interpreted carefully.

---

## 7. Synthetic-only transfer to real subjects

Synthetic-to-real testing checks whether synthetic data alone can train a classifier that works on unseen real subjects.

| modality   | synthetic_method_display_name   |   accuracy |   macro_f1 |   weighted_f1 |   balanced_accuracy |
|:-----------|:--------------------------------|-----------:|-----------:|--------------:|--------------------:|
| acc        | KoVAE-Posterior                 |     0.6546 |     0.6875 |        0.6509 |              0.698  |
| acc        | KoVAE-Rollout                   |     0.2277 |     0.2555 |        0.1756 |              0.3841 |
| acc        | TimeVAE-Prior                   |     0.3148 |     0.2456 |        0.227  |              0.3348 |
| bvp        | KoVAE-Posterior                 |     0.4618 |     0.4671 |        0.4545 |              0.4761 |
| bvp        | KoVAE-Rollout                   |     0.1592 |     0.1022 |        0.0803 |              0.1704 |
| bvp        | TimeVAE-Prior                   |     0.1872 |     0.1295 |        0.1681 |              0.1634 |
| eda        | KoVAE-Posterior                 |     0.2697 |     0.1854 |        0.2178 |              0.2538 |
| eda        | KoVAE-Rollout                   |     0.2513 |     0.1225 |        0.173  |              0.1797 |
| eda        | TimeVAE-Prior                   |     0.1837 |     0.1107 |        0.1311 |              0.1959 |
| fused      | KoVAE-Posterior                 |     0.6506 |     0.6688 |        0.6299 |              0.7266 |
| fused      | KoVAE-Rollout                   |     0.3026 |     0.295  |        0.2131 |              0.447  |
| fused      | TimeVAE-Prior                   |     0.3078 |     0.3148 |        0.2515 |              0.4138 |
| temp       | KoVAE-Posterior                 |     0.2152 |     0.1619 |        0.2019 |              0.2277 |
| temp       | KoVAE-Rollout                   |     0.2178 |     0.0975 |        0.174  |              0.1923 |
| temp       | TimeVAE-Prior                   |     0.2849 |     0.0993 |        0.1689 |              0.1958 |

### Interpretation

KoVAE-Posterior was the strongest synthetic-only method in almost every important modality. In the fused setting, it achieved:

```text
KoVAE-Posterior synthetic-to-real macro-F1: 0.6688
```

This is much closer to the real-only fused baseline than KoVAE-Rollout or TimeVAE-Prior.

KoVAE-Rollout performed poorly as synthetic-only training data, especially in fused and BVP settings. This agrees with the previous visual and realism/diversity analysis, where rollout produced smoother and less realistic signals.

TimeVAE-Prior also performed poorly as synthetic-only training data. Its fused synthetic-to-real macro-F1 was only:

```text
TimeVAE-Prior synthetic-to-real macro-F1: 0.3148
```

This suggests that although TimeVAE-Prior can help when added to real data, it does not produce a synthetic distribution that can fully replace real data for activity recognition.

---

## 8. Real-trained classifier tested on synthetic subjects

Real-to-synthetic testing checks whether a classifier trained on real subjects recognizes the generated synthetic subjects.

| modality   | synthetic_method_display_name   |   accuracy |   macro_f1 |   weighted_f1 |   balanced_accuracy |
|:-----------|:--------------------------------|-----------:|-----------:|--------------:|--------------------:|
| acc        | KoVAE-Posterior                 |     0.6723 |     0.7055 |        0.6427 |              0.6741 |
| acc        | KoVAE-Rollout                   |     0.2772 |     0.0603 |        0.1254 |              0.1272 |
| acc        | TimeVAE-Prior                   |     0.3    |     0.1152 |        0.1769 |              0.154  |
| bvp        | KoVAE-Posterior                 |     0.5933 |     0.5752 |        0.5878 |              0.5542 |
| bvp        | KoVAE-Rollout                   |     0.1786 |     0.0999 |        0.1491 |              0.1402 |
| bvp        | TimeVAE-Prior                   |     0.2266 |     0.1297 |        0.1882 |              0.148  |
| eda        | KoVAE-Posterior                 |     0.2466 |     0.199  |        0.2236 |              0.2307 |
| eda        | KoVAE-Rollout                   |     0.1527 |     0.1164 |        0.1409 |              0.1449 |
| eda        | TimeVAE-Prior                   |     0.096  |     0.0762 |        0.0883 |              0.1136 |
| fused      | KoVAE-Posterior                 |     0.5179 |     0.5151 |        0.4622 |              0.469  |
| fused      | KoVAE-Rollout                   |     0.1937 |     0.094  |        0.1512 |              0.131  |
| fused      | TimeVAE-Prior                   |     0.181  |     0.0628 |        0.0824 |              0.1314 |
| temp       | KoVAE-Posterior                 |     0.2927 |     0.2015 |        0.2506 |              0.1993 |
| temp       | KoVAE-Rollout                   |     0.2113 |     0.1362 |        0.1906 |              0.1592 |
| temp       | TimeVAE-Prior                   |     0.0797 |     0.038  |        0.031  |              0.126  |

### Interpretation

KoVAE-Posterior again showed the best alignment with the real-trained classifier. This was especially clear in ACC and BVP:

```text
ACC real-to-synthetic macro-F1 with KoVAE-Posterior: 0.7055
BVP real-to-synthetic macro-F1 with KoVAE-Posterior: 0.5752
```

KoVAE-Rollout and TimeVAE-Prior were much weaker under real-to-synthetic testing. This means the real-trained classifier did not recognize their generated windows as consistently activity-discriminative.

The fused real-to-synthetic score for KoVAE-Posterior was lower than its ACC-only and BVP-only score. This suggests that adding all channels together does not automatically make synthetic subjects easier to classify. The weaker EDA/TEMP channels may introduce noise or distribution mismatch in the fused representation.

---

## 9. Best augmentation result per modality

| modality   | synthetic_method_display_name   |   accuracy |   macro_f1 |   weighted_f1 |   balanced_accuracy |   macro_f1_delta_vs_real |
|:-----------|:--------------------------------|-----------:|-----------:|--------------:|--------------------:|-------------------------:|
| fused      | TimeVAE-Prior                   |     0.7204 |     0.7455 |        0.7282 |              0.786  |                   0.0174 |
| acc        | KoVAE-Posterior                 |     0.7039 |     0.7389 |        0.7074 |              0.726  |                   0.012  |
| bvp        | KoVAE-Posterior                 |     0.4742 |     0.4836 |        0.4692 |              0.4825 |                   0.0178 |
| eda        | KoVAE-Posterior                 |     0.3    |     0.2387 |        0.2828 |              0.265  |                  -0.0061 |
| temp       | TimeVAE-Prior                   |     0.2115 |     0.1808 |        0.2119 |              0.2189 |                   0.0351 |

The best method depends on the modality:

```text
ACC:   KoVAE-Posterior
BVP:   KoVAE-Posterior
EDA:   KoVAE-Posterior, but the gain is small and still below the real-only baseline
TEMP:  TimeVAE-Prior
Fused: TimeVAE-Prior
```

This is an important finding. KoVAE-Posterior is the most reliable synthetic generator across modalities, especially when synthetic data is used alone or tested against real-trained classifiers. TimeVAE-Prior gives the best fused augmentation score, but it is much weaker as synthetic-only data.

---

## 10. Top extended results by macro-F1

| modality   | experiment                  | synthetic_method_display_name   |   accuracy |   macro_f1 |   weighted_f1 |   balanced_accuracy |
|:-----------|:----------------------------|:--------------------------------|-----------:|-----------:|--------------:|--------------------:|
| fused      | real_plus_synthetic_to_real | TimeVAE-Prior                   |     0.7204 |     0.7455 |        0.7282 |              0.786  |
| acc        | real_plus_synthetic_to_real | KoVAE-Posterior                 |     0.7039 |     0.7389 |        0.7074 |              0.726  |
| fused      | real_plus_synthetic_to_real | KoVAE-Posterior                 |     0.7076 |     0.7366 |        0.7123 |              0.7827 |
| acc        | real_plus_synthetic_to_real | TimeVAE-Prior                   |     0.7036 |     0.7347 |        0.7071 |              0.7208 |
| acc        | real_plus_synthetic_to_real | KoVAE-Rollout                   |     0.7006 |     0.7285 |        0.7038 |              0.7136 |
| fused      | real_plus_synthetic_to_real | KoVAE-Rollout                   |     0.7004 |     0.7283 |        0.7024 |              0.7678 |
| acc        | real_to_3synthetic          | KoVAE-Posterior                 |     0.6723 |     0.7055 |        0.6427 |              0.6741 |
| acc        | synthetic_to_real           | KoVAE-Posterior                 |     0.6546 |     0.6875 |        0.6509 |              0.698  |
| fused      | synthetic_to_real           | KoVAE-Posterior                 |     0.6506 |     0.6688 |        0.6299 |              0.7266 |
| bvp        | real_to_3synthetic          | KoVAE-Posterior                 |     0.5933 |     0.5752 |        0.5878 |              0.5542 |
| fused      | real_to_3synthetic          | KoVAE-Posterior                 |     0.5179 |     0.5151 |        0.4622 |              0.469  |
| bvp        | real_plus_synthetic_to_real | KoVAE-Posterior                 |     0.4742 |     0.4836 |        0.4692 |              0.4825 |

This ranking shows that the strongest extended results are mostly augmentation experiments. This supports the main project conclusion:

```text
Synthetic data is most useful as augmentation, not as a full replacement for real data.
```

---

## 11. Per-activity fused augmentation effects

The table below shows per-activity F1 change compared with the fused real-only baseline.

|   activity_label |   KoVAE-Rollout |   KoVAE-Posterior |   TimeVAE-Prior |
|-----------------:|----------------:|------------------:|----------------:|
|                1 |          0.0106 |            0.0174 |          0.0869 |
|                2 |         -0.0097 |            0.0053 |         -0.0019 |
|                3 |         -0.0216 |           -0.026  |         -0.0464 |
|                4 |         -0.0204 |           -0.0127 |         -0.0351 |
|                5 |          0.0242 |           -0.0156 |          0.0039 |
|                6 |          0.0485 |            0.0655 |          0.1017 |
|                7 |         -0.0617 |           -0.0107 |         -0.0094 |
|                8 |          0.0315 |            0.045  |          0.0395 |

### Interpretation

For fused augmentation:

- TimeVAE-Prior improved activity 1 and activity 6 the most.
- KoVAE-Posterior improved activity 1, activity 2, activity 6, and activity 8.
- KoVAE-Rollout improved activity 5, activity 6, and activity 8, but reduced activity 7 more strongly.

The strongest per-class improvement was for activity 6 using TimeVAE-Prior:

```text
Activity 6 F1 gain with TimeVAE-Prior: +0.1017
```

KoVAE-Posterior also improved activity 6:

```text
Activity 6 F1 gain with KoVAE-Posterior: +0.0655
```

However, activity 3 decreased for all augmentation methods. This means synthetic augmentation does not help all activity classes equally.

---

## 12. Relationship to the previous downstream evaluation

The earlier downstream notebook already showed that KoVAE-Posterior was the best KoVAE method in the fused setting. The extended evaluation confirms that conclusion more broadly:

```text
KoVAE-Posterior is much better than KoVAE-Rollout for synthetic-only training.
KoVAE-Posterior is much better than KoVAE-Rollout for real-to-synthetic recognition.
KoVAE-Posterior improves fused real-test performance when used as augmentation.
```

The new result from this extended evaluation is that TimeVAE-Prior achieved the highest fused augmentation score:

```text
Real-only fused macro-F1:            0.7281
Real + TimeVAE-Prior fused macro-F1: 0.7455
```

This should be reported carefully. TimeVAE-Prior is not the best standalone synthetic generator because its synthetic-only and real-to-synthetic results are weak. Its benefit is mainly in the real-plus-synthetic setting.

---

## 13. Main conclusion

The extended downstream evaluation supports three main conclusions.

First, ACC is the strongest individual modality for activity recognition. Fused data gives strong performance, but ACC alone is already highly informative.

Second, KoVAE-Posterior is the most realistic and transferable synthetic generator. It performs best in synthetic-only training and real-to-synthetic testing, especially for ACC, BVP, and fused data.

Third, TimeVAE-Prior gives the best fused augmentation result, but its weak synthetic-only transfer suggests that it acts more like an augmentation regularizer than a faithful standalone synthetic replacement.

The final interpretation is:

```text
KoVAE-Posterior is the strongest overall synthetic generator.
TimeVAE-Prior gives the highest fused augmentation score.
Synthetic data is useful mainly as augmentation, not as a full replacement for real data.
```

---

## 14. Report-ready paragraph

The extended downstream evaluation was performed using aeon time-series classifiers across ACC, BVP, EDA, TEMP, and fused multi-channel inputs. The real-only baseline showed that ACC and fused signals were the strongest activity-recognition inputs, achieving macro-F1 scores of 0.727 and 0.728, respectively. Among the synthetic methods, KoVAE-Posterior was the strongest standalone generator: in the fused synthetic-to-real setting, it achieved 0.669 macro-F1, clearly outperforming KoVAE-Rollout (0.295) and TimeVAE-Prior (0.315). In the augmentation setting, the highest fused result was obtained by adding TimeVAE-Prior to real data, increasing macro-F1 from 0.728 to 0.746. KoVAE-Posterior also improved fused macro-F1 to 0.737. These results show that synthetic physiological data is most useful as augmentation rather than as a full replacement for real data. KoVAE-Posterior provides the most transferable synthetic distribution, while TimeVAE-Prior can still provide useful regularization when combined with real training data.

---

## 15. Limitations of the extended evaluation

The extended evaluation has several limitations.

First, the real-to-real and real-to-synthetic settings used pretrained real classifiers. This saves time and keeps the real-to-synthetic comparison consistent, but it means those rows depend on the pretrained model state.

Second, synthetic-to-real and real-plus-synthetic-to-real classifiers were trained freshly because pretrained real models cannot be reused when the training data changes.

Third, the extended results use one classifier family, aeon Rocket-style classification. A stronger final study could also include tsai/InceptionTime results if there is enough time.

Fourth, EDA and TEMP remained weak individual modalities. Their low performance does not necessarily mean they are useless physiologically; it means they are less useful for this window-level activity classification setup.

Fifth, TimeVAE-Prior produced the best fused augmentation score but weak synthetic-only transfer. Therefore, its result should not be interpreted as evidence that TimeVAE generated the most realistic synthetic data.

---

## 16. Final paper message

For the final paper, the strongest message is:

```text
Realism and downstream utility are not the same.
KoVAE-Posterior gives the best transferable synthetic data.
TimeVAE-Prior gives the best fused augmentation score.
Synthetic data improves activity recognition when added to real training data, but it does not fully replace real data.
```
