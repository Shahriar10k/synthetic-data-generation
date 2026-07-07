# 05 Realism, No-Copying, and Diversity Discussion

## 1. Purpose of this notebook

Notebook 05 evaluates the statistical quality of the generated synthetic wearable windows.

The previous notebook used visual comparison. That helped us inspect the shape of the generated signals, but visual inspection alone is not enough. Notebook 05 adds quantitative metrics for three questions:

```text
1. Realism:
   Do synthetic windows have similar statistical and frequency properties to real windows?

2. No-copying:
   Are synthetic windows too close to real training windows?

3. Diversity:
   Are synthetic windows diverse, or did the generator collapse to very similar samples?
```

The evaluation was run for both final KoVAE generation methods:

```text
rollout_v1        -> KoVAE-Rollout
posterior_bank_v2 -> KoVAE-Posterior
```

The reference real data split was:

```text
realism_reference_split = train
```

This is appropriate because the synthetic data was generated from a model trained on the training subjects.

The evaluated signals were:

```text
ACC_x, ACC_y, ACC_z, BVP, EDA, TEMP
```

and the activities were:

```text
1, 2, 3, 4, 5, 6, 7, 8
```

---

## 2. Metrics used

### 2.1 Realism metrics

The realism metrics were:

```text
absolute mean difference       lower is better
absolute standard deviation difference lower is better
histogram overlap              higher is better, range 0 to 1
FFT log-magnitude MAE          lower is better
```

Histogram overlap measures how similar the real and synthetic value distributions are. A value close to 1 means the distributions overlap strongly.

FFT log-magnitude MAE compares the average frequency-domain structure. Lower values mean the synthetic signal has more similar spectral behavior to the real signal.

### 2.2 No-copying metrics

The no-copying metrics were based on nearest-neighbor mean squared distance.

The main metrics were:

```text
copy_ratio_mean
near_duplicate_rate_p01_lower_is_better
```

A very low copy ratio can indicate that synthetic samples are too close to real training samples. A very high copy ratio means synthetic samples are far from real training data. A near-duplicate rate close to 0 is best.

### 2.3 Diversity metric

The diversity metric was:

```text
synthetic_diversity_ratio_mean
```

A value near 1 is ideal. A value much lower than 1 suggests low diversity or collapse. A value much higher than 1 suggests the synthetic data may be too variable or noisy compared with real data.

---

## 3. Overall results

The overall results strongly favor KoVAE-Posterior over KoVAE-Rollout.

| method_display_name   |   abs_mean_diff_lower_is_better |   abs_std_diff_lower_is_better |   histogram_overlap_0_to_1_higher_is_better |   fft_logmag_mae_lower_is_better |   copy_ratio_mean |   near_duplicate_rate_p01_lower_is_better |   synthetic_diversity_ratio_mean |
|:----------------------|--------------------------------:|-------------------------------:|--------------------------------------------:|---------------------------------:|------------------:|------------------------------------------:|---------------------------------:|
| KoVAE-Posterior       |                           0.03  |                          0.143 |                                       0.831 |                            0.169 |             2.027 |                                     0.076 |                            0.944 |
| KoVAE-Rollout         |                           0.352 |                          0.655 |                                       0.275 |                            0.475 |            44.552 |                                     0.03  |                            2.1   |

### Interpretation

KoVAE-Posterior achieved:

```text
histogram overlap: 0.831
FFT log-magnitude MAE: 0.169
synthetic diversity ratio: 0.944
near-duplicate rate: 0.076
```

KoVAE-Rollout achieved:

```text
histogram overlap: 0.275
FFT log-magnitude MAE: 0.475
synthetic diversity ratio: 2.100
near-duplicate rate: 0.030
```

The realism gap is large. KoVAE-Posterior has much better distributional similarity and much better frequency-domain similarity. This confirms the visual comparison from Notebook 04, where the posterior method looked more expressive and realistic than direct rollout.

---

## 4. Realism by signal

| method_display_name   | signal   |   histogram_overlap_0_to_1_higher_is_better |   fft_logmag_mae_lower_is_better |   abs_mean_diff_lower_is_better |   abs_std_diff_lower_is_better |
|:----------------------|:---------|--------------------------------------------:|---------------------------------:|--------------------------------:|-------------------------------:|
| KoVAE-Rollout         | ACC_x    |                                       0.306 |                            0.734 |                           0.311 |                          0.703 |
| KoVAE-Rollout         | ACC_y    |                                       0.161 |                            0.497 |                           0.582 |                          0.613 |
| KoVAE-Rollout         | ACC_z    |                                       0.254 |                            0.643 |                           0.433 |                          0.663 |
| KoVAE-Rollout         | BVP      |                                       0.519 |                            0.251 |                           0.078 |                          0.832 |
| KoVAE-Rollout         | EDA      |                                       0.225 |                            0.326 |                           0.285 |                          0.597 |
| KoVAE-Rollout         | TEMP     |                                       0.186 |                            0.4   |                           0.422 |                          0.521 |
| KoVAE-Posterior       | ACC_x    |                                       0.874 |                            0.268 |                           0.031 |                          0.162 |
| KoVAE-Posterior       | ACC_y    |                                       0.847 |                            0.297 |                           0.025 |                          0.125 |
| KoVAE-Posterior       | ACC_z    |                                       0.872 |                            0.311 |                           0.032 |                          0.139 |
| KoVAE-Posterior       | BVP      |                                       0.941 |                            0.09  |                           0.016 |                          0.184 |
| KoVAE-Posterior       | EDA      |                                       0.74  |                            0.022 |                           0.041 |                          0.137 |
| KoVAE-Posterior       | TEMP     |                                       0.71  |                            0.029 |                           0.034 |                          0.108 |

### 4.1 ACC realism

For the three accelerometer channels, KoVAE-Posterior achieved high histogram overlap:

```text
ACC_x: 0.874
ACC_y: 0.847
ACC_z: 0.872
```

These values are good. They suggest that the posterior method produces accelerometer values with distributions close to real data.

The FFT errors for ACC are still not zero, but they are clearly lower than KoVAE-Rollout. This means KoVAE-Posterior captures ACC signal structure much better than direct rollout.

KoVAE-Rollout performs poorly on ACC. Its histogram overlaps are much lower, and its FFT errors are much higher. This matches the comparison plots, where rollout ACC looked overly smooth and low-amplitude.

### 4.2 BVP realism

BVP is one of the strongest results for KoVAE-Posterior.

KoVAE-Posterior BVP histogram overlap:

```text
0.941
```

KoVAE-Posterior BVP FFT MAE:

```text
0.090
```

This indicates that the posterior-bank method preserves both the value distribution and the frequency-domain behavior of BVP better than rollout.

KoVAE-Rollout BVP has a much lower histogram overlap and higher FFT error. The rollout plot also showed that BVP became too flat after direct Koopman rollout.

### 4.3 EDA and TEMP realism

EDA and TEMP are interesting because the visual timeline looked weaker, but the window-level realism metrics are relatively good for KoVAE-Posterior.

KoVAE-Posterior achieved:

```text
EDA histogram overlap:  0.740
TEMP histogram overlap: 0.710

EDA FFT MAE:  0.022
TEMP FFT MAE: 0.029
```

This means that individual EDA/TEMP windows are statistically close to real EDA/TEMP windows. However, Notebook 04 showed that when many synthetic windows are stitched into a long timeline, EDA/TEMP does not preserve smooth long-term physiological continuity.

So the correct interpretation is:

```text
EDA/TEMP are acceptable at window level,
but weak at long continuous subject-level visualization.
```

This distinction is important for the report.

---

## 5. No-copying and diversity by signal

| method_display_name   | signal   |   copy_ratio_mean |   near_duplicate_rate_p01_lower_is_better |   synthetic_diversity_ratio_mean |
|:----------------------|:---------|------------------:|------------------------------------------:|---------------------------------:|
| KoVAE-Rollout         | ACC_x    |             0.348 |                                     0.1   |                            0.125 |
| KoVAE-Rollout         | ACC_y    |             0.981 |                                     0     |                            0.177 |
| KoVAE-Rollout         | ACC_z    |             0.372 |                                     0.01  |                            0.096 |
| KoVAE-Rollout         | BVP      |             0.087 |                                     0.071 |                            0.063 |
| KoVAE-Rollout         | EDA      |            62.613 |                                     0     |                            4.041 |
| KoVAE-Rollout         | TEMP     |           202.908 |                                     0     |                            8.098 |
| KoVAE-Posterior       | ACC_x    |             0.544 |                                     0.111 |                            0.56  |
| KoVAE-Posterior       | ACC_y    |             0.647 |                                     0.092 |                            0.559 |
| KoVAE-Posterior       | ACC_z    |             0.535 |                                     0.094 |                            0.528 |
| KoVAE-Posterior       | BVP      |             0.514 |                                     0.162 |                            0.532 |
| KoVAE-Posterior       | EDA      |             5.71  |                                     0     |                            1.191 |
| KoVAE-Posterior       | TEMP     |             4.213 |                                     0     |                            2.295 |

## 6. Copying analysis

The no-copying results need careful interpretation.

KoVAE-Posterior has an overall near-duplicate rate of:

```text
0.076
```

KoVAE-Rollout has an overall near-duplicate rate of:

```text
0.030
```

At first, rollout looks safer because its near-duplicate rate is lower. But this does not mean rollout is better. Rollout is often far from real data or collapsed, which can reduce near-duplicate rate while still producing poor synthetic data.

For KoVAE-Posterior, the near-duplicate rate is higher, especially for BVP:

```text
BVP near-duplicate rate: 0.162
```

This is a warning sign. It means some posterior-bank synthetic BVP windows are close to real training windows according to the nearest-neighbor threshold.

However, the rate is not extreme enough to conclude direct memorization from this metric alone. It means we should mention a mild copying-risk limitation and keep no-copying evaluation in the report.

A strong statement would be wrong. A careful statement is:

```text
KoVAE-Posterior improves realism but shows a higher near-duplicate rate than rollout,
so realism comes with a stronger need to monitor copying risk.
```

---

## 7. Diversity analysis

KoVAE-Posterior has an overall synthetic diversity ratio of:

```text
0.944
```

This is close to 1, which is good. It means the posterior method has a reasonable overall diversity level.

However, diversity differs by signal.

For KoVAE-Posterior:

```text
ACC_x diversity ratio: 0.560
ACC_y diversity ratio: 0.559
ACC_z diversity ratio: 0.528
BVP diversity ratio:   0.532
EDA diversity ratio:   1.191
TEMP diversity ratio:  2.295
```

ACC and BVP diversity are around 0.5. This suggests they are somewhat less diverse than real training windows. This is not catastrophic, but it means the posterior generator still has some diversity limitation.

EDA is close to 1, which is good. TEMP is above 2, meaning TEMP may be more variable than real data.

For KoVAE-Rollout, the diversity pattern is worse. ACC and BVP have very low diversity, while EDA and TEMP have very high ratios. This indicates unstable and inconsistent diversity behavior.

---

## 8. Why KoVAE-Posterior is better than KoVAE-Rollout

The results show that KoVAE-Posterior is better for the main goal of synthetic data generation.

The reasons are:

```text
1. Much higher histogram overlap
2. Lower FFT error
3. Overall diversity close to 1
4. More realistic ACC and BVP behavior
5. Better balance between realism and diversity
```

KoVAE-Rollout is stable but too conservative. Its ACC and BVP diversity ratios are very low, and its realism metrics are poor. This suggests that the direct stabilized Koopman rollout compresses the latent dynamics too much.

Therefore, KoVAE-Rollout should be kept as a baseline/ablation, while KoVAE-Posterior should be treated as the main KoVAE synthetic dataset for downstream experiments.

---

## 9. Connection with Notebook 04 visual comparison

Notebook 04 showed:

```text
KoVAE-Rollout:
- smooth
- low-amplitude
- weak ACC/BVP dynamics

KoVAE-Posterior:
- more dynamic
- stronger ACC/BVP variation
- weaker long-term EDA/TEMP continuity
```

Notebook 05 confirms this quantitatively.

The posterior method has clearly better window-level realism. The high histogram overlap and lower FFT MAE show that posterior synthetic windows are much closer to real windows than rollout synthetic windows.

The EDA/TEMP result is nuanced. The window-level metrics are good, but continuous visualization still looks weak. This means EDA/TEMP quality depends on the evaluation perspective:

```text
window-level distribution: acceptable
long continuous physiological trajectory: limited
```

---

## 10. Report-ready results paragraph

A report-ready paragraph is:

> The realism and diversity evaluation showed that KoVAE-Posterior substantially outperformed direct KoVAE-Rollout. KoVAE-Posterior achieved a much higher average histogram overlap with real training data (0.831 vs. 0.275) and a lower FFT log-magnitude MAE (0.169 vs. 0.475), indicating better distributional and frequency-domain realism. Signal-level results showed strong realism for ACC and BVP, with BVP reaching a histogram overlap of 0.941. The overall synthetic diversity ratio of KoVAE-Posterior was close to 1 (0.944), suggesting reasonable diversity, whereas KoVAE-Rollout showed inconsistent diversity with collapsed ACC/BVP behavior and over-dispersed slow signals. However, KoVAE-Posterior had a higher near-duplicate rate (0.076 vs. 0.030), especially for BVP, indicating that improved realism should be interpreted together with copying-risk analysis. Overall, KoVAE-Posterior is selected as the stronger synthetic generation method for downstream activity-recognition experiments.

---

## 11. Limitations

The evaluation has several limitations.

First, the metrics are computed at the window level. They do not prove that the synthetic data forms a smooth continuous physiological recording.

Second, nearest-neighbor copying metrics depend on the selected distance metric and threshold. A higher near-duplicate rate does not automatically prove memorization, but it does indicate that copying risk should be discussed.

Third, EDA and TEMP are slow physiological signals. They can look statistically reasonable at window level while still failing to show smooth long-term continuity.

Fourth, realism does not automatically imply downstream usefulness. The final question is whether synthetic data improves or harms human activity recognition on unseen real subjects.

---

## 12. Next step

The next notebook should evaluate downstream utility using a human activity classifier.

The recommended downstream experiments are:

```text
1. Train on real training data only -> test on real test subjects
2. Train on KoVAE-Rollout only -> test on real test subjects
3. Train on KoVAE-Posterior only -> test on real test subjects
4. Train on real + KoVAE-Rollout -> test on real test subjects
5. Train on real + KoVAE-Posterior -> test on real test subjects
```

The most important final comparison will be:

```text
real only vs real + KoVAE-Posterior
```

If adding KoVAE-Posterior improves macro-F1 or per-class F1 on real test subjects, then the synthetic data is useful for the downstream task.

---

## 13. Final conclusion

Notebook 05 shows that KoVAE-Posterior is the stronger synthetic generation method.

It is much more realistic than KoVAE-Rollout in both distributional and frequency-domain metrics. It also has a much better overall diversity profile. The main remaining concerns are moderate near-duplicate risk, especially for BVP, and imperfect long-term continuity for EDA/TEMP.

Therefore, the project should continue with KoVAE-Posterior as the main synthetic dataset for downstream activity detection, while keeping KoVAE-Rollout as a baseline method.
