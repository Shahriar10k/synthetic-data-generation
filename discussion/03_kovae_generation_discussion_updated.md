# 03 KoVAE Synthetic Generation Discussion and Analysis

## 1. Purpose of this notebook

The purpose of Notebook 03 was to generate synthetic multi-channel wearable time-series windows using the trained KoVAE model from Notebook 02.

The updated KoVAE model is activity-conditioned and subject-conditioned. Therefore, the generation notebook uses the trained checkpoint:

```text
models/checkpoints/kovae_best.pt
```

and generates synthetic data using two KoVAE generation strategies:

```text
rollout_v1
posterior_bank_v2
```

The generated synthetic data keeps the same native-rate format as the real preprocessed data:

```text
ACC:      [N, 256, 3]   at 32 Hz
BVP:      [N, 512, 1]   at 64 Hz
EDA/TEMP: [N, 32, 2]    at 4 Hz
```

The purpose of this notebook is not to prove downstream usefulness yet. It creates synthetic datasets that will be evaluated later using comparison plots, realism/diversity metrics, and CNN-BiLSTM downstream activity detection.

---

## 2. Why the generation was updated

The previous generation notebook was designed for the earlier KoVAE. After updating Notebook 02 to use activity embeddings, subject embeddings, activity classifier loss, and activity-conditioned Koopman matrices, Notebook 03 also had to be updated.

The new generation code uses:

```text
activity-conditioned decoder
subject-conditioned decoder
activity-specific Koopman matrices
activity-specific posterior latent banks
```

This is important because wearable signal patterns depend strongly on activity. For example, sitting should have different accelerometer behavior from walking, stairs, or running.

Because the new KoVAE is already activity-conditioned, no extra quality filtering was used in this generation stage.

```text
uses_quality_filter = false
```

The correct workflow is:

```text
train activity-conditioned KoVAE
generate without filtering
evaluate realism/diversity and downstream utility
only add filtering later if still needed
```

---

## 3. Generation methods

### 3.1 rollout_v1

The first method is the Koopman rollout baseline.

It works as follows:

```text
1. choose an activity label
2. sample an initial latent state z0 from the activity-specific latent bank
3. roll forward using the stabilized activity-specific Koopman matrix
4. decode the latent sequence using activity and subject conditions
```

This method tests whether the learned Koopman dynamics can directly generate plausible synthetic windows.

The method uses:

```text
rollout_z0_noise_scale = 0.05
rollout_latent_noise_scale = 0.03
```

### 3.2 posterior_bank_v2

The second method is the posterior-bank generation method.

It works as follows:

```text
1. choose an activity label
2. sample a full posterior latent trajectory from that activity's real train latent bank
3. optionally interpolate with another same-activity trajectory
4. add controlled latent noise
5. apply light Koopman guidance
6. decode using activity and subject conditions
```

This method relies less on long pure rollout and more on realistic encoded latent trajectories. It should generally be more stable than direct rollout.

The method uses:

```text
posterior_noise_scale = 0.04
posterior_interpolation_prob = 0.5
posterior_koopman_blend_weight = 0.1
```

---

## 4. Generated dataset size

Both methods generated the same total number of windows:

```text
rollout_v1 windows:        30,000
posterior_bank_v2 windows: 30,000
```

Each method generated:

```text
number of synthetic subjects = 10
windows per subject = 3000
```

This gives:

```text
10 synthetic subjects × 3000 windows = 30,000 synthetic windows per method
```

The saved array shapes were:

```text
ACC:      [30000, 256, 3]
BVP:      [30000, 512, 1]
EDA/TEMP: [30000, 32, 2]
```

These shapes match the expected input format for the later CNN-BiLSTM downstream notebook.

---

## 5. Activity distribution

The synthetic activity labels were sampled using:

```text
activity_sampling_mode = train_distribution
```

This means the generated activity distribution follows the real training distribution instead of forcing a fully balanced synthetic dataset.

### rollout_v1 activity counts

|   Activity |   Windows | Percent   |
|-----------:|----------:|:----------|
|          1 |      2900 | 9.67%     |
|          2 |      2051 | 6.84%     |
|          3 |      1397 | 4.66%     |
|          4 |      2190 | 7.30%     |
|          5 |      4616 | 15.39%    |
|          6 |      8628 | 28.76%    |
|          7 |      2756 | 9.19%     |
|          8 |      5462 | 18.21%    |

### posterior_bank_v2 activity counts

|   Activity |   Windows | Percent   |
|-----------:|----------:|:----------|
|          1 |      2951 | 9.84%     |
|          2 |      2108 | 7.03%     |
|          3 |      1443 | 4.81%     |
|          4 |      2204 | 7.35%     |
|          5 |      4400 | 14.67%    |
|          6 |      8778 | 29.26%    |
|          7 |      2779 | 9.26%     |
|          8 |      5337 | 17.79%    |

The dominant generated activity is activity 6, which is expected because the synthetic labels follow the real training activity distribution.

This choice is useful for realism because the synthetic dataset keeps a distribution closer to the real training data. However, for downstream experiments, it also means class imbalance remains present. The downstream classifier should therefore still report macro-F1 and per-class performance, not accuracy alone.

---

## 6. Synthetic subject conditioning

Synthetic windows were grouped into artificial subjects:

```text
synthetic_subject_01
synthetic_subject_02
...
synthetic_subject_10
```

Each synthetic subject contains 3000 windows.

The generation used:

```text
synthetic_subject_condition_mode = sample_train_subject
```

This means each artificial subject was assigned a train-subject condition token. This gives the decoder a subject-style condition without using validation or test subjects.

For rollout_v1, the subject condition map was:

```text
{
  "synthetic_subject_01": "S11",
  "synthetic_subject_02": "S3",
  "synthetic_subject_03": "S11",
  "synthetic_subject_04": "S5",
  "synthetic_subject_05": "S4",
  "synthetic_subject_06": "S11",
  "synthetic_subject_07": "S3",
  "synthetic_subject_08": "S13",
  "synthetic_subject_09": "S13",
  "synthetic_subject_10": "S12"
}
```

For posterior_bank_v2, the subject condition map was:

```text
{
  "synthetic_subject_01": "S2",
  "synthetic_subject_02": "S2",
  "synthetic_subject_03": "S6",
  "synthetic_subject_04": "S11",
  "synthetic_subject_05": "S2",
  "synthetic_subject_06": "S11",
  "synthetic_subject_07": "S11",
  "synthetic_subject_08": "S1",
  "synthetic_subject_09": "S12",
  "synthetic_subject_10": "S13"
}
```

This is useful because synthetic subjects can have different subject-style embeddings while still avoiding test-subject leakage.

---

## 7. Koopman stabilization

Notebook 02 showed that the learned activity-conditioned Koopman matrices had spectral radii greater than 1. Therefore, Notebook 03 stabilized each activity-specific Koopman matrix before generation.

The target spectral radius was:

```text
target_spectral_radius = 0.98
```

The stabilization summary was:

|   activity_idx |   raw_spectral_radius |   target_spectral_radius |   scaling_factor |   stable_spectral_radius |
|---------------:|----------------------:|-------------------------:|-----------------:|-------------------------:|
|              0 |                1.4517 |                     0.98 |           0.6751 |                     0.98 |
|              1 |                1.3931 |                     0.98 |           0.7035 |                     0.98 |
|              2 |                1.279  |                     0.98 |           0.7662 |                     0.98 |
|              3 |                1.209  |                     0.98 |           0.8106 |                     0.98 |
|              4 |                1.2876 |                     0.98 |           0.7611 |                     0.98 |
|              5 |                1.4086 |                     0.98 |           0.6957 |                     0.98 |
|              6 |                1.4173 |                     0.98 |           0.6914 |                     0.98 |
|              7 |                1.4554 |                     0.98 |           0.6733 |                     0.98 |

All activity matrices were scaled to a stable spectral radius of approximately 0.98.

This is important because raw repeated Koopman rollout with spectral radius greater than 1 can amplify latent values over time. Stabilization makes rollout generation safer while preserving the learned transition structure.

---

## 8. Visual analysis of rollout_v1

The rollout_v1 examples show that the generated signals are structured but relatively smooth.

### BVP

The rollout BVP windows show oscillatory patterns and local peaks. However, they sometimes include a strong initial transient at the beginning of the window. This may be caused by the sampled initial latent state and the stabilized rollout dynamics.

### ACC

The rollout ACC signals are smoother than real high-frequency movement signals. They show clear channel separation and some activity-dependent motion structure, but the variation is limited in some windows.

This is expected because rollout_v1 repeatedly applies a stable Koopman matrix. Stabilization prevents exploding trajectories, but it can also make generated dynamics more conservative and smooth.

### EDA/TEMP

The slow branch in rollout_v1 is smooth and slowly varying, which is reasonable for EDA/TEMP. Some windows show monotonic drift or strong initial movement in TEMP. This should be evaluated quantitatively later.

Overall, rollout_v1 is useful as a baseline because it tests direct latent rollout, but it may be less diverse and less realistic than posterior-bank generation.

---

## 9. Visual analysis of posterior_bank_v2

The posterior_bank_v2 examples show stronger variation than rollout_v1.

### BVP

The posterior-bank BVP examples have larger oscillatory structure and stronger peaks. They look less over-smoothed than rollout_v1. This suggests that sampling full posterior latent trajectories preserves more temporal detail than pure rollout.

### ACC

The posterior-bank ACC windows show larger movement patterns and more dynamic changes than rollout_v1. This is useful because ACC is activity-sensitive and should not be overly flat.

However, some examples show strong offsets or activity-specific channel biases. This is not automatically wrong because the data is normalized and activity/subject-conditioned, but it must be checked with realism/diversity metrics.

### EDA/TEMP

The posterior-bank slow branch appears very smooth, and in some examples TEMP stays almost constant at a high normalized value while EDA remains in a narrow range.

This may be realistic for short 8-second windows because EDA and temperature are slow-changing signals. However, the branch could also be under-diverse. The Realism/Diversity notebook must verify this using histogram overlap, mean/std differences, and diversity ratio.

Overall, posterior_bank_v2 appears more expressive than rollout_v1, especially for BVP and ACC. It is the stronger candidate for the final synthetic dataset, but it must be validated quantitatively.

---

## 10. Comparison between rollout_v1 and posterior_bank_v2

The two methods have different strengths.

| Aspect | rollout_v1 | posterior_bank_v2 |
|---|---|---|
| Latent source | z0 + repeated Koopman rollout | full posterior latent trajectory bank |
| Stability | controlled by stabilized Koopman rollout | more stable because it starts from realistic full trajectories |
| BVP appearance | smoother, sometimes initial transient | stronger oscillations and peaks |
| ACC appearance | smoother and conservative | more dynamic and expressive |
| Slow branch | smooth drift | very smooth, sometimes nearly constant |
| Expected diversity | lower | higher |
| Role in project | baseline / ablation | main KoVAE generation method |

The final decision should not be made from visual plots only. The next notebooks must evaluate:

```text
comparison plots
realism/diversity metrics
CNN-BiLSTM downstream activity detection
```

---

## 11. Saved outputs

Notebook 03 saved the synthetic data in the final project structure.

### rollout_v1

```text
data/synthetic_subjects/kovae/rollout_v1/generated_subjects_X_acc_32hz.npy
data/synthetic_subjects/kovae/rollout_v1/generated_subjects_X_bvp_64hz.npy
data/synthetic_subjects/kovae/rollout_v1/generated_subjects_X_slow_4hz.npy
data/synthetic_subjects/kovae/rollout_v1/generated_subjects_all_y.npy
data/synthetic_subjects/kovae/rollout_v1/generated_subjects_all_subject.npy
data/synthetic_subjects/kovae/rollout_v1/generated_subjects_metadata.csv
```

### posterior_bank_v2

```text
data/synthetic_subjects/kovae/posterior_bank_v2/generated_subjects_X_acc_32hz.npy
data/synthetic_subjects/kovae/posterior_bank_v2/generated_subjects_X_bvp_64hz.npy
data/synthetic_subjects/kovae/posterior_bank_v2/generated_subjects_X_slow_4hz.npy
data/synthetic_subjects/kovae/posterior_bank_v2/generated_subjects_all_y.npy
data/synthetic_subjects/kovae/posterior_bank_v2/generated_subjects_all_subject.npy
data/synthetic_subjects/kovae/posterior_bank_v2/generated_subjects_metadata.csv
```

The notebook also saved summaries and figures:

```text
results/kovae_generation/combined_generation_summary.json
results/kovae_generation/activity_koopman_generation_stability.csv
results/kovae_generation/rollout_v1/generation_summary.json
results/kovae_generation/posterior_bank_v2/generation_summary.json

figures/kovae_generation/rollout_v1/
figures/kovae_generation/posterior_bank_v2/
```

---

## 12. What to write in the report

A report-ready generation paragraph is:

> Synthetic native-rate windows were generated from the trained activity-conditioned KoVAE using two strategies. The rollout baseline sampled an initial latent state from an activity-specific posterior bank and applied a stabilized activity-specific Koopman matrix to generate the latent trajectory. The posterior-bank method sampled complete activity-specific latent trajectories from encoded training windows, added controlled perturbations and interpolation, and applied light Koopman guidance before decoding. Both methods used the trained activity and subject conditions during decoding and generated 30,000 synthetic windows across 10 artificial subjects.

A report-ready result paragraph is:

> Both generation methods produced synthetic datasets with the expected native-rate shapes: ACC `[30000, 256, 3]`, BVP `[30000, 512, 1]`, and EDA/TEMP `[30000, 32, 2]`. The activity labels were sampled according to the real training distribution, preserving the natural class imbalance. Activity-conditioned Koopman matrices were stabilized to a spectral radius of 0.98 before rollout generation. Visual inspection showed that rollout generation produced smoother signals, while posterior-bank generation produced more dynamic BVP and ACC patterns. Quantitative realism and downstream utility are evaluated in the following notebooks.

---

## 13. Limitations

The generation step has several limitations:

1. The synthetic subjects are assembled from generated windows, not from a learned full-subject activity protocol.
2. Activity labels follow the training distribution, so class imbalance remains.
3. Rollout generation may be over-smoothed because Koopman matrices are stabilized.
4. Posterior-bank generation can preserve more detail but may also reproduce latent biases from the training posterior bank.
5. Slow EDA/TEMP signals are difficult to judge visually because they change slowly over 8-second windows.
6. No conclusion about downstream utility can be made before CNN-BiLSTM evaluation.

---

## 14. Next step

The next notebook should be:

```text
04_comparison_kovae_final_clean.ipynb
```

It should reconstruct visual timelines for real and synthetic subjects using center-crop visualization and save comparison plots.

After that, Notebook 05 should compute realism, no-copying, and diversity metrics. Finally, the CNN-BiLSTM downstream notebook should test whether synthetic data improves activity detection on unseen real subjects.

---

## 15. Final conclusion

Notebook 03 successfully generated synthetic native-rate wearable windows using the updated activity-conditioned KoVAE.

Both `rollout_v1` and `posterior_bank_v2` generated 30,000 windows with the correct shapes and metadata. Koopman matrices were stabilized before generation, and synthetic subjects were assigned train-subject condition tokens without using validation or test subjects.

The visual examples suggest that `posterior_bank_v2` is more expressive than direct rollout, while `rollout_v1` remains useful as a Koopman baseline. The quality and utility of both methods must now be tested using quantitative realism/diversity metrics and downstream human activity detection.
