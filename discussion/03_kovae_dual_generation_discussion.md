# 03 KoVAE Synthetic Subject Generation Discussion  
## Rollout v1 vs Posterior Bank v2

## 1. Purpose of this notebook

The purpose of this notebook was to generate synthetic multi-channel physiological and motion windows using the trained multi-branch KoVAE model.

This notebook is important because the project is not only about reconstructing real signals. The final goal is to generate useful synthetic subjects and test whether these synthetic subjects improve human activity detection as a downstream task.

The notebook keeps two KoVAE-based generation implementations:

```text
v1 = rollout_v1
v2 = posterior_bank_v2
```

Both methods generate synthetic windows with the same native-rate output format:

```text
BVP / PPG:  [N, 512, 1]   at 64 Hz
ACC:        [N, 256, 3]   at 32 Hz
EDA/TEMP:   [N, 32, 2]    at 4 Hz
```

Both methods save synthetic subjects separately so that they can be evaluated independently using realism metrics, diversity metrics, and downstream activity classification.

---

## 2. Why two generation implementations were kept

The first generation method, `rollout_v1`, was the direct Koopman rollout approach. It used a sampled initial latent state and repeatedly applied the stabilized Koopman matrix to generate a full latent sequence.

However, after visual inspection, this method produced signals that were technically valid but too narrow and too average. The histograms showed that the synthetic distributions were much sharper than the real distributions. This suggests low diversity or partial synthetic collapse.

Because of this, a second method was implemented: `posterior_bank_v2`.

The second method uses full posterior latent trajectories from real training windows. These trajectories are sampled by activity class, interpolated, perturbed, guided slightly by the Koopman matrix, and then decoded into synthetic native-rate sensor signals.

So the two methods are kept for a clear experimental comparison:

```text
rollout_v1:
    pure stabilized Koopman rollout
    useful as first attempt / ablation

posterior_bank_v2:
    posterior trajectory sampling with Koopman guidance
    used as improved main KoVAE synthetic generation method
```

This is useful for the final report because it shows an actual research process:

```text
initial generation -> quality problem discovered -> improved generation -> compare impact
```

---

## 3. Shared model and input files

Both methods use the same trained KoVAE model from Notebook 02:

```text
models/checkpoints/kovae_best.pt
```

Both methods also use the same learned Koopman matrix:

```text
results/kovae/best_koopman_matrix.npy
```

Both methods load the same preprocessed native-rate real dataset:

```text
data/processed/native_rates/
```

This keeps the comparison fair. The only difference between v1 and v2 is the synthetic latent generation strategy.

---

## 4. Shared generated output size

Both methods generated the same number of synthetic samples:

```text
Number of synthetic subjects: 10
Windows per synthetic subject: 500
Total generated windows: 5000
```

Both methods produced the same output shapes:

```text
BVP:      [5000, 512, 1]
ACC:      [5000, 256, 3]
EDA/TEMP: [5000, 32, 2]
Labels:   [5000]
Subjects: [5000]
```

The generated outputs are normalized, because the model was trained on normalized preprocessed windows.

---

## 5. Koopman stability issue

The trained Koopman matrix had the following spectral radius:

```text
Raw spectral radius: 1.4776456
```

The spectral radius is the largest absolute eigenvalue of the Koopman matrix. If the spectral radius is larger than one, repeated Koopman rollout can become unstable because one or more latent components may grow over time.

To prevent unstable rollout, the matrix was stabilized by scaling it to a target spectral radius:

```text
Target spectral radius: 0.98
Stable spectral radius: 0.9800000
Scaling factor: 0.6632172
```

The stabilization rule was:

```text
A_stable = A_raw × (target_radius / raw_spectral_radius)
```

In this experiment:

```text
A_stable = A_raw × (0.98 / 1.4776456)
```

This stabilized matrix was used in both generation methods.

### Why this matters

The raw Koopman matrix is learned from the latent dynamics of training windows. However, because one eigenvalue is outside the unit circle, long rollout with the raw matrix may cause exploding latent states.

The stabilized matrix avoids this problem. However, strong stabilization can also make the rollout too conservative. This is one reason why the first method, `rollout_v1`, generated narrow and average-looking synthetic signals.

---

## 6. Implementation 1: rollout_v1

### 6.1 Method idea

The first method generates latent sequences by starting from one initial latent state and rolling forward using the stabilized Koopman matrix.

The process is:

```text
1. Encode real train windows using the KoVAE encoder.
2. Store the first latent state z0 for each activity.
3. Sample an activity label.
4. Sample one z0 from the activity-specific z0 bank.
5. Add small noise to z0.
6. Generate the latent sequence using stabilized Koopman rollout.
7. Decode the latent sequence using the KoVAE decoder.
8. Save synthetic BVP, ACC, EDA/TEMP, label, subject ID, and metadata.
```

In simplified form:

```text
z0 -> A_stable z0 -> A_stable z1 -> A_stable z2 -> ... -> decoder
```

### 6.2 Why this is a reasonable first method

This method directly uses the Koopman matrix during generation. It is a clean and interpretable way to test whether the learned Koopman dynamics can produce realistic synthetic time-series windows.

It is also useful as an ablation because it answers this question:

```text
Can stabilized Koopman rollout alone generate realistic wearable signals?
```

### 6.3 Parameters used

The important rollout parameters were:

```text
rollout_latent_noise_scale = 0.03
rollout_z0_noise_scale = 0.05
target_spectral_radius = 0.98
```

### 6.4 Saved folder

The rollout v1 outputs are saved in:

```text
data/synthetic_subjects/kovae/rollout_v1/
results/kovae_generation/rollout_v1/
figures/kovae_generation/rollout_v1/
```

### 6.5 v1 generated data summary

```text
Generation method: rollout_v1
Generated windows: 5000
Synthetic subjects: 10
Windows per subject: 500
```

Output shapes:

```text
BVP:      [5000, 512, 1]
ACC:      [5000, 256, 3]
EDA/TEMP: [5000, 32, 2]
```

Activity distribution:

```text
{
  "1": 490,
  "2": 334,
  "3": 232,
  "4": 377,
  "5": 745,
  "6": 1431,
  "7": 489,
  "8": 902
}
```

### 6.6 v1 visual analysis

The rollout v1 generation was technically successful, but the synthetic data quality was weak.

The histogram comparison showed that:

```text
real distributions      = wider
v1 synthetic distributions = very narrow and concentrated
```

This was especially visible for BVP and ACC. The synthetic histograms formed sharp peaks near the center of the normalized range.

The example plots showed:

```text
BVP:
    low diversity and repeated starting artifacts

ACC:
    smooth, plateau-like behavior

EDA/TEMP:
    too smooth and too regular
```

This suggests that rollout v1 suffers from a kind of latent collapse. The generated windows are valid arrays, but they do not fully represent the variability of the real wearable signals.

### 6.7 Interpretation of v1

The main issue is not the trained KoVAE model itself. The issue is the generation strategy.

Because the raw Koopman matrix had spectral radius larger than one, it was scaled to 0.98 for stability. This prevents exploding rollouts, but it also makes the latent trajectory contract toward average behavior. With small noise, this produces conservative synthetic signals.

So v1 should be treated as:

```text
a useful baseline / ablation,
not the final best synthetic generation method.
```

---

## 7. Implementation 2: posterior_bank_v2

### 7.1 Method idea

The second method improves synthetic generation by using full posterior latent trajectories instead of only the first latent state.

The process is:

```text
1. Encode all real train windows using the KoVAE encoder.
2. Store the full posterior mean latent sequence μ for each window.
3. Group these posterior trajectories by activity label.
4. Sample an activity label.
5. Sample a full latent trajectory from the matching activity posterior bank.
6. Optionally interpolate it with another trajectory from the same activity.
7. Add controlled latent noise.
8. Apply a small Koopman-guided blend for temporal consistency.
9. Decode the modified latent trajectory using the KoVAE decoder.
10. Save synthetic BVP, ACC, EDA/TEMP, label, subject ID, and metadata.
```

In simplified form:

```text
real train window -> encoder -> posterior trajectory bank
posterior trajectory -> interpolation/noise -> Koopman-guided blend -> decoder
```

### 7.2 Why posterior trajectory bank is better

The KoVAE decoder was trained mostly on latent trajectories produced by the encoder. Therefore, the decoder expects latent sequences that look like encoded real windows.

The v1 method creates full latent sequences by repeated rollout. These sequences may become too smooth or too centered after stabilization.

The v2 method samples from the actual learned latent manifold:

```text
real-like latent trajectory -> controlled perturbation -> decode
```

This keeps more realistic signal diversity while still using the Koopman-shaped latent space.

### 7.3 Is v2 still KoVAE?

Yes, v2 is still KoVAE-based because it uses:

```text
KoVAE-trained encoder
KoVAE-trained decoder
latent space trained with Koopman loss
learned Koopman matrix
stabilized Koopman guidance
activity-conditioned posterior trajectories
subject-conditioned decoding
```

However, it should be described carefully. The best wording is:

```text
KoVAE posterior-trajectory generation with Koopman-guided latent perturbation
```

or:

```text
activity-conditioned posterior trajectory bank with stabilized Koopman guidance
```

Avoid describing it as pure Koopman rollout, because v2 does not rely only on long rollout from z0.

### 7.4 Parameters used

The important v2 parameters were:

```text
posterior_noise_scale = 0.06
posterior_interpolation_prob = 0.70
posterior_koopman_blend_weight = 0.15
target_spectral_radius = 0.98
```

### 7.5 Saved folder

The posterior bank v2 outputs are saved in:

```text
data/synthetic_subjects/kovae/posterior_bank_v2/
results/kovae_generation/posterior_bank_v2/
figures/kovae_generation/posterior_bank_v2/
```

### 7.6 v2 generated data summary

```text
Generation method: posterior_bank_v2
Generated windows: 5000
Synthetic subjects: 10
Windows per subject: 500
```

Output shapes:

```text
BVP:      [5000, 512, 1]
ACC:      [5000, 256, 3]
EDA/TEMP: [5000, 32, 2]
```

Activity distribution:

```text
{
  "1": 475,
  "2": 382,
  "3": 218,
  "4": 377,
  "5": 755,
  "6": 1469,
  "7": 478,
  "8": 846
}
```

### 7.7 v2 visual analysis

The posterior bank v2 generation was visually better than rollout v1.

The histogram comparison showed that v2 synthetic distributions overlap more with the real distributions. The improvement is especially clear for ACC and EDA/TEMP.

The generated examples showed:

```text
BVP:
    stronger temporal variation than v1

ACC:
    more movement-like variation and less plateau collapse

EDA/TEMP:
    more natural slow signal behavior than v1
```

This means v2 improves synthetic diversity and reduces the narrow-distribution problem from v1.

### 7.8 Remaining limitations of v2

v2 is better, but it is not perfect.

Some generated examples still show:

```text
BVP:
    sometimes too periodic or too strong

ACC:
    sometimes high-amplitude movement

EDA/TEMP:
    sometimes shifted baseline
```

Therefore, v2 should not be considered proven only from visual plots. It must still be evaluated using quantitative realism/diversity metrics and downstream activity classification.

---

## 8. Comparison of v1 and v2

| Aspect | rollout_v1 | posterior_bank_v2 |
|---|---|---|
| Latent source | sampled z0 only | full posterior latent trajectory |
| Koopman use | long stabilized rollout | short Koopman-guided blending |
| Diversity | low | better |
| Histogram match | weak | improved |
| Signal realism | too smooth / average | more realistic |
| Main weakness | collapse due to conservative rollout | may still contain baseline shifts or strong oscillation |
| Best use | ablation / first attempt | main KoVAE synthetic data |
| Recommended for final downstream augmentation | no, only compare | yes |

---

## 9. Why v2 should be the main method

The final project goal is not only to produce arrays with the correct shape. The goal is to generate synthetic data that is realistic and useful for downstream human activity recognition.

v1 produced valid synthetic arrays, but the visual and histogram analysis showed that the generated distributions were too narrow. This may reduce downstream utility because the classifier could learn unrealistic average patterns.

v2 produced more diverse and more realistic synthetic signals. Therefore, v2 should be used as the main KoVAE synthetic dataset.

However, v1 should still be kept as an ablation experiment because it explains why posterior trajectory generation was needed.

Final interpretation:

```text
rollout_v1:
    technically successful but limited diversity

posterior_bank_v2:
    better realism and diversity, selected as main KoVAE generation method
```

---

## 10. Files saved by both methods

Each method folder contains:

```text
generated_subjects_X_bvp_64hz.npy
generated_subjects_X_acc_32hz.npy
generated_subjects_X_slow_4hz.npy
generated_subjects_all_y.npy
generated_subjects_all_subject.npy
generated_subjects_metadata.csv
generated_subjects_native_rate_arrays.npz
```

The metadata file includes:

```text
synthetic_global_id
synthetic_subject
window_in_subject
activity_encoded
activity_label
generation_method
generation_mode
raw_spectral_radius
stable_spectral_radius
koopman_scaling_factor
subject style information
```

This metadata is important for later analysis because it allows generated windows to be traced back to the generation method and synthetic subject identity.

---

## 11. Figures saved by both methods

Each method saves the following figures:

```text
raw_vs_stable_koopman_eigenvalues.png
synthetic_subject_examples.png
synthetic_activity_distribution.png
real_vs_synthetic_value_histograms.png
```

These plots are used for early qualitative evaluation.

### Important interpretation

The plots should not be treated as the final proof of quality. They are diagnostic plots. Final quality should be judged using:

```text
realism metrics
diversity metrics
downstream classifier performance
```

---

## 12. How to use both methods in downstream classification

The downstream classifier should be trained and evaluated under the following settings:

| Experiment | Training data | Test data | Purpose |
|---|---|---|---|
| E1 | Real train only | Real test subjects | Baseline |
| E2 | rollout_v1 synthetic only | Real test subjects | Utility of v1 alone |
| E3 | posterior_bank_v2 synthetic only | Real test subjects | Utility of v2 alone |
| E4 | Real train + rollout_v1 synthetic | Real test subjects | Augmentation with v1 |
| E5 | Real train + posterior_bank_v2 synthetic | Real test subjects | Augmentation with v2 |

The main comparison should be:

```text
Real only
vs
Real + rollout_v1
vs
Real + posterior_bank_v2
```

The test set should remain the reserved real test subjects:

```text
S7, S8, S10
```

This avoids evaluating synthetic data on synthetic test data, which would not prove real-world utility.

---

## 13. Downstream metrics to report

For human activity recognition, report:

```text
accuracy
macro F1
weighted F1
per-class precision
per-class recall
per-class F1
confusion matrix
```

Macro F1 is especially important because the activity labels are imbalanced.

If v2 is useful, the expected pattern is:

```text
Real + posterior_bank_v2 >= Real only
Real + posterior_bank_v2 > Real + rollout_v1
```

Even if v2 does not beat real-only, it may still be acceptable if it performs close to real-only and clearly better than v1.

---

## 14. What to write in the final report

### Methodology text

> Two KoVAE-based synthetic generation strategies were implemented. The first strategy, rollout_v1, sampled an activity-conditioned initial latent state and generated the remaining latent sequence through stabilized Koopman rollout. The second strategy, posterior_bank_v2, sampled full activity-conditioned posterior latent trajectories from the KoVAE encoder, applied interpolation and controlled perturbation, and used a stabilized Koopman-guided blend before decoding. Both methods used the same trained KoVAE checkpoint, the same subject split, and the same native-rate output format.

### Koopman stability text

> The learned Koopman matrix had a spectral radius of 1.4776, indicating one unstable latent mode for long rollout. Therefore, the Koopman matrix was stabilized by scaling its spectral radius to 0.98 before generation. This prevented exploding latent rollouts. However, the first rollout-based method produced narrow synthetic distributions, motivating the posterior trajectory bank method.

### Result text

> Both methods generated 5,000 synthetic windows from 10 artificial subjects. The output shapes matched the real preprocessed native-rate format: BVP [5000, 512, 1], ACC [5000, 256, 3], and EDA/TEMP [5000, 32, 2]. Qualitative comparison showed that rollout_v1 produced overly narrow synthetic distributions, while posterior_bank_v2 produced wider distributions and more realistic temporal variation. Therefore, posterior_bank_v2 was selected as the main KoVAE synthetic dataset, while rollout_v1 was retained as an ablation.

---

## 15. What to include in the presentation

Use one slide for the two generation methods.

### Slide title

```text
KoVAE Synthetic Subject Generation
```

### Slide bullets

```text
- v1: stabilized Koopman rollout from sampled z0
- v2: posterior trajectory bank with Koopman-guided perturbation
- Both generated 10 synthetic subjects and 5000 windows
- v1 was stable but produced narrow distributions
- v2 produced better diversity and more realistic signal patterns
- Both methods will be compared in downstream activity classification
```

### Speaker note

> I first tried a direct Koopman rollout approach. It was technically stable after spectral radius scaling, but the generated signals were too narrow and average-looking. Therefore, I added a second method using activity-conditioned posterior latent trajectories from the KoVAE encoder. This kept the generation closer to the learned real latent manifold while still using Koopman-guided temporal structure. The posterior bank version is visually more realistic and will be used as the main KoVAE synthetic dataset.

---

## 16. Limitations

The current generation analysis has several limitations:

1. The comparison is currently based mainly on visual plots and histograms.
2. v2 improves diversity but may still produce some over-regular or high-amplitude signals.
3. The posterior bank method is closer to the learned latent manifold, but it may also preserve some training-set structure.
4. Realism must be evaluated quantitatively.
5. Utility must be evaluated using a downstream classifier on real test subjects.
6. Synthetic data should not be judged only by reconstruction or plots.

---

## 17. Next steps

The next notebooks should evaluate both synthetic methods.

### Realism and diversity evaluation

Compare:

```text
real train data vs rollout_v1 synthetic
real train data vs posterior_bank_v2 synthetic
```

Possible metrics:

```text
mean/std comparison
histogram distance
correlation structure
PCA/t-SNE/UMAP visualization
MMD or distribution distance
diversity score
nearest-neighbor distance
```

### Downstream activity classification

Train the same classifier under:

```text
real only
synthetic only v1
synthetic only v2
real + v1
real + v2
```

Then evaluate on:

```text
real test subjects S7, S8, S10
```

This will show whether synthetic data is useful for human activity detection.

---

## 18. Final conclusion

The dual-method generation notebook is useful because it keeps both the initial KoVAE rollout method and the improved posterior bank method.

The first method, rollout_v1, demonstrates that pure stabilized Koopman rollout is not sufficient for realistic wearable data generation because it tends to collapse toward narrow synthetic distributions.

The second method, posterior_bank_v2, improves realism by sampling full activity-conditioned posterior latent trajectories and applying controlled perturbation with Koopman guidance.

Therefore:

```text
Use rollout_v1 as an ablation.
Use posterior_bank_v2 as the main KoVAE synthetic generation method.
Compare both in realism/diversity metrics and downstream classification.
```
