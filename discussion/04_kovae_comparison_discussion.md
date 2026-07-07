# 04 KoVAE Comparison Discussion

## 1. Purpose of this notebook

Notebook 04 provides a qualitative visual comparison between real wearable signals and synthetic KoVAE signals.

The purpose is not to calculate final model performance. Instead, this notebook checks whether the generated synthetic windows look visually reasonable when compared with real windows from the same native-rate preprocessing pipeline.

The comparison uses the final KoVAE generation methods:

```text
rollout_v1        -> KoVAE-Rollout
posterior_bank_v2 -> KoVAE-Posterior
```

The summary file confirms that both methods were compared under the `kovae` model family, using activity-block ordering and window-based reconstruction. The notebook produced 12 comparison plot rows in total.

---

## 2. Comparison setup

The comparison used:

```text
real subject:        S1
synthetic subject:   synthetic_subject_01
pairing mode:        same_index
synthetic ordering:  activity_blocks
reconstruction:      window_based
```

The selected real subject contained:

```text
real windows:        3445
synthetic windows:   3000
```

The durations were:

```text
real spaced timeline duration:       8890.0 seconds
real window-based timeline duration: 6890.0 seconds
synthetic timeline duration:         6000.0 seconds
```

The real consecutiveness ratio was approximately:

```text
0.9980
```

This means the real windows are mostly consecutive according to the expected window shift, so the real timeline is a meaningful continuous reference.

---

## 3. Why three timelines are shown

Each plot has three rows.

### 3.1 Real spaced timeline

This is the closest visual reconstruction of the original subject recording. It uses the real metadata start positions and overlap-add reconstruction.

This row is useful because it shows the true activity protocol of the real subject, including gaps and activity blocks.

### 3.2 Real window-based timeline

This row uses the same window-based visualization strategy that is applied to synthetic data. It compresses the real windows into a continuous-looking sequence by taking the stable central part of each window.

This row is important because it gives a fairer comparison with the synthetic window-based timeline.

### 3.3 Synthetic window-based timeline

The synthetic row shows generated windows from one artificial subject. Since the generated synthetic windows are not guaranteed to form a true continuous recording, the synthetic windows were ordered by activity blocks before visualization.

This means the synthetic timeline is easier to interpret:

```text
activity 1 -> activity 2 -> activity 3 -> ... -> activity 8
```

However, this is still only a qualitative visualization. It should not be interpreted as a true long continuous sensor recording.

---

## 4. Why activity-block ordering was used

The first version of the comparison plotted synthetic windows in saved order. That was misleading because the generated windows were independently sampled and the saved order did not represent a real activity protocol.

The updated comparison uses:

```text
synthetic_plot_order_mode = activity_blocks
```

This groups windows by activity label before plotting. The advantage is that we can see whether the synthetic signal changes across activity classes.

This is especially useful for ACC and BVP, because those channels should show different behavior under different activities.

---

## 5. KoVAE-Rollout visual interpretation

### 5.1 ACC

The KoVAE-Rollout ACC plot is too smooth and low-amplitude compared with the real ACC signal.

The real ACC signal shows large changes across activities, especially in active movement periods. In contrast, the rollout synthetic ACC stays close to a narrow band for most of the timeline.

This suggests that direct Koopman rollout is overly conservative. The stabilization of the Koopman matrix prevents unstable latent growth, but it also reduces the strength of generated movement dynamics.

### 5.2 BVP

The KoVAE-Rollout BVP plot is also very flat compared with the real BVP signal. In the real signal, BVP contains strong variation and spikes. The rollout synthetic BVP has much smaller amplitude and weaker visible structure.

Part of this visual difference can be affected by the shared y-axis, because real BVP contains large outliers. However, even with that caveat, rollout appears under-dynamic.

### 5.3 EDA

The KoVAE-Rollout EDA plot is smoother than the posterior method, but it lacks the large physiological EDA responses visible in the real subject. It shows some activity-block changes, but the dynamic range is much smaller than real EDA.

Overall, KoVAE-Rollout is useful as a baseline, but it should not be treated as the strongest synthetic generation method.

---

## 6. KoVAE-Posterior visual interpretation

### 6.1 ACC

The KoVAE-Posterior ACC plot is clearly better than KoVAE-Rollout. It has stronger amplitude, more activity-dependent variation, and more realistic movement intensity.

The synthetic ACC is not identical to the real signal, but it captures the important idea that different activity blocks should have different motion patterns.

This is important because ACC is one of the most useful channels for human activity recognition.

### 6.2 BVP

The KoVAE-Posterior BVP plot is also better than rollout. It contains more variation and visible spikes, while rollout was almost flat.

However, the posterior BVP still does not fully match the structure of the real BVP. The real signal has clearer activity-dependent intensity changes and strong physiological/artifact variation. The posterior BVP is closer than rollout but still imperfect.

### 6.3 EDA

The KoVAE-Posterior EDA plot shows a major limitation. It contains much stronger variation than rollout, but it looks too noisy compared with the real EDA.

Real EDA changes slowly and contains smoother rises, decays, and occasional physiological responses. The posterior synthetic EDA contains many rapid fluctuations after stitching the generated windows.

This does not necessarily mean every individual synthetic EDA window is bad. It means the generated windows do not preserve smooth long-term EDA continuity when stitched into a long timeline.

---

## 7. Main comparison between rollout and posterior generation

| Channel | KoVAE-Rollout | KoVAE-Posterior | Interpretation |
|---|---|---|---|
| ACC | Too smooth and low-amplitude | More dynamic and activity-dependent | Posterior is better for movement signals |
| BVP | Nearly flat in long view | More variable, closer to real | Posterior is better but still imperfect |
| EDA | Smooth but weak | More expressive but noisy | Slow branch remains difficult |
| Overall | Stable but over-smoothed | More realistic dynamics | Posterior is stronger final candidate |

The visual comparison supports using KoVAE-Posterior as the main KoVAE synthetic dataset and KoVAE-Rollout as a baseline or ablation.

---

## 8. Why Koopman did not guarantee smooth long-term EDA/TEMP

This result does not mean the Koopman idea was useless. The issue is the time scale of the learned dynamics.

The Koopman component in this KoVAE learns short-term latent transitions inside an 8-second window:

```text
z_t -> z_(t+1) within a window
```

It does not learn a full subject-level protocol:

```text
window 1 -> window 2 -> window 3 -> ... over many minutes
```

This matters most for EDA and TEMP because they are slow physiological signals. Their most realistic behavior is long-term drift and recovery over minutes, not only short-window variation.

Therefore, the comparison plots show that the current KoVAE captures some window-level dynamics, especially for ACC and BVP, but it does not guarantee smooth long-term cross-window continuity for EDA/TEMP.

---

## 9. Important limitation of this comparison

This comparison is qualitative. It helps us inspect signal shape, amplitude, and activity-block behavior, but it is not enough to decide whether the synthetic data is useful.

The synthetic data will be used as windows for downstream activity recognition. The downstream classifier does not require a perfect continuous multi-hour synthetic recording. It requires realistic and useful training windows.

Therefore, the final decision must use:

```text
Notebook 05: realism, diversity, and no-copying metrics
Notebook 06: downstream CNN-BiLSTM human activity detection
```

---

## 10. Report-ready discussion paragraph

A report-ready version is:

> The qualitative comparison shows clear differences between the two KoVAE generation strategies. The direct Koopman rollout method produced stable but overly smooth signals, especially for ACC and BVP, suggesting that the stabilized Koopman rollout is too conservative for long visual reconstruction. The posterior-bank method produced more expressive ACC and BVP patterns and showed clearer activity-dependent variation, making it the stronger candidate for downstream activity recognition. However, the slow EDA/TEMP branch remained difficult: real EDA/TEMP signals show smooth long-term physiological trends, while the synthetic window-based timeline can appear noisy or discontinuous. This limitation is expected because the model generates short windows and learns short-term latent dynamics rather than full subject-level physiological continuity. Therefore, these plots are used as qualitative diagnostics, while final conclusions are based on realism/diversity metrics and downstream activity detection performance.

---

## 11. What figures to include in the final report

Recommended figures:

```text
1. KoVAE-Posterior ACC_x comparison
2. KoVAE-Posterior BVP comparison
3. KoVAE-Rollout ACC_x or BVP comparison as baseline
4. One EDA comparison plot as a limitation figure
```

Do not include all 12 plots in the main report. Put extra plots in the appendix if needed.

---

## 12. Final conclusion

Notebook 04 shows that the updated visualization is now more interpretable because synthetic windows are grouped by activity blocks.

The main finding is:

```text
KoVAE-Posterior is visually stronger than KoVAE-Rollout.
```

KoVAE-Rollout is stable but over-smoothed. KoVAE-Posterior preserves more useful movement and BVP variability, especially for activity-related channels. The slow EDA/TEMP branch remains the weakest part because smooth long-term physiological continuity is not explicitly modeled.

The next step is to run quantitative realism/diversity evaluation and downstream activity recognition to test whether the visually stronger posterior synthetic data is also useful for classification.
