# 02 KoVAE Training Discussion and Analysis — Updated Version

## 1. Purpose of this notebook

The purpose of this notebook was to train the final **multi-branch KoVAE** model for synthetic wearable time-series generation.

This is now the main KoVAE model in the project. The old non-conditional version was removed, so the improved model is still named simply:

```text
kovae
```

The model was trained on preprocessed PPG-DaLiA native-rate windows with three sensor branches:

```text
BVP / PPG:  [N, 512, 1]   at 64 Hz
ACC:        [N, 256, 3]   at 32 Hz
EDA/TEMP:   [N, 32, 2]    at 4 Hz
```

The goal of this notebook was to train a generative representation that can later be used to create synthetic multi-channel physiological windows for artificial subjects.

The trained model is used by Notebook 03 to generate synthetic data with two generation strategies:

```text
rollout_v1
posterior_bank_v2
```

---

## 2. What changed compared with the earlier KoVAE

The earlier KoVAE used a multi-branch encoder and decoder and applied Koopman-style latent dynamics. However, the activity structure in generated accelerometer windows was weak.

The updated model keeps the same native-rate multi-branch idea but adds stronger conditioning:

```text
activity embeddings
subject embeddings
activity classifier loss
activity-conditioned Koopman matrices
```

This means the model is not only learning a generic latent space. It is also trained to represent activity-specific and subject-specific patterns.

In simple terms:

```text
Old KoVAE:
    learn a general latent space and later sample activity-specific latent windows

Updated KoVAE:
    learn the latent space together with activity and subject conditions
```

This is important because activity labels strongly affect wearable signals. Sitting, walking, running, and stairs should not have the same accelerometer dynamics.

---

## 3. Dataset and split

The notebook used the shared subject split:

```text
Train subjects:      S1, S2, S3, S4, S5, S6, S9, S11, S12, S13
Validation subjects: S14, S15
Test subjects:       S7, S8, S10
```

The number of windows was:

```text
Training windows:      30,762
Validation windows:     6,100
Test windows reserved:  10,063
```

The test subjects were not used during KoVAE training. They remain reserved for the downstream activity detection experiments.

The subject mapping used during training was:

```text
{
  "UNK": 0,
  "S1": 1,
  "S2": 2,
  "S3": 3,
  "S4": 4,
  "S5": 5,
  "S6": 6,
  "S9": 7,
  "S11": 8,
  "S12": 9,
  "S13": 10
}
```

The activity mapping was:

```text
{
  "1": 0,
  "2": 1,
  "3": 2,
  "4": 3,
  "5": 4,
  "6": 5,
  "7": 6,
  "8": 7
}
```

---

## 4. Model architecture

The model is a **conditional multi-branch KoVAE**.

The architecture has three sensor-specific encoders:

```text
BVP encoder      -> processes [B, 512, 1]
ACC encoder      -> processes [B, 256, 3]
EDA/TEMP encoder -> processes [B, 32, 2]
```

Each branch converts its own native-rate signal into a common latent temporal resolution. These branch representations are then fused together with condition embeddings.

The decoder reconstructs the original native-rate shapes:

```text
BVP decoder      -> [B, 512, 1]
ACC decoder      -> [B, 256, 3]
EDA/TEMP decoder -> [B, 32, 2]
```

This avoids resampling all sensors into one artificial rate and keeps the physiological structure of the data.

---

## 5. Conditioning strategy

The updated model uses two types of conditioning.

### 5.1 Activity conditioning

Each activity label is mapped to a learnable embedding vector.

This helps the model learn that different activity classes should have different signal structures. For example:

```text
sitting  -> low accelerometer variation
walking  -> periodic medium movement
running  -> stronger and faster movement
stairs   -> different motion dynamics
```

The activity embedding is used during encoding and decoding.

### 5.2 Subject conditioning

The model also uses subject embeddings. The training subjects are represented as subject tokens, and an `UNK` token is available for unknown or generic subject conditions.

This helps capture subject-specific physiological and movement styles without using test subjects.

---

## 6. Activity classifier loss

The updated model adds an auxiliary activity classifier from the latent representation.

The classifier predicts the activity label from the latent mean representation. This encourages the latent space to preserve activity information.

The best validation activity accuracy was:

```text
Validation activity accuracy: 1.0000
```

At the best epoch, the activity classifier loss was very small:

```text
Train activity CE: 0.00000599
Validation activity CE: 0.00000114
```

### Important interpretation

This high activity accuracy should not be interpreted as a downstream activity recognition result.

The model receives activity conditioning during training, so the activity classifier mainly confirms that activity information is preserved in the latent representation. The real downstream evaluation is still the CNN-BiLSTM human activity detection task on unseen real test subjects.

---

## 7. Activity-conditioned Koopman dynamics

The main scientific change is that the model learns **one Koopman matrix per activity**.

Instead of using only one global transition matrix:

```text
z_next ≈ A z_current
```

the updated model learns:

```text
z_next ≈ A_activity z_current
```

So each activity has its own latent transition behavior.

This is useful because sitting, walking, running, and stairs have different temporal dynamics.

The learned activity-conditioned Koopman spectral radii were:

|   activity_encoded |   activity_label |   spectral_radius |   max_real_eigenvalue |   min_real_eigenvalue |
|-------------------:|-----------------:|------------------:|----------------------:|----------------------:|
|                  0 |                1 |            1.4517 |                1.4517 |               -0.4366 |
|                  1 |                2 |            1.3931 |                1.3931 |               -0.4196 |
|                  2 |                3 |            1.279  |                1.279  |               -0.3548 |
|                  3 |                4 |            1.209  |                1.209  |               -0.363  |
|                  4 |                5 |            1.2876 |                1.2876 |               -0.4342 |
|                  5 |                6 |            1.4086 |                1.4086 |               -0.4031 |
|                  6 |                7 |            1.4173 |                1.4173 |               -0.4477 |
|                  7 |                8 |            1.4554 |                1.4554 |               -0.446  |

The spectral radii range was:

```text
Minimum spectral radius: 1.2090
Maximum spectral radius: 1.4554
Mean spectral radius:    1.3627
```

Most eigenvalues are visually concentrated near or inside the unit circle, but each activity matrix still has a spectral radius above 1. This means raw long Koopman rollouts may still become unstable.

For Notebook 03, generation should therefore stabilize each activity-specific Koopman matrix before rollout, for example by scaling to a target spectral radius such as:

```text
target_spectral_radius = 0.98
```

---

## 8. Loss function

The updated total loss is:

```text
total loss =
    reconstruction loss
    + beta_kl × KL loss
    + alpha_koopman × activity-conditioned Koopman loss
    + activity_classifier_weight × activity classifier loss
```

The loss terms have different purposes:

```text
reconstruction loss:
    rebuild BVP, ACC, and EDA/TEMP windows

KL loss:
    regularize the latent distribution for generation

Koopman loss:
    encourage structured latent temporal dynamics

activity classifier loss:
    encourage the latent representation to preserve activity information
```

---

## 9. Training results

The best validation model was achieved at:

```text
Best epoch: 52
Best validation total loss: 0.065054
```

The model size was:

```text
Total parameters:     358,334
Trainable parameters: 358,334
```

From epoch 1 to the best epoch:

```text
Train total loss: 0.457394 -> 0.069512
Validation total loss: 0.203691 -> 0.065054
```

Approximate reductions:

```text
Train total loss reduction:          84.8%
Validation total loss reduction:     68.1%
Train reconstruction loss reduction: 83.6%
Validation reconstruction reduction: 68.6%
Train Koopman loss reduction:        89.4%
Validation Koopman loss reduction:   86.7%
```

The best-epoch validation losses were:

```text
Validation reconstruction loss: 0.060803
Validation BVP loss:            0.002106
Validation ACC loss:            0.027752
Validation EDA/TEMP loss:       0.030945
Validation KL loss:             3.501256
Validation Koopman loss:        0.007498
Validation activity CE:         0.00000114
Validation activity accuracy:   1.0000
```

---

## 10. Training curve analysis

The training curve shows that the model learned quickly during the first epochs, followed by gradual improvement and stable convergence.

The total and reconstruction losses decrease strongly from the beginning of training. The validation loss remains close to the training loss, which suggests that the model is not severely overfitting.

The best validation total loss occurs at epoch 52. After that, the validation loss fluctuates slightly and does not improve enough to beat the best checkpoint.

The activity classifier loss drops almost to zero very early. This is expected because activity conditioning is explicitly provided to the model. Its purpose here is not to replace the downstream classifier, but to keep activity information aligned with the latent representation.

---

## 11. Reconstruction analysis

The reconstruction examples show different behavior across branches.

### BVP reconstruction

BVP reconstruction is strong. The reconstructed BVP almost overlaps the real BVP in the examples. This is also supported by the low validation BVP loss:

```text
Validation BVP loss: 0.002106
```

### ACC reconstruction

ACC reconstruction captures the main movement trend and many transitions. Some high-frequency movement noise is smoothed, which is expected for an autoencoder-style model.

The validation ACC loss was:

```text
Validation ACC loss: 0.027752
```

### EDA/TEMP reconstruction

EDA/TEMP remains more difficult. The plotted EDA reconstruction has a visible offset from the real curve. This is partly because EDA is slow-changing and has a narrow normalized range, so small numerical differences can look large visually.

The validation slow-branch loss was:

```text
Validation EDA/TEMP loss: 0.030945
```

This means the slow branch is usable but remains a limitation.

---

## 12. Koopman eigenvalue analysis

The global Koopman eigenvalue plot and the activity-conditioned eigenvalue plot show that the latent dynamics are structured but not automatically stable for long rollout.

The activity-conditioned plot is more informative for this updated model because generation uses activity-specific Koopman matrices.

The spectral radii are above 1 for all activities:

```text
Activity 1: 1.4517
Activity 2: 1.3931
Activity 3: 1.2790
Activity 4: 1.2090
Activity 5: 1.2876
Activity 6: 1.4086
Activity 7: 1.4173
Activity 8: 1.4554
```

This is not a training failure. The model is trained on fixed windows and learns useful latent dynamics, but long repeated rollout can still amplify latent values.

Therefore, Notebook 03 should stabilize activity-specific Koopman matrices before rollout generation.

---

## 13. What this means for synthetic generation

The new generation notebook should use the updated checkpoint:

```text
models/checkpoints/kovae_best.pt
```

and the learned activity-conditioned Koopman matrices.

Generation should include:

```text
rollout_v1:
    sample z0 from the activity-specific posterior bank
    roll out using the stabilized activity-specific Koopman matrix

posterior_bank_v2:
    sample full posterior latent trajectories from the activity-specific bank
    add controlled noise and interpolation
    apply light activity-specific Koopman guidance
    decode with activity and subject conditions
```

Because the model is now explicitly activity-conditioned, the extra quality filter is not needed as the first default generation method. The correct next step is to generate without filtering and then evaluate:

```text
comparison plots
realism/diversity metrics
CNN-BiLSTM downstream activity detection
```

If synthetic quality is still weak after these evaluations, filtering can be introduced later as an additional post-processing method.

---

## 14. Saved outputs from this notebook

This notebook saves the main KoVAE outputs using the standard project naming:

```text
configs/kovae_config.json

models/checkpoints/kovae_best.pt
models/checkpoints/kovae_last.pt
models/generators/kovae_model_summary.json

results/kovae/training_history.csv
results/kovae/final_metrics.json
results/kovae/activity_mapping.json
results/kovae/subject_mapping.json
results/kovae/split_indices_used_for_kovae.npz
results/kovae/best_koopman_matrix.npy
results/kovae/best_activity_koopman_matrices.npy
results/kovae/activity_koopman_summary.csv

figures/kovae/training_loss_curve.png
figures/kovae/reconstruction_examples.png
figures/kovae/koopman_eigenvalues.png
figures/kovae/activity_koopman_eigenvalues.png

logs/kovae_training.log
```

The most important files for Notebook 03 are:

```text
models/checkpoints/kovae_best.pt
results/kovae/best_activity_koopman_matrices.npy
results/kovae/activity_mapping.json
results/kovae/subject_mapping.json
```

---

## 15. Report-ready methodology text

A report-ready description is:

> A multi-branch KoVAE was trained on native-rate wearable windows from PPG-DaLiA. Separate encoders were used for BVP, accelerometer, and slow physiological signals, and the encoded features were fused into a shared latent sequence. Activity and subject embeddings were added to condition the latent representation and decoder. The training objective combined reconstruction loss, KL regularization, activity-conditioned Koopman latent prediction loss, and an auxiliary activity classification loss. The Koopman component learned separate latent transition matrices for each activity class, allowing activity-specific latent dynamics during synthetic generation.

---

## 16. Report-ready result text

A report-ready result paragraph is:

> The updated KoVAE achieved its best validation total loss of 0.06505 at epoch 52. The validation reconstruction loss decreased from 0.19393 at epoch 1 to 0.06080 at the best checkpoint. The auxiliary activity classifier reached validation accuracy of 1.00, indicating that activity information was preserved in the latent representation. Reconstruction quality was strongest for BVP, with a validation BVP loss of 0.00211. The activity-conditioned Koopman matrices had spectral radii between 1.21 and 1.46; therefore, stabilized matrices are used during synthetic rollout generation.

---

## 17. Presentation notes

### Slide: Updated KoVAE model

```text
- Native-rate multi-branch encoder/decoder
- Activity and subject conditioning
- Activity classifier regularization
- One Koopman matrix per activity
- Output used for synthetic subject generation
```

Speaker note:

> The updated model is still called KoVAE in my project, but internally it is stronger than the first version. It uses activity and subject embeddings and learns separate Koopman dynamics for each activity. This is important because wearable signals are strongly activity-dependent.

### Slide: Training results

```text
Best epoch: 52
Best validation total loss: 0.06505
Validation activity accuracy: 1.00
Validation BVP loss: 0.00211
Validation ACC loss: 0.02775
Validation EDA/TEMP loss: 0.03095
```

Speaker note:

> The model converged successfully. BVP reconstruction is strongest, ACC captures the main motion structure, and EDA/TEMP remains the hardest branch. The activity classifier confirms that the latent space keeps the conditioning information.

### Slide: Koopman stability

```text
- Activity-specific Koopman matrices learned
- Spectral radii: 1.21 to 1.46
- Raw long rollout may be unstable
- Notebook 03 stabilizes matrices before generation
```

Speaker note:

> The learned activity-specific matrices are useful for activity-dependent generation, but their spectral radii are above one. Therefore, for synthetic rollout generation, I scale the matrices to a stable target radius.

---

## 18. Limitations

The updated KoVAE is stronger than the first version, but it still has limitations:

1. It is trained on fixed 8-second windows, not full continuous subject recordings.
2. EDA/TEMP reconstruction remains weaker than BVP and ACC.
3. Activity classifier accuracy is not a downstream HAR result because activity labels are provided as conditions during training.
4. Activity-specific Koopman matrices still require stabilization for long rollout.
5. Synthetic realism and downstream utility must still be evaluated in later notebooks.

---

## 19. Next step

The next notebook is:

```text
03_generate_kovae_synthetic_subjects_final.ipynb
```

It should:

1. load `models/checkpoints/kovae_best.pt`,
2. collect activity-specific posterior latent banks from train subjects,
3. stabilize each activity-conditioned Koopman matrix,
4. generate `rollout_v1` and `posterior_bank_v2`,
5. save synthetic native-rate arrays and metadata,
6. create preview plots,
7. save generation summaries.

Expected synthetic output format:

```text
generated_subjects_X_acc_32hz.npy
generated_subjects_X_bvp_64hz.npy
generated_subjects_X_slow_4hz.npy
generated_subjects_all_y.npy
generated_subjects_all_subject.npy
generated_subjects_metadata.csv
```

---

## 20. Final conclusion

The updated KoVAE training was successful. The model learned to reconstruct native-rate physiological and motion windows while preserving activity information in the latent space. The key improvement over the earlier model is the use of activity-conditioned Koopman matrices, which allow different activities to have different latent dynamics during synthetic generation.

The results support moving to synthetic subject generation. The most important technical point for the next stage is to stabilize the activity-specific Koopman matrices before rollout-based generation.
