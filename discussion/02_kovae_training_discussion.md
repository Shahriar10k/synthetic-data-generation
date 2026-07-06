# 02 KoVAE Training Discussion and Analysis

## 1. Purpose of this notebook

The purpose of this notebook was to train the proposed **Koopman Variational Autoencoder (KoVAE)** model on the preprocessed PPG-DaLiA native-rate windows.

The project uses PPG-DaLiA as a multi-modal wearable time-series dataset. The preprocessed input contains three sensor branches:

```text
BVP / PPG:  [N, 512, 1]   at 64 Hz
ACC:        [N, 256, 3]   at 32 Hz
EDA/TEMP:   [N, 32, 2]    at 4 Hz
```

The goal of this notebook was not yet to generate synthetic subjects. The goal was to train a generative model that learns:

1. how to reconstruct native-rate physiological and motion windows,
2. how to represent the windows in a shared latent space,
3. how to impose Koopman-style linear latent dynamics,
4. how to save a trained checkpoint for later synthetic subject generation.

The trained checkpoint will be used in the next notebook to generate new synthetic subjects.

---

## 2. Why KoVAE is used

A normal VAE learns a latent representation and reconstructs the input. However, in time-series data, the order and temporal dynamics are important. A simple VAE may reconstruct windows but may not explicitly learn how latent states evolve over time.

KoVAE improves this by adding a Koopman-inspired latent dynamics constraint. The main idea is that the model learns a latent sequence where the next latent state can be approximated using a linear transition matrix:

```text
z_next ≈ A × z_current
```

Here:

```text
z = latent sequence
A = learned Koopman transition matrix
```

This is useful for time-series generation because the model does not only learn static reconstruction. It also learns a structured latent transition rule that can later be used for controlled synthetic generation.

---

## 3. Model architecture

The model was implemented as a **multi-branch KoVAE** because PPG-DaLiA signals have different native sampling rates.

The architecture is:

```text
BVP branch encoder      ┐
ACC branch encoder      ├── fused shared latent sequence ── Koopman dynamics ── branch decoders
EDA/TEMP branch encoder ┘
```

### 3.1 Branch encoders

Each modality has its own encoder:

```text
BVP encoder:      processes [B, 512, 1]
ACC encoder:      processes [B, 256, 3]
EDA/TEMP encoder: processes [B, 32, 2]
```

The encoders convert different native-rate signals into a common latent time resolution. This avoids forcing all raw signals into one artificial sampling rate.

### 3.2 Shared latent representation

The encoded BVP, ACC, and EDA/TEMP features are fused into one shared latent sequence.

The model also uses conditioning information:

```text
activity label embedding
subject embedding
```

Activity conditioning helps the model learn activity-specific patterns. Subject conditioning helps the model learn person-specific physiological and motion style.

### 3.3 Branch decoders

The decoder reconstructs each modality back to its original native-rate shape:

```text
BVP decoder      -> [B, 512, 1]
ACC decoder      -> [B, 256, 3]
EDA/TEMP decoder -> [B, 32, 2]
```

This means the model keeps the original multi-rate structure of the dataset.

---

## 4. Data split used during KoVAE training

The notebook used the shared subject protocol:

```text
Train subjects: S1, S2, S3, S4, S5, S6, S9, S11, S12, S13
Validation subjects: S14, S15
Test subjects reserved: S7, S8, S10
```

The number of windows used was:

```text
Training windows:   30,762
Validation windows:  6,100
Test windows:       10,063
```

The test subjects were not used during KoVAE training. They are reserved for later downstream evaluation.

This is important because the final project should evaluate whether synthetic subject generation improves generalization to unseen real subjects.

---

## 5. Loss functions used

The total KoVAE loss contains three main parts:

```text
Total loss = reconstruction loss + beta × KL loss + alpha × Koopman loss
```

In this experiment:

```text
alpha_koopman = 0.10
beta_kl = 0.001
```

The loss was designed this way because the model must satisfy three goals:

1. reconstruct the real multi-modal windows,
2. keep the latent space regularized like a VAE,
3. force the latent sequence to follow approximately linear Koopman dynamics.

---

## 6. Reconstruction loss

The reconstruction loss measures how close the reconstructed signals are to the real input signals.

Because the dataset has three signal branches, the reconstruction loss is computed separately:

```text
BVP reconstruction loss
ACC reconstruction loss
EDA/TEMP reconstruction loss
```

Then the branch losses are combined:

```text
reconstruction loss =
    BVP loss + ACC loss + SLOW loss
```

where:

```text
SLOW = EDA + TEMP
```

Mean squared error was used for reconstruction because the signals are continuous-valued time-series measurements.

### Why separate branch reconstruction loss is needed

The three branches have different shapes:

```text
BVP:      512 time steps
ACC:      256 time steps
EDA/TEMP:  32 time steps
```

If the model used only one merged loss without care, high-resolution branches could dominate the training. By keeping the branch losses explicit, we can analyze which modality is reconstructed well and which modality is harder.

At the best epoch, the validation reconstruction losses were:

```text
BVP validation loss:      0.00249
ACC validation loss:      0.02364
EDA/TEMP validation loss: 0.02083
```

This shows that BVP was reconstructed very well. ACC was also learned reasonably well. The slow EDA/TEMP branch had a larger validation loss than BVP, which suggests that slow physiological signals are more difficult to generalize across unseen validation subjects.

---

## 7. KL loss

The KL loss is the standard VAE regularization term. It pushes the approximate posterior latent distribution toward a simple prior distribution.

In simple terms, it prevents the latent space from becoming arbitrary or overly memorized.

The KL term is useful because later we want to sample from the latent space to generate synthetic data. Without KL regularization, the latent space may reconstruct training samples but may not be easy to sample from.

In this experiment, the KL loss had a small weight:

```text
beta_kl = 0.001
```

This was intentional. If the KL weight is too high, the model may over-regularize the latent space and produce blurry or weak reconstructions. A small KL weight lets the model reconstruct physiological windows well while still keeping the latent space somewhat regularized.

At the best epoch:

```text
train KL loss: 3.4041
validation KL loss: 3.3844
```

Because this term is multiplied by `0.001`, its contribution to the total loss is controlled.

---

## 8. Koopman prediction loss

The Koopman loss is the key part that makes this model a KoVAE instead of a normal VAE.

The model learns a latent sequence:

```text
z1, z2, z3, ..., zT
```

Then it estimates a linear transition matrix `A` so that:

```text
z_{t+1} ≈ A z_t
```

The Koopman loss penalizes the difference between the predicted next latent state and the actual next latent state.

In practical form:

```text
Koopman loss = MSE(A z_t, z_{t+1})
```

### Why this loss is important

The reconstruction loss only says:

```text
Can the model rebuild the input window?
```

The Koopman loss asks a different question:

```text
Does the latent sequence follow a simple transition rule over time?
```

This is important for synthetic generation. If the latent space has meaningful temporal dynamics, then new time-series windows can be generated in a more structured way.

At the best epoch:

```text
train Koopman loss: 0.01054
validation Koopman loss: 0.00501
```

The validation Koopman loss is low, meaning the latent sequence learned by the encoder is reasonably consistent with a linear Koopman transition.

From epoch 1 to the best validation epoch, the Koopman loss improved:

```text
train Koopman loss: 0.07955 -> 0.01054
validation Koopman loss: 0.01801 -> 0.00501
```

This supports the claim that the model learned a more linear latent dynamics structure during training.

---

## 9. Training results

The best validation result was achieved at:

```text
Best epoch: 78
Best validation total loss: 0.05084
```

The model size was:

```text
Total parameters: 347,446
Trainable parameters: 347,446
```

From epoch 1 to the best epoch:

```text
train total loss: 0.39147 -> 0.05921
validation total loss: 0.27555 -> 0.05084
```

Approximate loss reductions:

```text
train total loss reduced by about 84.9%
validation total loss reduced by about 81.5%
train reconstruction loss reduced by about 85.7%
validation reconstruction loss reduced by about 82.7%
```

This shows that the model learned strongly and did not collapse.

---

## 10. Training curve analysis

The training curve shows three stages.

### Stage 1: Fast early learning

During the first few epochs, the loss decreases sharply. This means the model quickly learned basic signal reconstruction patterns.

### Stage 2: Gradual improvement

After the early drop, the loss continues to improve more slowly. This is expected for deep time-series reconstruction models.

### Stage 3: Stable convergence

Near the later epochs, the validation loss fluctuates but remains low. The best validation result occurs at epoch 78.

The validation loss stays close to the training loss, which suggests that the model is not severely overfitting.

Some fluctuations in validation loss are expected because PPG-DaLiA contains noisy wearable signals, motion artifacts, and subject-specific physiological variation.

---

## 11. Reconstruction analysis

The reconstruction examples show that the model reconstructs BVP and ACC signals well.

### BVP reconstruction

The BVP reconstruction overlaps strongly with the real BVP signal. This is a good sign because BVP is a key physiological signal in PPG-DaLiA.

### ACC reconstruction

The ACC reconstruction captures the main movement trend and important changes in the signal. Some high-frequency noise is smoothed, but this is normal for autoencoder-based reconstruction.

### EDA/TEMP reconstruction

The EDA example is weaker visually, especially near the beginning of the window. However, EDA is a slow-changing signal and has a very narrow normalized range in the plot. Because of that, small numerical differences can look large visually.

The branch-wise validation loss confirms this interpretation:

```text
BVP validation loss:      0.00249
ACC validation loss:      0.02364
EDA/TEMP validation loss: 0.02083
```

The slow branch is not perfect, but it is still usable for the next synthetic generation stage.

---

## 12. Koopman matrix and eigenvalue stability

After training, the learned Koopman matrix was analyzed using its eigenvalues.

The spectral radius was:

```text
Spectral radius: 1.4776456
Maximum eigenvalue: 1.4776456 + 0j
```

The spectral radius is the largest absolute eigenvalue of the Koopman matrix.

### Interpretation

If all eigenvalues are inside the unit circle:

```text
|lambda| < 1
```

then repeated Koopman rollout is stable or decaying.

If an eigenvalue is exactly near the unit circle:

```text
|lambda| ≈ 1
```

then the corresponding latent mode can preserve information over time.

If an eigenvalue is outside the unit circle:

```text
|lambda| > 1
```

then repeated rollout can become unstable.

In this experiment, most eigenvalues are inside the unit circle, but one eigenvalue is outside:

```text
lambda_max ≈ 1.4776
```

This means the learned latent dynamics contain one unstable mode.

---

## 13. Is the stability issue a failure?

No, this is not a complete failure.

The model was trained to reconstruct fixed-length windows and to encourage linear latent transitions. It achieved good validation loss and good reconstructions.

However, the unstable eigenvalue means that if we repeatedly apply the Koopman matrix for long generation rollouts, the latent values may grow too much.

So the model is good for reconstruction and short-window generation, but long uncontrolled Koopman rollout must be handled carefully.

This is actually useful analysis because it shows that we are not treating KoVAE as a black box. We trained the model and also inspected the learned dynamics.

---

## 14. How to handle stability in synthetic generation

For the next notebook, synthetic generation should use a stabilized Koopman matrix.

A simple and practical method is spectral radius scaling.

If the learned matrix has spectral radius:

```text
rho(A) = 1.4776
```

and we want a stable target radius:

```text
rho_target = 0.98
```

then we can define:

```text
A_stable = A × (rho_target / rho(A))
```

This keeps the learned transition direction but prevents unstable growth during rollout.

For this project, Notebook 03 should save both:

```text
raw Koopman matrix
stabilized Koopman matrix
```

and report:

```text
raw spectral radius
stable spectral radius
generation strategy
```

Suggested generation strategy:

```text
Use the trained decoder and a stabilized Koopman latent rollout to generate synthetic native-rate windows for new artificial subjects.
```

---

## 15. Saved outputs from this notebook

This notebook saved:

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

figures/kovae/training_loss_curve.png
figures/kovae/reconstruction_examples.png
figures/kovae/koopman_eigenvalues.png

logs/kovae_training.log
```

The most important files for the next notebook are:

```text
models/checkpoints/kovae_best.pt
results/kovae/best_koopman_matrix.npy
results/kovae/activity_mapping.json
results/kovae/subject_mapping.json
```

---

## 16. What to report in the final report

In the report, this notebook should be described in the **Methodology** and **Model Training** sections.

Suggested report text:

> A multi-branch KoVAE was trained on native-rate PPG-DaLiA windows. Separate encoders were used for BVP, accelerometer, and slow physiological signals, and the encoded representations were fused into a shared latent sequence. The decoder reconstructed each signal branch at its original sampling rate. The training objective combined reconstruction loss, KL regularization, and a Koopman latent prediction loss. The Koopman loss encouraged the latent sequence to follow an approximately linear transition, making the model suitable for structured synthetic time-series generation.

Suggested result text:

> The model achieved its best validation loss at epoch 78 with a validation total loss of 0.05084. The validation reconstruction loss decreased from 0.27106 at epoch 1 to 0.04696 at the best epoch. BVP reconstruction was strongest, with validation BVP loss of 0.00249. The Koopman loss also decreased, indicating that the learned latent sequence became more consistent with linear latent dynamics.

Suggested stability text:

> The learned Koopman matrix was analyzed using its eigenvalues. Most eigenvalues were inside the unit circle, but the spectral radius was 1.4776 due to one unstable eigenvalue. Since eigenvalues larger than one can lead to unstable long rollouts, synthetic generation will use a stabilized Koopman matrix by scaling the spectral radius below one.

---

## 17. What to include in the presentation

Use 2 or 3 slides for this notebook.

### Slide 1: Multi-branch KoVAE architecture

Bullet points:

```text
- Native-rate input branches: BVP, ACC, EDA/TEMP
- Separate branch encoders
- Shared latent sequence
- Koopman linear latent dynamics
- Branch decoders reconstruct native-rate outputs
```

Speaker note:

> Since PPG-DaLiA has multiple sensors with different sampling rates, I used a multi-branch KoVAE. Each branch processes one signal type, then all branches are fused into a shared latent sequence. The Koopman module encourages this latent sequence to follow linear temporal dynamics.

### Slide 2: Training objective

Bullet points:

```text
Total loss = reconstruction + beta KL + alpha Koopman
Reconstruction: rebuild BVP, ACC, EDA/TEMP
KL: regularize latent space
Koopman loss: enforce z_next ≈ A z_current
```

Speaker note:

> The reconstruction loss makes the outputs match the real signals. The KL loss keeps the latent space usable for generation. The Koopman loss is the key part of KoVAE because it encourages the latent states to follow a linear transition.

### Slide 3: Results and stability

Bullet points:

```text
Best epoch: 78
Best validation loss: 0.05084
BVP reconstruction strongest
Koopman loss decreased clearly
Spectral radius: 1.4776
Stabilized Koopman rollout needed for generation
```

Speaker note:

> The model trained successfully and reached a low validation loss. The learned Koopman matrix was mostly stable, but one eigenvalue was outside the unit circle. For synthetic generation, I will stabilize the Koopman matrix before rollout to avoid exploding latent signals.

---

## 18. Limitations

The current KoVAE model has some limitations:

1. The model is trained on fixed windows, not full continuous subject recordings.
2. EDA/TEMP reconstruction is weaker than BVP reconstruction.
3. One Koopman eigenvalue is outside the unit circle, so long rollout requires stabilization.
4. The model has not yet been evaluated on synthetic subject realism or downstream activity detection.
5. The current results show reconstruction quality, but synthetic generation quality must be tested in the next notebooks.

---

## 19. Next step

The next notebook should be:

```text
03_generate_kovae_synthetic_subjects.ipynb
```

It should:

1. load `kovae_best.pt`,
2. load the learned Koopman matrix,
3. stabilize the Koopman matrix,
4. generate synthetic native-rate windows,
5. group generated windows into artificial subjects,
6. save synthetic subject arrays and metadata,
7. create preview plots,
8. save generation statistics.

The generated output should be saved in:

```text
data/synthetic_subjects/kovae/
```

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

The multi-branch KoVAE training was successful. The model learned to reconstruct native-rate physiological and motion windows and learned a latent space with Koopman-style linear transition behavior.

The training results support moving to synthetic subject generation. The main technical point for the next stage is to handle the Koopman stability issue by using a stabilized transition matrix during generation.
