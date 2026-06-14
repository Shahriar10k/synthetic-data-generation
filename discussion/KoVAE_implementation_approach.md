# Koopman Variational Autoencoder (KoVAE) for Synthetic Physiological Time-Series Generation

## 1. Objective

The objective of this project is to generate realistic synthetic multi-channel physiological signals for Human Activity Recognition (HAR) using a Koopman Variational Autoencoder (KoVAE).

The generated synthetic data should:

* Preserve physiological characteristics of the original signals.
* Preserve activity-specific patterns.
* Preserve subject-specific characteristics.
* Be useful for downstream Human Activity Recognition tasks.
* Be statistically similar to real physiological measurements.

The project uses the PPG-DaLiA dataset, which contains synchronized physiological signals collected from multiple subjects performing different activities.

---

# 2. Dataset

The dataset was preprocessed into overlapping windows.

Each sample consists of:

```text
Window length = 512 samples
Sampling frequency = 64 Hz
Duration = 8 seconds
Overlap = 75%
Stride = 128 samples = 2 seconds
```

Each window contains six physiological channels:

```text
ACC_x
ACC_y
ACC_z
BVP
EDA
TEMP
```

Each window is associated with:

```text
Activity label
Subject ID
```

The final dataset shape is:

```text
(46907, 512, 6)
```

where:

* 46907 = number of windows
* 512 = sequence length
* 6 = physiological channels

---

# 3. Why KoVAE?

Traditional Variational Autoencoders assume that latent representations do not explicitly follow temporal dynamics.

Physiological signals are inherently dynamic systems:

* Heart activity evolves over time.
* Motion evolves over time.
* Temperature evolves over time.
* Skin conductance evolves over time.

KoVAE introduces Koopman dynamics into the latent space.

Instead of learning arbitrary latent representations, KoVAE learns a latent space where future states can be predicted through a linear dynamical system.

This provides:

* Better temporal consistency.
* Better dynamical modeling.
* Better long-term signal behavior.
* More physically meaningful latent representations.

---

# 4. Core Idea of KoVAE

The model consists of three main components:

```text
Input Signal
      ↓
Encoder
      ↓
Latent State z
      ↓
Koopman Operator K
      ↓
Future Latent State
      ↓
Decoder
      ↓
Reconstructed Signal
```

The encoder compresses the physiological signal into a latent representation.

The Koopman operator learns latent dynamics:

z(t+1) = K z(t)

The decoder reconstructs the original signal from the latent representation.

---

# 5. Conditioning Strategy

The original KoVAE paper is not specifically designed for Human Activity Recognition datasets containing multiple subjects and activities.

Therefore, conditioning was introduced.

Two conditioning sources were used:

## Subject Conditioning

Each subject receives a learnable embedding vector.

The embedding captures:

* Individual physiology
* Sensor placement differences
* Personal movement patterns

This allows synthetic subjects to retain subject-specific characteristics.

---

## Activity Conditioning

Each activity receives a learnable embedding.

Activities:

```text
1
2
3
4
5
6
7
8
```

represent different physical activities.

The activity embedding helps the model generate signals consistent with the selected activity.

---

# 6. Condition Fusion

Subject and activity embeddings are concatenated:

```text
Condition Vector =
    [Subject Embedding ; Activity Embedding]
```

This condition vector is injected into:

* Encoder
* Decoder

This allows the model to learn:

```text
Same activity
Different subject

Different activity
Same subject
```

relationships.

---

# 7. Latent Representation

The encoder produces:

```text
μ
logσ²
```

which define a Gaussian latent distribution.

The latent vector is sampled using:

```text
z = μ + σ ε
```

where:

```text
ε ~ N(0,1)
```

This is the standard VAE reparameterization trick.

---

# 8. Koopman Dynamics

The Koopman layer learns a latent transition matrix:

```text
z_future = K z
```

The matrix K is trainable.

The Koopman loss encourages latent transitions to follow predictable dynamics.

This provides:

* Temporal smoothness
* Dynamic consistency
* Better modeling of physiological processes

---

# 9. Loss Function

The total loss contains three terms.

## Reconstruction Loss

Measures signal reconstruction quality.

```text
L_recon
```

---

## Koopman Prediction Loss

Measures how well latent dynamics follow Koopman evolution.

```text
L_koopman
```

---

## KL Divergence

Regularizes latent space.

```text
L_KL
```

---

Final loss:

```text
L_total =
    L_recon
    + λ1 L_koopman
    + β L_KL
```

---

# 10. Why Reconstruction Was Evaluated First

Before generating synthetic data, reconstruction quality must be verified.

If reconstruction fails:

```text
Encoder fails
Decoder fails
Latent space fails
```

and synthetic generation becomes meaningless.

Therefore reconstruction plots were examined first.

The model successfully reconstructed:

* Accelerometer channels
* BVP
* EDA
* Temperature

with low reconstruction error.

This indicates the latent representation learned meaningful physiological information.

---

# 11. Synthetic Generation Approaches

Two generation approaches were considered.

---

## Approach 1: Prior Sampling

Sample:

```text
z ~ N(0,1)
```

Then decode.

Advantages:

```text
Simple
Paper-consistent
```

Disadvantages:

```text
Weak signal quality
Overly smooth outputs
Poor diversity
```

Observed in experiments:

* Flat signals
* Low variability
* Unrealistic physiological patterns

---

## Approach 2: Posterior Latent Bank Sampling

Final selected method.

The encoder is first used on real windows.

Latent vectors are collected:

```text
z_real
```

for every activity.

A latent bank is created:

```text
Activity 1 → latent bank
Activity 2 → latent bank
...
Activity 8 → latent bank
```

Synthetic generation:

```text
Choose activity
Select latent from bank
Add small perturbation
Decode
```

Advantages:

```text
Much more realistic
Preserves activity structure
Preserves physiological dynamics
Higher diversity
```

This became the primary generation strategy.

---

# 12. Why Posterior Latent Bank Was Added

This modification is not explicitly proposed in the original paper.

It was introduced because:

1. Prior sampling produced weak synthetic signals.
2. Real latent distributions contained useful physiological structure.
3. Sampling around learned latent states improves realism.

This is a practical engineering improvement for physiological signal generation.

---

# 13. Synthetic Subjects

Synthetic subjects are generated using learned subject embeddings.

Procedure:

```text
Compute mean subject embedding
Compute embedding variance
Sample new synthetic subject embedding
Combine with activity embedding
Generate signals
```

This creates synthetic individuals that do not correspond to any real participant.

---

# 14. Evaluation Strategy

The generated data is evaluated using:

## Reconstruction Quality

Real vs reconstructed signals.

---

## Distribution Similarity

Histogram overlap.

Higher is better.

---

## KS Statistic

Kolmogorov-Smirnov distance.

Lower is better.

---

## Frequency Similarity

FFT comparison.

Measures whether generated signals preserve spectral behavior.

---

## Diversity Metrics

Nearest-neighbor distances.

Measures:

```text
Synthetic → Synthetic
Synthetic → Real
Real → Real
```

to detect:

* Mode collapse
* Low diversity
* Overfitting

---

# 15. Important Implementation Choices

Several practical modifications were made compared to the original paper.

## Added subject conditioning

Reason:

```text
Dataset contains multiple individuals.
```

---

## Added activity conditioning

Reason:

```text
Dataset contains multiple activities.
```

---

## Added posterior latent bank sampling

Reason:

```text
Improves realism.
```

---

## Added synthetic subject generation

Reason:

```text
Generate unseen subjects.
```

---

## Added HAR-oriented evaluation

Reason:

```text
Project objective is Human Activity Recognition.
```

---

# 16. Limitations

Several limitations remain.

## Synthetic signals are window-based

The model generates individual windows.

Long continuous physiological streams are not explicitly modeled.

---

## No adversarial discriminator

Unlike GAN-based approaches:

```text
No explicit realism critic.
```

---

## Prior sampling remains weak

Posterior-bank generation performs significantly better.

---

## Activity transitions are not modeled

Activities are generated independently.

Transition dynamics between activities are not learned.

---

# 17. Final Justification

The implemented KoVAE extends the original framework to a realistic Human Activity Recognition setting.

The final system:

* Models temporal dynamics through Koopman operators.
* Preserves subject information.
* Preserves activity information.
* Generates synthetic physiological signals.
* Produces more realistic outputs using posterior latent bank sampling.
* Enables quantitative realism and diversity evaluation.

The resulting synthetic dataset can be used for downstream Human Activity Recognition experiments and compared against real physiological data.
