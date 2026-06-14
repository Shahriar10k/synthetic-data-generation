# VAE Evaluation Notes

## Why We Performed Two Types of Evaluation

After training the VAE, we performed two different evaluations:

1. Reconstruction Report (Qualitative Evaluation)
2. Full Test Set Evaluation (Quantitative Evaluation)

Both are important because they answer different questions.

---

# 1. Reconstruction Report (Qualitative Evaluation)

## Code

```python
batch = next(iter(test_loader))[0].to(device)

model.eval()

with torch.no_grad():
    reconstruction, mu, logvar = model(batch)
```

## What does this do?

This takes the first batch from the test set.

Example:

```python
batch.shape
```

Output:

```python
(64, 6, 512)
```

Meaning:

* 64 windows
* 6 physiological channels
* 512 time steps

The model then reconstructs these windows.

---

## Plotting a Sample

```python
sample_idx = 0

original = batch[sample_idx].cpu().numpy()
reconstructed = reconstruction[sample_idx].cpu().numpy()

plt.plot(original[0])
plt.plot(reconstructed[0])
```

This plots the BVP channel.

Channel mapping:

```text
0 = BVP
1 = ACC_X
2 = ACC_Y
3 = ACC_Z
4 = EDA
5 = TEMP
```

---

## Purpose

This answers:

> Can the model visually reconstruct physiological signals?

The reconstruction plot allows us to inspect:

* Waveform shape
* Peaks and valleys
* Amplitude
* Signal trends

---

## Example Interpretation

### Bad Reconstruction

```text
Original BVP:
Large oscillations

Reconstructed BVP:
Almost flat line
```

Meaning:

```text
Model is not capturing signal dynamics.
```

---

### Good Reconstruction

```text
Original BVP:
Waveform visible

Reconstructed BVP:
Similar waveform visible
```

Meaning:

```text
Model learned meaningful physiological patterns.
```

---

# 2. Full Test Set Evaluation (Quantitative Evaluation)

## Code

```python
for batch in test_loader:
    ...
```

Instead of using one batch, this evaluates ALL test windows.

Example:

```text
~461 windows
```

are evaluated.

---

## Purpose

This answers:

> How good is the model overall?

Instead of looking at one reconstruction, we measure average performance across the entire test set.

---

## Metrics

### Test Loss

```python
Test Loss
```

Overall objective value.

For β-VAE:

```python
Test Loss =
Recon Loss + β × KL Loss
```

Lower is better.

---

### Reconstruction Loss

```python
Recon Loss
```

Measures how accurately the reconstructed signal matches the original signal.

Lower is better.

---

### KL Divergence Loss

```python
KL Loss
```

Measures how closely the latent space follows a Gaussian distribution.

Lower values indicate stronger regularization.

Higher values indicate the latent space is storing more information.

---

# Why Both Evaluations Are Needed

Suppose:

```text
One reconstruction plot looks excellent.
```

That only proves the model works for that particular example.

We do not know how it performs on all other test windows.

---

Similarly:

Suppose:

```text
Test Loss = 0.13
```

That only tells us the average performance.

It does not tell us:

* Whether BVP is reconstructed correctly
* Whether peaks are preserved
* Whether signal shape is realistic

---

Therefore:

## Reconstruction Report

Answers:

```text
How does the reconstruction look?
```

This is qualitative evaluation.

---

## Full Test Evaluation

Answers:

```text
How good is the model overall?
```

This is quantitative evaluation.

---

# Additional Diagnostic Statistics

## Signal Range

```python
print(reconstructed[0].min(), reconstructed[0].max())
print(original[0].min(), original[0].max())
```

Purpose:

Check whether reconstructed signals have realistic amplitudes.

Example:

```text
Original:
-1.74 → 2.51

Reconstructed:
-0.09 → 0.06
```

Meaning:

```text
Reconstruction is nearly flat.
```

---

# Global Mean and Standard Deviation

```python
print(batch.mean().item())
print(batch.std().item())

print(reconstruction.mean().item())
print(reconstruction.std().item())
```

Purpose:

Compare the statistical distribution of the reconstructed data against the original data.

---

# Channel-wise Standard Deviation

```python
for i, name in enumerate(channels):
    ...
```

Purpose:

Determine which physiological channels are reconstructed well.

Example:

```text
BVP:
Poor reconstruction

EDA:
Good reconstruction
```

This helps identify weaknesses in the model.

---

# Latent Space Statistics

```python
print(mu.mean().item())
print(mu.std().item())

print(logvar.mean().item())
print(logvar.std().item())
```

Purpose:

Understand how the encoder uses the latent space.

---

## mu.mean()

Checks whether the latent space remains centered around zero.

Desired:

```text
Close to 0
```

---

## mu.std()

Measures how much information is stored in the latent space.

Small value:

```text
Possible posterior collapse.
```

Large value:

```text
Latent space actively used.
```

---

## logvar.mean()

Represents average uncertainty.

More negative values:

```text
Lower variance
Higher confidence
```

---

## logvar.std()

Measures variation in uncertainty across latent dimensions.

---

# Best Model So Far

Configuration:

```text
Latent Dimension = 128
Beta = 0.01
Epochs = 75
```

Results:

```text
Test Loss      = 0.134562
Recon Loss     = 0.117443
KL Loss        = 1.711989
```

Current Status:

```text
✓ Data pipeline complete
✓ Windowing complete
✓ Normalization complete
✓ Train/Test split complete
✓ VAE complete
✓ Reconstruction successful
✓ Latent space functioning correctly
✓ Strong baseline established
```

# Interpretation of Best Model Results

## Best Model Configuration

```text
Latent Dimension = 128
Beta = 0.01
Epochs = 75
```

---

# Full Test Set Evaluation

```text
Test Loss      = 0.134562
Test Recon Loss= 0.117443
Test KL Loss   = 1.711989
```

## What does this mean?

The Full Test Set Evaluation measures performance across every window in the test dataset.

Think of it like an exam:

```text
Reconstruction Report
=
Inspecting one student's answer sheet

Full Test Evaluation
=
Calculating the average score of the entire class
```

The model achieves:

```text
Test Loss = 0.1346
```

which indicates strong overall performance on unseen physiological data.

The reconstruction error is:

```text
Recon Loss = 0.1174
```

showing that the model can accurately rebuild physiological windows that it has never seen during training.

The KL divergence is:

```text
KL Loss = 1.712
```

which indicates that the latent space is actively storing useful information instead of collapsing into a trivial representation.

---

# Latent Space Statistics

```text
mu mean      : 0.004659
mu std       : 0.908471

logvar mean  : -3.394615
logvar std   : 1.181361
```

---

## Intuition

The latent space is the compressed representation produced by the encoder.

```text
Input Window
      ↓
Encoder
      ↓
Latent Space
      ↓
Decoder
      ↓
Reconstruction
```

---

### mu mean

```text
0.004659
```

This is extremely close to zero.

This is desirable because VAE latent spaces are designed to remain centered around zero.

---

### mu std

```text
0.908471
```

This is one of the most important metrics.

If:

```text
mu std ≈ 0
```

then every latent vector would look nearly identical.

That situation is called:

```text
Posterior Collapse
```

and means the encoder is not actually using the latent space.

Our result:

```text
mu std = 0.91
```

indicates that the latent space is actively storing information.

The encoder is learning meaningful representations of physiological signals.

---

### logvar mean

```text
-3.39
```

This corresponds to a very small variance.

The encoder is essentially saying:

```text
"I am confident about my latent representation."
```

rather than:

```text
"I'm uncertain."
```

This is a positive sign.

---

# Global Reconstruction Statistics

```text
Original Mean       : -0.060753
Original Std        : 0.933577

Reconstruction Mean : -0.071198
Reconstruction Std  : 0.883363
```

---

## Intuition

The reconstructed data should statistically resemble the original data.

The means are nearly identical:

```text
-0.061
vs
-0.071
```

The standard deviations are also very close:

```text
0.934
vs
0.883
```

This indicates that the model preserves the overall distribution of physiological signals.

In simple terms:

```text
The reconstructed signals "behave" like the original signals.
```

---

# BVP Range Analysis

```text
Original Min : -1.741460
Original Max :  2.513438

Recon Min    : -1.311599
Recon Max    :  2.428737
```

---

## Intuition

Earlier experiments produced nearly flat BVP reconstructions.

For example:

```text
Original:
-1.7 → 2.5

Reconstructed:
-0.09 → 0.06
```

which meant the model was not capturing physiological dynamics.

The current model produces:

```text
-1.31 → 2.43
```

which is very close to the original range.

This indicates that the model has learned to reconstruct signal amplitude successfully.

---

# Channel-Wise Analysis

```text
BVP
Original STD = 0.719
Recon STD    = 0.725
```

This is an excellent result.

The reconstructed BVP signal has almost exactly the same variability as the original signal.

This suggests that the model successfully captures cardiovascular dynamics.

---

```text
ACC_X
Original STD = 0.464
Recon STD    = 0.350
```

Some motion information is preserved, although the amplitude is slightly reduced.

---

```text
ACC_Y
Original STD = 0.124
Recon STD    = 0.082
```

Moderate reconstruction quality.

---

```text
ACC_Z
Original STD = 0.483
Recon STD    = 0.343
```

Reasonable reconstruction, although some variability is lost.

---

```text
EDA
Original STD = 0.028
Recon STD    = 0.053
```

The model reconstructs EDA, although it slightly exaggerates variability.

---

```text
TEMP
Original STD = 0.006
Recon STD    = 0.032
```

Temperature is reconstructed but with somewhat higher variation than the original signal.

---

# Comparison With Earlier Experiments

## Beta = 1.0

```text
Reconstruction almost flat
BVP dynamics lost
Poor physiological reconstruction
```

The model focused too heavily on latent regularization.

---

## Beta = 0.01

```text
Meaningful waveform reconstruction
Improved BVP dynamics
Active latent space
Better overall performance
```

Reducing the KL weight allowed the model to prioritize reconstruction quality.

This proved to be the most important improvement during experimentation.

---

# Final Conclusion

The final VAE model successfully reconstructs unseen physiological windows from the WESAD dataset.

Key findings:

```text
✓ Strong reconstruction quality

✓ Active latent space

✓ No posterior collapse

✓ Accurate BVP reconstruction

✓ Preserves signal distribution

✓ Generalizes well to unseen test data
```

This model serves as the strongest baseline and provides a suitable foundation for implementing the Koopman Operator in the next phase of the project.
