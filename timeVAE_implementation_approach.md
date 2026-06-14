# TimeVAE Baseline Approach for Synthetic Physiological Time-Series Generation

## 1. Objective

The objective of this part of the project is to implement **TimeVAE** as a baseline model for synthetic multichannel physiological time-series generation.

The generated synthetic data should follow the same format as the already-preprocessed real PPG-DaLiA data so that the existing comparison and evaluation code can directly use it.

The main goal is:

> Generate synthetic 6-channel physiological windows conditioned on activity labels.

The synthetic data will later be compared with real data using realism checks, diversity checks, visual comparison, and downstream human activity recognition performance.

---

## 2. Why TimeVAE?

TimeVAE is used as a baseline because it is a VAE-based generative model designed for multivariate time-series data.

A Variational Autoencoder is usually more stable and simpler to train than GAN-based models. Instead of using a generator-discriminator competition, TimeVAE learns an encoder-decoder structure.

The encoder compresses a real time-series window into a latent representation. The decoder reconstructs the signal from this latent representation. After training, we can sample a random latent vector and use the decoder to generate new synthetic time-series windows.

The TimeVAE paper proposes two versions:

1. Base TimeVAE
2. Interpretable TimeVAE

For this project, we use the **Base TimeVAE idea** because it is cleaner and more suitable as a baseline. The interpretable version with trend and seasonality blocks is more complex and is not necessary for the first baseline implementation.

---

## 3. Real Data Format

The real data was already preprocessed using the same preprocessing pipeline used for the KoVAE project.

The important real data files are stored in:

```text
/home/iailab42/khans1/projects/data/
```

The main files are:

```text
all_X.npy
all_y.npy
all_subject.npy
all_metadata.csv
normalization_stats.npz
```

The main input file is:

```text
all_X.npy
```

Its shape is:

```text
[N, 512, 6]
```

Meaning:

| Dimension | Meaning |
|---|---|
| `N` | Number of windows |
| `512` | Number of time steps per window |
| `6` | Number of physiological channels |

Each window represents:

```text
512 samples / 64 Hz = 8 seconds
```

The six channels are:

```text
ACC_x, ACC_y, ACC_z, BVP, EDA, TEMP
```

The activity labels are stored in:

```text
all_y.npy
```

The subject IDs are stored in:

```text
all_subject.npy
```

The preprocessing settings are:

```text
Target frequency: 64 Hz
Window length: 512 samples
Window duration: 8 seconds
Shift length: 128 samples
Overlap: 75%
Minimum dominant label coverage: 80%
Dropped activity label: 0
```

The data is globally normalized using the saved mean and standard deviation values from:

```text
normalization_stats.npz
```

---

## 4. What TimeVAE Learns

TimeVAE learns to generate a synthetic multichannel time-series window.

In this project, the model input and output shape is:

```text
Input window:  [512, 6]
Output window: [512, 6]
```

The model learns patterns across all six channels:

```text
ACC_x
ACC_y
ACC_z
BVP
EDA
TEMP
```

The goal is for each generated synthetic window to look like a realistic physiological window for a given activity.

---

## 5. Why Activity Conditioning Is Used

A plain TimeVAE can generate synthetic time-series windows, but it does not naturally know which activity label belongs to each generated window.

For this project, activity labels are necessary because the synthetic data will be used for human activity detection.

Therefore, the implemented model is:

```text
Activity-conditioned TimeVAE
```

This means the model receives the activity label as additional information.

The generation process is:

```text
Random latent vector z + activity label
        ↓
TimeVAE decoder
        ↓
Synthetic 6-channel physiological window
```

This allows us to generate synthetic windows for each activity class.

---

## 6. Why One TimeVAE Model Is Used

We use **one single TimeVAE model**, not one separate VAE per activity.

The model is trained on all activity windows together.

This is better because:

1. It keeps the baseline clean.
2. It shares learning across all activities.
3. It avoids training many separate models.
4. It is easier to compare with KoVAE.
5. It better follows the general TimeVAE idea of learning one multivariate time-series distribution.

The model is saved as:

```text
/home/iailab42/khans1/projects/models/timevae/timevae_conditional_6ch.pt
```

If this model already exists, the notebook loads it instead of retraining.

---

## 7. Subject Handling

The real dataset has real subject IDs such as:

```text
S1, S2, S3, ..., S15
```

However, TimeVAE is not explicitly trained to create true new subject identities.

In this implementation, TimeVAE is activity-conditioned, not subject-conditioned.

Therefore, after generating synthetic windows, artificial synthetic subject IDs are assigned:

```text
SYN_001, SYN_002, SYN_003, ...
```

These synthetic subject IDs are used for compatibility with the existing comparison and evaluation code.

Important interpretation:

> The synthetic subject IDs are grouping IDs for evaluation compatibility. They should not be interpreted as fully realistic new human identities learned by TimeVAE.

KoVAE is expected to be stronger for subject-aware generation, while TimeVAE is used as a baseline.

---

## 8. Model Architecture

The implemented model is a conditional VAE for multivariate time-series data.

The architecture has four main parts:

1. Encoder
2. Latent Gaussian space
3. Reparameterization
4. Decoder

---

## 9. Encoder

The encoder receives:

```text
Real time-series window + activity label
```

The real window has shape:

```text
[batch_size, 512, 6]
```

The encoder first applies 1D convolution layers to the time-series signal.

The activity label is converted into a label embedding vector.

Then the encoded time-series representation and label embedding are concatenated.

The encoder outputs:

```text
mu
logvar
```

These two outputs define the latent Gaussian distribution.

---

## 10. Latent Space

The latent space is where the model stores a compressed representation of the input time-series window.

Instead of encoding each input into one fixed vector, the VAE encodes the input into a probability distribution.

The latent distribution is represented by:

```text
mu
logvar
```

where:

```text
mu     = mean of the latent distribution
logvar = log variance of the latent distribution
```

This allows the model to generate new samples by sampling from the latent space.

---

## 11. Reparameterization Trick

The latent vector is sampled using the VAE reparameterization trick:

```text
z = mu + epsilon * std
```

where:

```text
std = exp(0.5 * logvar)
epsilon ~ Normal(0, 1)
```

This trick allows the sampling operation to remain trainable with backpropagation.

---

## 12. Decoder

The decoder receives:

```text
Latent vector z + activity label embedding
```

Then it reconstructs or generates a time-series window with shape:

```text
[batch_size, 512, 6]
```

During generation, the decoder receives:

```text
Random z + selected activity label
```

and produces:

```text
Synthetic 6-channel physiological window
```

---

## 13. Loss Function

The basic VAE loss has two parts:

```text
Total loss = Reconstruction loss + KL divergence loss
```

In the implementation:

```text
total_loss = reconstruction_weight * reconstruction_loss + kl_weight * kl_loss
```

The reconstruction loss encourages the reconstructed signal to be close to the real input signal.

The KL divergence loss encourages the latent space to follow a standard normal distribution.

---

## 14. Why KL Weight Is Small

In physiological time-series generation, too much KL pressure can make the model produce overly smooth or average-looking signals.

For example, accelerometer signals may lose sharp movement peaks.

Therefore, a small KL weight is used.

This makes the model focus more on reconstruction quality and less on forcing the latent space too strongly toward a standard normal distribution.

---

## 15. Important Practical Issue: VAE Smoothing

One limitation observed during visual inspection is that TimeVAE-generated signals can be too smooth.

This is especially visible in accelerometer channels:

```text
Real ACC signals may contain sharp peaks and movement changes.
Synthetic ACC signals may look smoother and less intense.
```

This happens because VAEs trained with reconstruction loss often learn average patterns.

This limitation is important and should be mentioned in the final report.

Possible improvements include:

1. Increasing reconstruction weight
2. Reducing KL weight
3. Increasing latent dimension
4. Adding temporal difference loss
5. Training longer
6. Using a stronger decoder
7. Comparing same-activity real and synthetic windows during visual checks

---

## 16. Training Strategy

The notebook uses a train-validation split only for training monitoring and early stopping.

The model is trained on the real preprocessed windows.

Training behavior:

```text
If model file exists:
    Load model
    Do not retrain

If model file does not exist:
    Train model
    Save model checkpoint
```

The checkpoint includes:

```text
model_state_dict
optimizer_state_dict
config
label mapping
training history
train indices
validation indices
```

The saved model path is:

```text
models/timevae/timevae_conditional_6ch.pt
```

---

## 17. Synthetic Data Generation Strategy

Synthetic data generation is always run fresh when the generation cell is executed.

Unlike the model, synthetic data is not loaded from an old file.

Generation behavior:

```text
Always generate synthetic data
Always overwrite previous synthetic files
```

For each activity label:

1. Count how many real windows exist for that activity.
2. Generate the same number of synthetic windows.
3. Store the original activity label.
4. Store the internal model label.
5. Assign synthetic subject IDs.
6. Save everything in the expected format.

The generation ratio is controlled by:

```text
synthetic_multiplier
```

For example:

```text
synthetic_multiplier = 1.0
```

means the model generates approximately the same number of synthetic windows as the real dataset.

---

## 18. Output Folder Structure

All TimeVAE outputs are saved separately so they do not mix with KoVAE outputs.

The folder structure is:

```text
models/timevae/
results/timevae/
figures/timevae/
configs/timevae/
logs/timevae/
data/synthetic/timevae/
```

The main synthetic data folder is:

```text
/home/iailab42/khans1/projects/data/synthetic/timevae/
```

---

## 19. Friend-Compatible Output Format

The existing comparison code expects synthetic files in this format:

```text
all_X_synthetic.npy
all_y_synthetic.npy
all_subject_synthetic.npy
all_metadata_synthetic.csv
```

Therefore, TimeVAE saves synthetic data using the same format.

The main saved files are:

```text
all_X_synthetic.npy
all_y_synthetic.npy
all_y_internal_synthetic.npy
all_subject_synthetic.npy
all_metadata_synthetic.csv
```

---

## 20. File Meanings

| File | Meaning |
|---|---|
| `all_X_synthetic.npy` | Synthetic normalized 6-channel windows |
| `all_y_synthetic.npy` | Original activity labels |
| `all_y_internal_synthetic.npy` | Internal zero-based label IDs used by the model |
| `all_subject_synthetic.npy` | Synthetic subject IDs |
| `all_metadata_synthetic.csv` | Metadata for synthetic windows |

Extra files are also saved:

```text
all_X_acc_bvp_synthetic.npy
all_coverage_synthetic.npy
all_start_indices_64hz_synthetic.npy
all_end_indices_64hz_synthetic.npy
subject_summary_synthetic.csv
normalization_stats.npz
all_X_raw_synthetic.npy
all_X_raw_acc_bvp_synthetic.npy
timevae_synthetic_manifest.json
```

These extra files are useful for future analysis and reproducibility.

---

## 21. Why Raw Synthetic Data Is Saved

The model generates normalized synthetic data because it is trained on normalized real data.

The real preprocessing saved:

```text
mean
std
```

inside:

```text
normalization_stats.npz
```

So raw-scale synthetic data is reconstructed using:

```text
X_raw = X_norm * std + mean
```

This produces:

```text
all_X_raw_synthetic.npy
```

This is useful for inspecting synthetic physiological values in the original scale.

---

## 22. Metadata Creation

The synthetic metadata file is saved as:

```text
all_metadata_synthetic.csv
```

It contains fields such as:

```text
global_window_id
synthetic_subject
subject
synthetic_window_in_subject
activity_label
activity_label_internal
dominant_label_coverage
start_sample_64hz
end_sample_64hz
start_time_sec
end_time_sec
is_synthetic
source_model
```

For synthetic data, the subject timeline is artificial.

The start and end sample indices are created using:

```text
start_sample = synthetic_window_in_subject * shift_len
end_sample = start_sample + window_len
```

This makes the synthetic windows compatible with the existing overlap-add visualization code.

---

## 23. Diagnostic Plots

The notebook saves diagnostic plots in:

```text
/home/iailab42/khans1/projects/figures/timevae/
```

The plots include:

```text
timevae_training_loss.png
quick_real_vs_synthetic_ACC_x.png
quick_real_vs_synthetic_ACC_y.png
quick_real_vs_synthetic_ACC_z.png
quick_real_vs_synthetic_BVP.png
quick_real_vs_synthetic_EDA.png
quick_real_vs_synthetic_TEMP.png
```

The plots are also displayed inside the notebook using:

```text
plt.show()
```

---

## 24. Same-Label Visual Comparison

It is important to compare real and synthetic windows from the same activity label.

A bad comparison would be:

```text
Real label 4
Synthetic label 6
```

because the two signals may naturally look different due to different activities.

A better comparison is:

```text
Real label k
Synthetic label k
```

This makes the visual check more meaningful.

---

## 25. Evaluation Plan

TimeVAE synthetic data will be evaluated using the same evaluation pipeline as KoVAE.

The expected comparison input directories are:

```text
Real:
data/

Synthetic TimeVAE:
data/synthetic/timevae/
```

The evaluation should include:

1. Visual comparison
2. Distribution comparison
3. Diversity check
4. Downstream human activity detection

---

## 26. Visual Comparison

Visual comparison should inspect all six channels:

```text
ACC_x
ACC_y
ACC_z
BVP
EDA
TEMP
```

The goal is to check whether synthetic windows look structurally similar to real windows.

Same-label comparison should be preferred.

---

## 27. Distribution Comparison

Distribution comparison should compare real and synthetic statistics per channel and per activity.

Useful statistics include:

```text
mean
standard deviation
minimum
maximum
median
percentiles
```

This helps determine whether synthetic data follows the same value distribution as real data.

---

## 28. Diversity Check

Diversity check is needed to ensure the model does not generate repeated or collapsed samples.

Useful diversity checks include:

```text
variance per channel
pairwise distance between generated windows
PCA visualization
t-SNE visualization
activity-wise distribution
```

Low diversity would mean the model is generating very similar samples repeatedly.

---

## 29. Downstream Activity Detection

The final utility test is human activity detection.

Possible experiments:

1. Train classifier on real data only.
2. Train classifier on synthetic data only.
3. Train classifier on real + synthetic data.
4. Train classifier on reduced real data + synthetic data.

The classifier should be evaluated on real held-out test data.

Important metrics:

```text
accuracy
precision
recall
F1-score
confusion matrix
```

The main question is:

> Does TimeVAE synthetic data improve or hurt human activity detection?

---

## 30. Relationship to KoVAE

KoVAE is the main method.

TimeVAE is used as the baseline.

The expected comparison is:

```text
KoVAE vs TimeVAE
```

TimeVAE is simpler and easier to train, but it may generate smoother signals.

KoVAE is expected to better handle subject-aware and activity-aware generation.

In the final report, TimeVAE should be described as:

> A VAE-based baseline for activity-conditioned multivariate time-series generation.

KoVAE should be described as:

> The proposed or stronger method for subject-aware and activity-aware synthetic physiological data generation.

---

## 31. Limitations of TimeVAE

The main limitations are:

1. It is not truly subject-aware.
2. Synthetic subject IDs are assigned after generation.
3. Generated accelerometer signals may be too smooth.
4. VAE reconstruction loss can average out sharp movements.
5. It may not capture rare or high-intensity motion patterns well.
6. It may generate realistic-looking windows but not realistic long continuous subject timelines.
7. Activity-conditioned generation depends strongly on label quality.

These limitations should be mentioned clearly in the report.

---

## 32. Important Implementation Choices

The implementation takes the following considerations:

### Separate TimeVAE folders

TimeVAE files are stored separately from KoVAE files.

```text
models/timevae/
results/timevae/
figures/timevae/
configs/timevae/
logs/timevae/
data/synthetic/timevae/
```

### Reproducibility

A fixed random seed is used:

```text
seed = 42
```

### Model reuse

If the trained model already exists, it is loaded.

This prevents unnecessary retraining.

### Synthetic overwrite

Synthetic data is always regenerated and overwritten.

This makes experimentation easier when testing different generation settings.

### names

The output file names match the existing comparison code:

```text
all_X_synthetic.npy
all_y_synthetic.npy
all_subject_synthetic.npy
all_metadata_synthetic.csv
```

---

## 33. Summary

TimeVAE is implemented as a single activity-conditioned VAE baseline.

It uses the preprocessed PPG-DaLiA 6-channel windows:

```text
[N, 512, 6]
```

The model learns to generate synthetic physiological windows conditioned on activity labels.

The generated data is saved in the same format expected by the existing KoVAE comparison and evaluation pipeline.

TimeVAE is useful as a baseline because it is simpler than GAN-based methods and easier to train, but it may produce smoother signals than real physiological data.

This limitation is important when interpreting the final realism and downstream activity recognition results.
