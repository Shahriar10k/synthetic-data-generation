# 01 Preprocessing Discussion — Common PPG-DaLiA Native-Rate Dataset

## Purpose

This notebook prepares the shared PPG-DaLiA preprocessing output for the joint synthetic data generation project.

The group has different generators, including KoVAE, GAN, and diffusion-based methods. Therefore, the first preprocessing step should be common for everyone.

## Implementation

The notebook loads raw PPG-DaLiA subject files from:

```text
/home/iailab42/g2-synthetic/PPG_FieldStudy
```

and saves project outputs to:

```text
/home/iailab42/khans1/projects/ir
```

It extracts wrist signals:

- BVP / PPG at 64 Hz
- ACC at 32 Hz
- EDA at 4 Hz
- TEMP at 4 Hz
- activity labels at 4 Hz

The signals are segmented into 8.0-second windows with a 2.0-second shift.

## Output shapes

The processed native-rate arrays are:

```text
ACC:      [46925, 256, 3]
BVP:      [46925, 512, 1]
EDA/TEMP: [46925, 32, 2]
Labels:   [46925]
Subjects: [46925]
```

## Why native rates are preserved

The sensors have different original sampling rates. Instead of forcing all signals into one frequency, the preprocessing keeps each modality at its native rate.

This is important for the later multi-branch KoVAE:

```text
BVP encoder  ┐
ACC encoder  ├── shared latent sequence ── Koopman dynamics ── branch decoders
SLOW encoder ┘
```

The same preprocessed arrays can also be used by the GAN and diffusion methods.

## Normalization

This shared preprocessing follows the common group format and uses global normalization over the processed dataset.

This matches the existing group evaluation notebooks. For strict subject-independent experiments, a later notebook can additionally compute train-subject-only normalization from the raw saved arrays.

## Shared split protocol

This notebook does not create split arrays. It only saves the common subject split protocol:

```text
Train: ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S9', 'S11', 'S12', 'S13']
Val:   ['S14', 'S15']
Test:  ['S7', 'S8', 'S10']
```

The split should be applied later during KoVAE training, synthetic subject generation, and downstream evaluation.

## Analysis

The preprocessing is suitable as a shared group input because:

1. all methods receive identical real data,
2. native sensor resolutions are preserved,
3. ambiguous windows are filtered using dominant label coverage,
4. subject IDs are saved for subject-level generation and evaluation,
5. metadata is saved for later analysis and visualization.

Possible limitations:

1. overlapping windows are not independent,
2. global normalization can introduce mild subject-level information leakage,
3. transition windows are removed by the label coverage filter,
4. native-rate arrays require multi-branch model design.

## What to report

In the final report, describe this step in the Dataset and Preprocessing section.

Suggested report text:

> We used the PPG-DaLiA wrist sensor signals and preserved their native sampling rates: BVP at 64 Hz, ACC at 32 Hz, and EDA/TEMP at 4 Hz. Signals were segmented into 8-second windows with a 2-second shift. For each window, the dominant activity label was selected, and windows with insufficient label coverage were removed. The resulting arrays were saved as native-rate branches, allowing all generative models in the group to use the same processed data.

Include this table:

| Signal | Sampling rate | Window length | Channels | Shape |
|---|---:|---:|---:|---|
| BVP / PPG | 64 Hz | 512 | 1 | `[N, 512, 1]` |
| ACC | 32 Hz | 256 | 3 | `[N, 256, 3]` |
| EDA/TEMP | 4 Hz | 32 | 2 | `[N, 32, 2]` |

## What to present in slides

Use one preprocessing slide:

**Title:** Common PPG-DaLiA preprocessing

Bullets:

- Same preprocessing for KoVAE, GAN, and diffusion models
- BVP kept at 64 Hz, ACC at 32 Hz, EDA/TEMP at 4 Hz
- 8-second windows with 2-second shift
- Dominant activity label per window
- Subject IDs saved for synthetic subject generation
- Native-rate output supports multi-branch KoVAE

## Saved outputs

Clean project folder:

```text
data/processed/native_rates/
```

Additional outputs:

```text
configs/preprocessing_config.json
configs/shared_subject_split.json
results/preprocessing/preprocessing_summary.json
figures/preprocessing/activity_distribution.png
figures/preprocessing/subject_window_counts.png
figures/preprocessing/example_native_rate_window.png
logs/preprocessing.log
discussion/01_preprocessing_discussion.md
```

## Next step

The next KoVAE notebook should load the common processed arrays and apply the shared subject split during training.
