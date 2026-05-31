# PPG-DaLiA Data Exploration Notes

## Project Goal

```text
Synthetic Data
      ↓
Train HAR Model
      ↓
Better Activity Recognition
```

The ultimate objective is not only to generate realistic synthetic physiological data but also to investigate whether synthetic data can improve Human Activity Recognition (HAR) performance.

---

# Dataset Overview

PPG-DaLiA is a multimodal physiological dataset containing wearable sensor data collected from subjects performing daily-life activities such as sitting, walking, cycling, driving, lunch, and working.

The dataset was created to support research in:

* PPG-based heart rate estimation
* Human activity analysis
* Physiological sensing under realistic conditions

---

# Why is PPG-DaLiA Useful for Synthetic Data Generation?

The dataset is particularly suitable because:

* It contains multiple sensor modalities.
* It includes recordings from multiple subjects.
* Activities are performed under realistic conditions.
* Sensor data contains natural movement artifacts and noise.

This diversity allows a generative model to learn realistic physiological behavior across subjects and activities.

---

# Sensor Modalities

## Wrist Signals

| Signal | Shape       | Channels | Sampling Rate |
| ------ | ----------- | -------- | ------------- |
| BVP    | (589568, 1) | 1        | 64 Hz         |
| ACC    | (294784, 3) | 3        | 32 Hz         |
| EDA    | (36848, 1)  | 1        | 4 Hz          |
| TEMP   | (36848, 1)  | 1        | 4 Hz          |

### Total Wrist Channels

```text
BVP      = 1
ACC_X    = 1
ACC_Y    = 1
ACC_Z    = 1
EDA      = 1
TEMP     = 1
----------------
Total    = 6 channels
```

---

# Activity Distribution Analysis

The dataset is imbalanced.

Activities such as:

* Transition
* Lunch

contain significantly more samples than activities such as:

* Stairs
* Table Soccer

This imbalance means that a generative model may learn dominant activities better while struggling to model underrepresented activities accurately.

However, this is not necessarily a data collection problem.

The imbalance reflects real-world behavior because activities such as lunch and working naturally last longer than activities such as climbing stairs.

---

# Should Synthetic Data Preserve Reality or Balance Activities?

## Option 1: Preserve Original Distribution

### Advantages

* Reflects realistic population behavior.
* Maintains real-world activity frequencies.

### Disadvantages

* Minority activities remain underrepresented.
* HAR models may still struggle on rare activities.

---

## Option 2: Balance Activities

### Advantages

* Improves activity classifier training.
* Provides more examples for underrepresented activities.

### Disadvantages

* Less representative of real-world activity distributions.

---

## Research Conclusion

The choice depends on the objective.

If the goal is realistic simulation, the synthetic data should preserve the original activity distribution.

If the goal is improving activity recognition, synthetic data can be used to augment underrepresented activities and create a more balanced training set.

---

# Evaluating Synthetic Data

## Realism Evaluation

Compare synthetic data with real data using:

* Signal plots
* Statistical similarity
* Histograms
* Feature distributions
* Real-vs-Synthetic classifiers

Examples:

* Real BVP vs Synthetic BVP
* Real ACC vs Synthetic ACC

Metrics:

* Mean
* Standard deviation
* Minimum and maximum values
* Signal variance
* Movement intensity
* Heart-rate characteristics

---

## Utility Evaluation

Evaluate whether synthetic data improves HAR performance.

Example:

```text
Train HAR on Real Data
vs
Train HAR on Real + Synthetic Data
```

If activity recognition improves, the synthetic data provides practical value.

---

# BVP Signal Analysis

## Observations

The BVP signal:

* Is not random.
* Exhibits periodic behavior.
* Contains repeating waveform patterns.
* Includes sudden spikes and irregular fluctuations.

These periodic patterns correspond to cardiovascular activity.

---

## Motion Artifacts

Large spikes are likely caused by:

* Hand movement
* Walking
* Climbing stairs
* Adjusting the wristband

Motion artifacts are one of the major challenges in wearable physiological sensing.

---

## ECG vs BVP

### ECG

* Measures electrical activity of the heart.
* Generally cleaner.

### BVP

* Measures blood volume changes.
* More susceptible to movement artifacts.
* Typically noisier than ECG.

---

## What Should Synthetic Data Learn?

A realistic generator should learn:

```text
Heartbeat Pattern
+
Motion Artifacts
+
Activity-Specific Behavior
```

The goal is not to generate perfectly clean physiological signals.

Instead, the model should reproduce realistic physiological variability and noise.

---

# Why Synthetic Physiological Data is Difficult

Unlike tabular data, physiological signals exhibit:

## Temporal Dependency

Current measurements depend on previous measurements.

Examples:

```text
Sample 100 depends on Sample 99
Sample 101 depends on Sample 100
```

The order of observations matters.

---

## Additional Challenges

The model must preserve:

* Temporal dependencies
* Activity-specific patterns
* Sensor noise
* Relationships between multiple physiological channels

---

# Accelerometer Analysis

## What do ACC X, Y, and Z Represent?

The accelerometer measures movement along three spatial dimensions.

Together, the three axes describe:

* Direction of movement
* Magnitude of movement

---

## Do ACC Channels Behave Identically?

No.

Each axis captures different movement components.

Meaningful movement is represented by the combination of all three channels.

---

## Increased Activity Around Samples 400–700

Possible explanation:

* Activity transition
* Walking
* Cycling
* Other movement-intensive activity

The activity labels should be checked to confirm the cause.

---

## Why Must ACC Channels Remain Consistent?

ACC_X, ACC_Y, and ACC_Z jointly describe human movement.

If synthetic channels become inconsistent, the generated movement may not correspond to any real activity.

Therefore, synthetic generation must preserve cross-channel relationships.

---

# Relationship Between ACC and BVP

ACC and BVP are not independent.

Example:

```text
High Movement
      ↓
ACC Changes
      ↓
Motion Artifacts in BVP
```

Physical movement affects signal quality.

Therefore, realistic synthetic data should preserve these relationships.

---

# Correlation Analysis

Example result:

```text
ACC_X ↔ ACC_Y = 0.52
ACC_X ↔ ACC_Z = -0.04
```

Interpretation:

* ACC_X and ACC_Y exhibit moderate positive correlation.
* ACC_X and ACC_Z show almost no linear relationship.

---

## Why Correlation Matters

Even if individual channels appear realistic, synthetic data is not realistic if relationships between channels are lost.

A realistic generator should preserve:

```text
Real ACC_X ↔ Real ACC_Y
```

within

```text
Synthetic ACC_X ↔ Synthetic ACC_Y
```

---

# Windowing Strategy

## Why Windowing?

Windowing:

* Reduces computational complexity.
* Increases the number of training samples.
* Allows learning localized activity patterns.

---

## Example Window

Window Length:

```text
8 seconds
```

BVP Sampling Rate:

```text
64 Hz
```

Therefore:

```text
64 × 8 = 512 samples
```

One BVP training sample:

```python
(512,)
```

---

# Multichannel Window Representation

For wrist signals:

```text
BVP
ACC_X
ACC_Y
ACC_Z
EDA
TEMP
```

One training window becomes:

```python
(512, 6)
```

Meaning:

```text
512 time steps
6 channels
```

---

# Preprocessing Challenge

Different channels have different sampling rates:

```text
BVP  = 64 Hz
ACC  = 32 Hz
EDA  = 4 Hz
TEMP = 4 Hz
```

These signals cannot be directly combined.

---

# Why Resampling is Necessary

To build a multichannel KoVAE input, all channels must share the same temporal resolution.

Example:

```text
Convert all channels → 64 Hz
```

After resampling:

```text
BVP  = 512 samples
ACC  = 512 samples
EDA  = 512 samples
TEMP = 512 samples
```

Now they can be combined into:

```python
(512, 6)
```

for training.
