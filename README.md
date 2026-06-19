# EEG Blink-Based Emotion Arousal Detection

Predicting emotional arousal from EEG eye-blink patterns — using ocular
artifacts as a signal, not noise to filter out.

---

## The Idea

Most EEG emotion-detection projects throw raw multi-channel brainwave data
into a deep network and hope it learns something. This project takes a
different, more interpretable route: **eye blinks themselves carry emotional
information**, and they're far easier to extract reliably from EEG than
subtle cortical activity patterns.

Blink rate, regularity, and timing are known in psychophysiology literature
to correlate with arousal — calmer states produce slower, more regular
blinking; heightened arousal produces faster, more irregular blinking. This
project tests that hypothesis end-to-end: extract blink features from raw
EEG, then see how well they predict emotional state.

## Data

14-channel EEG recordings from an Emotiv headset (channels: AF3, F7, F3, FC5,
T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4), labeled with Cowen's 27-category
emotion taxonomy per recording.

## Pipeline

### 1. Signal Extraction
Instead of using all 14 channels directly, two derived signals isolate eye
movement from the surrounding brain activity:
- **VEOG** (vertical eye movement) = AF3 − AF4
- **HEOG** (horizontal eye movement) = F7 − F8

### 2. Blink Detection
A simple, classical signal-processing chain (no deep learning needed here):
1. Bandpass filter the VEOG signal (1–15 Hz) to isolate blink-frequency activity
2. Take the second derivative to sharpen blink peaks
3. Smooth with a moving-average filter
4. Threshold-based peak detection (`scipy.signal.find_peaks`) to locate
   individual blinks

### 3. Feature Engineering
From detected blink peaks, 9 features are computed per recording:
- `blink_count`, `blink_rate_per_min`
- `ibi_mean`, `ibi_std`, `ibi_entropy` (inter-blink interval statistics)
- `blink_duration_mean`, `blink_duration_std`
- `blink_amplitude_mean`, `blink_amplitude_std`
- `blink_regularity`

This pipeline is automated across the full dataset (1000+ raw `.txt` files),
producing one feature row per recording in `blink_features_all.csv`.

### 4. Classification
Two framings were tested:

**27-class emotion classification** (Cowen taxonomy) — Random Forest
→ **~24% accuracy**. Far above the ~3.7% random-chance baseline for 27
classes, but not strong enough to be practically useful. Fine-grained emotion
categories are simply too subtle to separate using blink behavior alone.

**Binary arousal classification** — re-framed the problem using the
arousal dimension of the Circumplex Model of emotion, mapping the 27 emotion
labels into two classes: **High Arousal** vs **Low Arousal**. Five models
were trained and compared:

| Model               | Accuracy |
|---------------------|----------|
| K-Nearest Neighbors | 63.1%    |
| Random Forest       | 64.3%    |
| Logistic Regression | 64.7%    |
| SVM                 | 65.6%    |
| **XGBoost**          | **68.9%** |

XGBoost was then hyperparameter-tuned via `GridSearchCV`
(`n_estimators`, `max_depth`, `learning_rate`), reaching **~70.7% best
cross-validation accuracy**.

### 5. Class Imbalance
Applied **SMOTE** to the training set to address class imbalance. This traded
~4.4 points of overall accuracy (70.7% → 64.3% on the held-out test set) for
significantly better recall on the minority "Low Arousal" class — a
deliberate, reported tradeoff rather than a clean win, since which to
prioritize depends on the downstream use case.

### 6. Feature Importance
Extracted from the trained XGBoost model: `blink_rate_per_min` was the
single most important feature (importance ≈ 0.21), consistent with
psychophysiology literature linking faster, more frequent blinking to higher
arousal states.

### 7. Model Persistence
Final tuned XGBoost model, label encoder, and feature scaler saved with
`joblib` — making the pipeline usable beyond the notebook, not just a
one-off experiment.

---

## Honest Limitations

- **27-class accuracy (~24%) is not production-usable.** Reframing to binary
  arousal was the right call, but fine-grained emotion classification from
  blinks alone remains an open problem — likely needs additional EEG channel
  features beyond just ocular artifacts.
- **SMOTE tradeoff is unresolved.** Whether to prioritize overall accuracy
  or minority-class recall depends on the application; this project reports
  both honestly rather than picking the better-looking number.
- **No held-out external validation.** All evaluation is on a train/test
  split from the same dataset and recording conditions — generalization to
  a different EEG device, population, or session hasn't been tested.
- **No deep learning baseline for comparison.** A CNN/LSTM directly on raw
  EEG channels wasn't tried, so it's unknown whether the interpretable
  feature-engineering approach used here is leaving accuracy on the table.

## Future Work

- Test deep learning baselines (CNN/LSTM on raw EEG) for comparison
- Incorporate additional EEG features beyond blink/ocular signals
- Validate on a held-out dataset from a different recording session/device
- Explore valence (positive/negative) as a second classification axis,
  alongside arousal

---

## Tech Stack

`Python` · `NumPy` · `pandas` · `SciPy` (signal processing) · `scikit-learn`
(Random Forest, Logistic Regression, SVM, KNN, `GridSearchCV`) · `XGBoost` ·
`imbalanced-learn` (SMOTE) · `matplotlib` / `seaborn` · `joblib`

## Repository Structure

```
├── 01_feature_extraction.ipynb   # Raw EEG → VEOG/HEOG → blink detection → feature CSV
├── 02_model_training.ipynb       # Classification experiments, tuning, SMOTE, feature importance
└── README.md
```

## How to Run

1. Run `01_feature_extraction.ipynb` first on raw EEG `.txt` files to produce
   `blink_features_all.csv`
2. Run `02_model_training.ipynb` on that CSV to reproduce the classification
   experiments and final tuned model
