# RespiScope-AI Dataset Description

## Overview

RespiScope-AI uses multiple publicly available respiratory sound databases for training and evaluation. This document provides detailed information about each dataset, preprocessing steps, and class mapping strategies.

---

## Supported Datasets

### 1. ICBHI 2017 Respiratory Sound Database

**Source:** [ICBHI Challenge](https://bhichallenge.med.auth.gr/)  
**Kaggle:** [Respiratory Sound Database](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database)

#### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Recordings | 920 |
| Total Patients | 126 |
| Recording Devices | 4 (AKGC417L, Meditron, Litt3200, WelchAllyn) |
| Duration Range | 10-90 seconds |
| Sampling Rate | 4000 - 44100 Hz (varied) |
| Total Duration | ~5.5 hours |
| Annotation Type | Crackles and Wheezes |

#### Patient Demographics

- **Age Range:** 6-90 years
- **Gender:** Male and Female
- **Locations:** 7 countries (Greece, Portugal, Spain, etc.)
- **Recording Sites:** 4 chest locations per patient

#### Diagnoses

| Diagnosis | Count | Percentage |
|-----------|-------|------------|
| Healthy | 32 | 25.4% |
| COPD | 32 | 25.4% |
| Bronchiectasis | 12 | 9.5% |
| Pneumonia | 10 | 7.9% |
| Bronchiolitis | 8 | 6.3% |
| Asthma | 8 | 6.3% |
| LRTI | 8 | 6.3% |
| URTI | 8 | 6.3% |
| Others | 8 | 6.3% |

#### Class Mapping for RespiScope-AI

```python
ICBHI_MAPPING = {
    'Asthma': 'Asthma',
    'COPD': 'COPD',
    'Pneumonia': 'Pneumonia',
    'LRTI': 'Pneumonia',  # Lower Respiratory Tract Infection
    'Bronchiectasis': 'Bronchitis',
    'Bronchiolitis': 'Bronchitis',
    'URTI': 'Healthy',  # Upper Respiratory Tract Infection (mild)
    'Healthy': 'Healthy'
}
```

#### Audio Characteristics

- **Sample Rates:** 4kHz, 10kHz, 44.1kHz (resampled to 16kHz)
- **Bit Depth:** 16-bit
- **Channels:** Mono
- **Format:** WAV (uncompressed)

#### Annotations

Each recording has corresponding `.txt` file with:
- Start time (seconds)
- End time (seconds)
- Crackles presence (0 or 1)
- Wheezes presence (0 or 1)

Example annotation:
```
0.000	2.500	0	1
2.500	5.000	1	0
5.000	7.500	0	0
```

---

### 2. Coswara Dataset

**Source:** [Coswara Project](https://coswara.iisc.ac.in/)  
**Paper:** [Coswara - A Dataset](https://arxiv.org/abs/2005.10548)

#### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Participants | 1500+ |
| Audio Types | 9 types (cough, breathing, voice) |
| Languages | 20+ |
| Countries | 60+ |
| Recording Method | Crowdsourced (mobile apps) |
| Data Collection Period | 2020-2023 |

#### Audio Types

1. **Cough Sounds:**
   - Shallow cough
   - Deep cough
   - Heavy cough

2. **Breathing Sounds:**
   - Normal breathing
   - Deep breathing
   - Fast breathing

3. **Voice Sounds:**
   - Vowel sounds (a, e, i, o, u)
   - Counting (1-20)
   - Reading sentences

#### Metadata

- COVID-19 status (positive/negative/recovered)
- Age, gender, location
- Symptoms present
- Medical history
- Smoking status
- Recording quality rating

#### Class Mapping

Since Coswara is primarily COVID-19 focused, mapping to respiratory conditions requires careful consideration:

```python
COSWARA_MAPPING = {
    'covid_positive_symptomatic': 'Pneumonia',  # COVID pneumonia
    'covid_positive_asymptomatic': 'Healthy',
    'covid_negative_healthy': 'Healthy',
    'covid_recovered': 'Healthy',
    'covid_negative_symptomatic': 'Bronchitis'  # Other respiratory symptoms
}
```

---

### 3. Additional Datasets (Future Integration)

#### 3.1 ESC-50 (Environmental Sound Classification)

- **Purpose:** Negative samples / noise augmentation
- **Sounds:** General environmental sounds
- **Use:** Background noise addition

#### 3.2 AudioSet (Google)

- **Purpose:** Transfer learning (PANN pretraining)
- **Size:** 2M+ audio clips
- **Categories:** 527 sound classes

---

## Preprocessing Pipeline

### Stage 1: Audio Loading

```python
# Load audio with librosa
audio, sr = librosa.load(
    audio_path,
    sr=16000,  # Target sample rate
    mono=True,  # Convert to mono
    duration=10.0  # Maximum duration
)
```

### Stage 2: Trimming & Normalization

```python
# Remove silence
audio_trimmed, _ = librosa.effects.trim(
    audio,
    top_db=30  # Threshold for silence
)

# Normalize to [-1, 1]
if np.abs(audio_trimmed).max() > 0:
    audio_normalized = audio_trimmed / np.abs(audio_trimmed).max()
```

### Stage 3: Length Standardization

```python
target_length = 16000 * 10  # 10 seconds at 16kHz

if len(audio) > target_length:
    audio = audio[:target_length]  # Truncate
else:
    # Pad with zeros
    audio = np.pad(audio, (0, target_length - len(audio)))
```

### Stage 4: Feature Extraction (PANN)

```python
# PANN extracts 2048-dimensional embeddings
embedding = pann_model.extract_embedding(audio)
# Shape: (2048,)
```

---

## Data Augmentation

### Techniques Applied

1. **Pitch Shifting**
   - Range: ±2 semitones
   - Probability: 30%
   - Simulates voice/cough variations

2. **Time Stretching**
   - Range: 0.9-1.1x speed
   - Probability: 30%
   - Simulates breathing rate variations

3. **Noise Injection**
   - SNR Range: 10-30 dB
   - Probability: 30%
   - Simulates recording conditions

4. **Time Masking**
   - Duration: 0.1-0.5 seconds
   - Probability: 20%
   - Improves robustness

5. **Frequency Masking**
   - Width: 5-15 mel bands
   - Probability: 20%
   - Improves robustness

### Implementation

```python
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
])

augmented_audio = augment(samples=audio, sample_rate=16000)
```

---

## Train/Val/Test Split Strategy

### Splitting Approach

**Patient-Level Split** (Prevents data leakage)

```python
# Split by patient ID, not by recording
train_patients: 70%
val_patients: 15%
test_patients: 15%
```

### Stratification

- Maintain class distribution across splits
- Balance gender and age groups when possible
- Ensure all diagnoses represented

### Split Statistics

| Split | Patients | Recordings | Duration (hours) |
|-------|----------|------------|------------------|
| Train | 88 (70%) | ~640 | ~3.8 |
| Val | 19 (15%) | ~140 | ~0.8 |
| Test | 19 (15%) | ~140 | ~0.8 |

---

## Class Distribution

### Target Classes (5-class)

| Class | Train | Val | Test | Total | Percentage |
|-------|-------|-----|------|-------|------------|
| Asthma | 120 | 25 | 25 | 170 | 18.5% |
| Bronchitis | 140 | 30 | 30 | 200 | 21.7% |
| COPD | 180 | 40 | 40 | 260 | 28.3% |
| Pneumonia | 100 | 20 | 20 | 140 | 15.2% |
| Healthy | 100 | 25 | 25 | 150 | 16.3% |
| **Total** | **640** | **140** | **140** | **920** | **100%** |

### Class Imbalance Handling

1. **Class Weights:**
   ```python
   class_weights = compute_class_weight(
       'balanced',
       classes=np.unique(labels),
       y=labels
   )
   ```

2. **Focal Loss:**
   - Focuses on hard examples
   - Reduces impact of easy negatives
   - α = 0.25, γ = 2.0

3. **Data Augmentation:**
   - More aggressive for minority classes
   - Synthetic sample generation

---

## Quality Control

### Inclusion Criteria

✅ **Include if:**
- Clear audio signal
- Minimal background noise (< -30dB)
- Proper duration (> 5 seconds)
- Valid annotations
- Correct diagnosis label

❌ **Exclude if:**
- Corrupted file
- Excessive noise
- Missing metadata
- Uncertain diagnosis
- Recording artifact

### Quality Metrics

```python
def assess_audio_quality(audio, sr):
    # Signal-to-Noise Ratio
    snr = calculate_snr(audio)
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio).mean()
    
    # RMS Energy
    rms = librosa.feature.rms(y=audio).mean()
    
    # Quality score
    quality = 'good' if snr > 20 and rms > 0.01 else 'poor'
    
    return quality, snr, zcr, rms
```

---

## Data Format

### Directory Structure

```
data/
├── raw/
│   ├── icbhi/
│   │   ├── audio_and_txt_files/
│   │   ├── patient_diagnosis.csv
│   │   └── README.txt
│   └── coswara/
│       ├── recordings/
│       └── metadata/
├── processed/
│   ├── icbhi/
│   │   ├── *.npy  # Preprocessed audio
│   │   └── metadata.csv
│   └── coswara/
│       ├── *.npy
│       └── metadata.csv
└── splits/
    ├── train/
    │   ├── audio/
    │   └── labels.csv
    ├── val/
    │   ├── audio/
    │   └── labels.csv
    └── test/
        ├── audio/
        └── labels.csv
```

### Metadata Format (CSV)

```csv
filename,patient_id,diagnosis,class,chest_location,equipment,duration,split
101_1b1_Al_sc_Meditron_processed.npy,101,COPD,COPD,Al,Meditron,20.5,train
101_1b2_Pr_sc_Meditron_processed.npy,101,COPD,COPD,Pr,Meditron,18.2,train
...
```

---

## Citation

### ICBHI Dataset

```bibtex
@article{rocha2019alpha,
  title={An open access database for the evaluation of respiratory sound classification algorithms},
  author={Rocha, Bruno M and Filos, Dimitris and Mendes, Luis and others},
  journal={Physiological Measurement},
  volume={40},
  number={3},
  pages={035001},
  year={2019},
  publisher={IOP Publishing}
}
```

### Coswara
