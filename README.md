heelooo

# EchoSense anomaly detection

This repository contains the reusable AI/ML module for detecting anomalous
machine sounds. It trains an Isolation Forest on healthy recordings and
reserves abnormal recordings for evaluation.

## Installation

Use Python 3.10 or later in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Dataset

Add real WAV recordings to these folders:

```text
dataset/
├── normal/       # healthy recordings used for training and evaluation
└── abnormal/     # labeled abnormal recordings used only for evaluation
```

At least two valid healthy recordings are required to train. The module
converts audio to mono, resamples to 22,050 Hz, and skips unreadable or
excessively short WAV files. It does not generate training recordings.

## Train

From the project root:

```bash
python -m modules.train --dataset dataset --model models/isolation_forest.joblib
```

Training uses only `dataset/normal/`; abnormal recordings are never loaded
for model fitting. The default contamination is `auto`; set, for example,
`--contamination 0.05` to configure it. A trained pipeline, including its
feature scaler, is saved with Joblib.

## Predict

```bash
python -m modules.predict path/to/recording.wav \
  --model models/isolation_forest.joblib
```

Reusable functions are available from `modules`:

```python
from modules.anomaly_detection import predict_from_features
from modules.feature_extraction import extract_features
from modules.predict import predict_anomaly

features = extract_features("recording.wav")
result = predict_from_features(features)  # loads the default saved model
# Or: result = predict_anomaly("recording.wav")
```

Predictions contain `status` (`NORMAL` or `ANOMALY`), `is_anomaly`,
`decision_score`, and `anomaly_score` (the negative decision score).
Isolation Forest scores are uncalibrated and are not probabilities.

## Evaluate

```bash
python -m modules.evaluation \
  --dataset dataset \
  --model models/isolation_forest.joblib \
  --output evaluation_results.json
```

Evaluation reports accuracy, precision, recall, F1, confusion matrix, false
positive/negative rates, and ROC-AUC when both labeled classes are present.
It writes JSON metrics and a confusion-matrix PNG alongside the output file.
No performance claims are made without running evaluation on labeled data.

## Tests

```bash
python -m unittest discover -s tests
```
