# Federated Learning for Human Activity Recognition

## Overview

This research project investigates **Federated Learning (FL) for Human
Activity Recognition (HAR)** using wearable and mobile inertial-sensor
data. The work brings together literature review, dataset integration,
feature engineering, federated model experimentation, hyperparameter
optimization, and detailed evaluation of predictive performance,
calibration, and learned representations.

The dataset is constructed by integrating **WISDM** and **MotionSense**
data into a unified HAR representation. In addition to the six inertial
channels, **pitch and roll** are derived from accelerometer measurements
to represent orientation-related information. Experiments compare **FedAvg, FedProx with
FedAvg server aggregation, ProtoFed, FedProto, and MOON** under a shared
evaluation workflow.

The emphasis is on research reproducibility and analysis, not just
reporting a single accuracy score: experiments include classification
metrics, calibration, confusion matrices, and t-SNE visualizations
across federated rounds.

## Research Objectives

- Literature Review on Federated Learning, identify and compare relevant
  methods for distributed Human Activity Recognition.
- Integrate WISDM and MotionSense into a consistent dataset while
  retaining participant identity for client construction.
- Derive pitch and roll from accelerometer signals and evaluate them
  as additional orientation-related features.
- Compare optimization-based and representation/prototype-based
  federated learning approaches under a common task.
- Study model performance, class-wise errors, probability calibration,
  and representation structure across federated rounds.
- Maintain a strict separation between training, validation, and test
  data for model fitting, selection, and final evaluation.

## Research Workflow

The project follows the workflow below, from literature review through
comparative evaluation.

```text
Literature Review
      ↓
WISDM + MotionSense Integration
      ↓
Resampling and Windowing (20 Hz, 60 time steps)
      ↓
Feature Engineering (6 sensor channels + pitch + roll)
      ↓
Client Construction and Train/Validation/Test Splits
      ↓
Federated Experiments (FedAvg, FedProx, ProtoFed, FedProto, MOON)
      ↓
Evaluation (Classification, Calibration, Confusion Matrices, t-SNE)
```

### 1. Literature Review

The project includes three presentation files documenting the literature
review and the development of the experimental plan:

- `FL_PPT.pdf` --- initial literature review and project context.
- `FL_PPT_2.pdf` --- continued literature review and research framing.
- `FL_PPT_3_Expirement.pdf` --- detailed presentation of the Federated
  Learning experiments.

The presentations provide the background and motivation for the selected
approaches. The original papers cited in the presentations should be
used as the primary sources for method definitions and academic claims.

### 2. Dataset Integration and Preparation

The dataset pipeline integrates selected activities from **WISDM** and
**MotionSense** into a unified representation.

### Activity Labels

- `A`: Walking
- `B`: Jogging
- `D`: Sitting
- `E`: Standing

**Signal representation**

- Target sampling frequency: **20 Hz**
- Window duration: **3 seconds**
- Samples per window: **60, timestamped at 50 ms each**
- Base sensor channels: `acc_x`, `acc_y`, `acc_z`, `gyro_x`, `gyro_y`,
  `gyro_z`
- Base window shape: `60 × 6`, equivalent to 360 sensor values per
  window
- Window overlap: **none**
- Filtering: **not applied**. No Butterworth, high-pass, low-pass, or
  band-pass filtering is used; the gravity component is retained.

Resampling and window construction are handled in the
dataset-preparation pipeline. Dataset-specific participant identifiers
are retained to support client construction and traceability.

**Pitch and roll feature engineering**

The `tilt_dataset.py` script derives pitch and roll from the three
accelerometer axes and appends them to the six base sensor channels. The
resulting representation has eight feature channels across 60 time
steps, or `60 × 8` values per window, excluding labels and metadata.

Pitch and roll provide orientation-related features that complement the
raw accelerometer and gyroscope measurements.

**Federated client construction**

Participant identity is preserved through `client_id` metadata, with
dataset-specific identifiers used to distinguish participants
originating from WISDM and MotionSense. This supports
participant-associated federated clients rather than treating the full
dataset as a single centralized pool.

> **Note:** UCI-HAR is inspected in `dataset_creation.py` but excluded from the final WISDM–MotionSense merge because its standard release contains pre-segmented 128-sample windows at 50 Hz, which do not match the project's 60-step window representation at 20 Hz.

## Train, Validation, and Test Protocol

The prepared dataset includes separate files:

- `tilt_fl_train.csv`
- `tilt_fl_validation.csv`
- `tilt_fl_test.csv`

The intended roles of the splits are:

- **Training set:** used to fit model parameters.
- **Validation set:** used for hyperparameter/model selection,
  validation monitoring, early stopping, and temperature fitting where
  implemented.
- **Test set:** reserved for final evaluation after the training and
  model-selection decisions have been made.

Feature scaling is fitted on the training split and then applied to
validation and test data using the same fitted transformation.

For a strict participant-independent protocol, a participant must not
contribute windows to more than one split. Separate CSV filenames alone
do not establish this property, so the split-generation records and
client IDs should be checked to verify it. Test results should not be
used to choose hyperparameters, checkpoints, or calibration parameters.

### 3. Federated Learning Experiments

- **FedAvg** (`fl_fedavg.py`): Baseline federated averaging.
- **FedProx + FedAvg** (`fl_fedprox_fedavg.py`): Proximal regularization during local training with FedAvg server aggregation.
- **ProtoFed** (`fl_protofed.py`): Prototype-oriented federated learning.
- **FedProto** (`fl_fedproto.py`): Class-prototype-based federated learning.
- **MOON** (`fl_moon.py`): Contrastive learning for local and global representation consistency.

The methods are evaluated on the same four-class HAR task. The
experiment scripts include method-specific training configurations,
validation monitoring, and reporting.

**Hyperparameter optimization (HPO)**

Optuna-based hyperparameter search is implemented in the FedAvg
experiment and uses **macro F1-score** as the optimization objective.
Hyperparameter search and the exact tuning procedure are
script-specific; refer to each implementation and its saved report when
comparing methods.

**Early stopping**

Validation-loss monitoring and patience-based early stopping are used to
stop training when validation performance no longer improves. This
limits unnecessary training and helps reduce overfitting. The validation
set is used for checkpoint/model-selection decisions, not as a
substitute for the held-out test set.

## Experimental Evaluation

### 1. Classification Metrics

- **Accuracy:** Overall prediction correctness.
- **Precision:** Correctness of predicted class labels.
- **Recall:** Ability to identify actual class samples.
- **F1-score:** Balance between precision and recall.
- **Macro F1-score:** Equal-weight performance across classes.
- **Weighted F1-score:** F1-score weighted by class support.
- **Balanced Accuracy:** Average recall across classes.
- **Classification Report:** Class-wise precision, recall, F1-score, and support.

### 2. Confusion Matrix

- Identifies correct and incorrect classifications.
- Highlights confusion between activities, particularly Walking–Jogging and Sitting–Standing.

### 3. Calibration Analysis

- **Reliability Diagram:** Compares confidence with observed accuracy.
- **Expected Calibration Error (ECE):** Average calibration error.
- **Maximum Calibration Error (MCE):** Maximum calibration error across bins.
- **Brier Score:** Measures probabilistic prediction error.
- **Log Loss (NLL):** Penalizes incorrect and overconfident predictions.
- **Temperature Scaling:** Post-training calibration using validation data.

### 4. Learning Curves and Early Stopping

- Training and validation loss/performance.
- Convergence and overfitting analysis.
- Early-stopping behavior.

### 5. t-SNE Visualization

- Visualizes learned representations across federated rounds.
- Examines class separation and overlap.
- Tracks representation patterns during training.

### 6. Comparative Experimental Reports

- Method-wise metrics and reports.
- Round-wise visualizations.
- Comparison of FedAvg, FedProx, ProtoFed, FedProto, and MOON.

## Repository Structure

```text
FL_code_github/
├── dataset_creation.py
├── tilt_dataset.py
├── fl_fedavg.py
├── fl_fedprox_fedavg.py
├── fl_protofed.py
├── fl_fedproto.py
├── fl_moon.py
├── FL_PPT.pdf
├── FL_PPT_2.pdf
├── FL_PPT_3_Expirement.pdf
├── tilt_fl_train.csv
├── tilt_fl_validation.csv
├── tilt_fl_test.csv
├── reports_fedavg/
├── reports_fedprox_fedavg/
├── reports_protofed/
├── reports_fedproto/
├── reports_moon/
├── visualizations_fedavg/
├── visualizations_fedprox_fedavg/
├── visualizations_fedproto/
├── visualizations_protofed/
├── visualizations_moon/
└── README.md
```

The repository also contains report spreadsheets and round-wise
visualization outputs for the implemented methods.

## Keywords

Federated Learning • Human Activity Recognition • Distributed Machine Learning • Wearable Sensor Analytics • Model Calibration • Representation Learning • FedAvg • FedProx • FedProto • MOON • WISDM • MotionSense

## Academic Supervision

This research project was conducted under the academic guidance and mentorship of:

**Dr. Suman Samui**\
Department of Electronics and Communication Engineering (ECE)\
National Institute of Technology (NIT) Durgapur

## Author

**Kingshuk Goswami**
