# Heart Disease Prediction - MLflow Model Training

Machine Learning model training pipeline for heart disease prediction using Random Forest Classifier with MLflow tracking and DagsHub integration.

## 🎯 Project Overview

This project implements a complete ML training pipeline that meets the **Advanced (4 pts)** criteria for MLOps submission:

- ✅ **Hyperparameter tuning** with GridSearchCV
- ✅ **Manual logging** (not autolog) with comprehensive metrics
- ✅ **DagsHub remote tracking** for MLflow experiments
- ✅ **2+ additional custom metrics** beyond standard autolog
- ✅ **Model registry** integration

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | 88.59% |
| ROC-AUC | 93.01% |
| F1-Score | 88.54% |
| Precision | 88.63% |
| Recall | 88.59% |
| Cohen's Kappa | 76.76% |
| Jaccard Score | 79.48% |
| Matthews Correlation | 76.88% |

**Best CV Score:** 85.96% (5-fold cross-validation)

## 🏗️ Project Structure

```
Membangun_model/
├── modelling.py                          # Basic model (2 pts) - Local MLflow with autolog
├── modelling_tuning.py                   # Advanced model (4 pts) - DagsHub + manual logging
├── requirements.txt                      # Python dependencies
├── DagsHub.txt                          # DagsHub repository URLs
├── .env.example                         # Environment variables template
├── .gitignore                           # Git ignore rules
├── README.md                            # This file
├── heart_preprocessing/                 # Preprocessed dataset
│   ├── X_train.csv                     # Training features (734 samples)
│   ├── X_test.csv                      # Testing features (184 samples)
│   ├── y_train.csv                     # Training labels
│   └── y_test.csv                      # Testing labels
├── screenshot_dashboard_dagshub.png     # MLflow experiment dashboard
└── screenshot_artifact_dagshub.png      # Model artifacts screenshot
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda package manager
- DagsHub account (for advanced training)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/notsuperganang/Eksperimen_SML_GanangSetyoHadi.git
   cd Membangun_model
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure DagsHub credentials** (for Advanced model)
   ```bash
   cp .env.example .env
   # Edit .env and add your DagsHub token
   ```

   Get your token from: https://dagshub.com/user/settings/tokens

## 📝 Usage

### Option 1: Basic Model (Local MLflow with Autolog)

Train a basic Random Forest model with MLflow autolog:

```bash
python modelling.py
```

This will:
- Train a Random Forest classifier (100 estimators)
- Log metrics automatically with MLflow autolog
- Save model and artifacts to local `mlruns/` directory
- Create experiment: `Heart_Disease_Prediction_Basic`

View results:
```bash
mlflow ui
# Open http://localhost:5000
```

### Option 2: Advanced Model (DagsHub + Manual Logging)

Train an optimized model with hyperparameter tuning and DagsHub tracking:

```bash
python modelling_tuning.py
```

This will:
- Perform GridSearchCV with 432 parameter combinations
- Train with 5-fold cross-validation
- Log all metrics manually (not autolog)
- Save to DagsHub remote tracking server
- Register model in DagsHub model registry
- Create experiment: `Heart_Disease_Prediction_Advanced`

View results on DagsHub:
- Repository: https://dagshub.com/notsuperganang/MSML-Dicoding
- Experiments: https://dagshub.com/notsuperganang/MSML-Dicoding.mlflow

## 🔧 Features

### Hyperparameter Tuning

GridSearchCV explores 432 combinations of:
- `n_estimators`: [100, 200, 300]
- `max_depth`: [10, 20, 30, None]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]
- `max_features`: ['sqrt', 'log2']
- `bootstrap`: [True, False]

### Logged Parameters

- All best hyperparameters from GridSearchCV
- Random state (42)
- Number of jobs (-1, all cores)
- CV folds (5)
- Tuning method (GridSearchCV)
- Total combinations tested

### Logged Metrics

**Standard Metrics (same as autolog):**
- test_accuracy
- test_precision_weighted
- test_recall_weighted
- test_f1_weighted
- test_roc_auc
- test_log_loss
- train_accuracy
- train_f1_weighted

**Additional Custom Metrics (Advanced Level):**
- test_matthews_corrcoef
- test_balanced_accuracy
- **test_cohen_kappa** ✨ (Additional #1)
- **test_jaccard_score** ✨ (Additional #2)
- best_cv_score
- training_time_seconds
- overfit_gap_accuracy
- overfit_gap_f1

### Logged Artifacts

- Trained model (sklearn format)
- Confusion matrix (text)
- Classification report
- Feature importances (all 16 features)
- CV results summary
- Model metadata and dependencies

## 📊 Dataset

**Heart Disease UCI Dataset** (preprocessed)

- **Features:** 16 (after preprocessing and encoding)
- **Samples:** 918 total
  - Training: 734 (80%)
  - Testing: 184 (20%)
- **Target:** Binary classification (0: No disease, 1: Disease)
- **Preprocessing:** StandardScaler normalization, One-Hot Encoding

### Top 5 Important Features

1. **ST_Slope** (28.37%) - Slope of peak exercise ST segment
2. **ChestPainType_ASY** (16.32%) - Asymptomatic chest pain
3. **Oldpeak** (9.90%) - ST depression induced by exercise
4. **MaxHR** (9.08%) - Maximum heart rate achieved
5. **ExerciseAngina** (8.94%) - Exercise-induced angina

## 🎓 Submission Criteria

This project fulfills the **Advanced (4 pts)** criteria:

| Criterion | Status |
|-----------|--------|
| Hyperparameter tuning | ✅ GridSearchCV with 432 combinations |
| Manual logging | ✅ All metrics logged manually (not autolog) |
| DagsHub remote tracking | ✅ MLflow experiments saved to DagsHub |
| Standard autolog metrics | ✅ accuracy, precision, recall, f1, roc_auc, log_loss |
| 2+ Additional metrics | ✅ Cohen's Kappa & Jaccard Score |
| Model registry | ✅ HeartDiseaseRF_DagsHub v1 registered |
| Artifacts | ✅ Model, confusion matrix, feature importances |
| Screenshots | ✅ Dashboard & artifacts screenshots included |

## 🔗 Links

- **DagsHub Repository:** https://dagshub.com/notsuperganang/MSML-Dicoding
- **MLflow Tracking:** https://dagshub.com/notsuperganang/MSML-Dicoding.mlflow
- **Latest Experiment:** https://dagshub.com/notsuperganang/MSML-Dicoding.mlflow/#/experiments/0
- **GitHub Repository:** https://github.com/notsuperganang/Eksperimen_SML_GanangSetyoHadi

## 📦 Dependencies

```
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
mlflow==2.9.0
matplotlib==3.8.2
seaborn==0.13.0
imbalanced-learn==0.11.0
python-dotenv==1.0.0
dagshub
```

## 🔐 Environment Variables

Create a `.env` file based on `.env.example`:

```bash
DAGSHUB_USERNAME=your_username
DAGSHUB_REPO_NAME=your_repo_name
DAGSHUB_TOKEN=your_token_here
MLFLOW_TRACKING_URI=https://dagshub.com/your_username/your_repo.mlflow
```

**Security Note:** Never commit the `.env` file to version control!

## 🐛 Troubleshooting

### Authentication Error (401)

If you get a 401 error when running `modelling_tuning.py`:

1. Check that your `.env` file exists and has the correct token
2. Verify token at: https://dagshub.com/user/settings/tokens
3. Ensure token has `read:repo`, `write:repo`, and `experiments` scopes

### Missing Dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Model Schema Warning

Warning about "Model logged without signature" is normal and safe. The model is still fully functional.

## 📄 License

This project is part of an academic submission for Machine Learning Operations course.

## 👤 Author

**Ganang Setyo Hadi**
- GitHub: [@notsuperganang](https://github.com/notsuperganang)
- DagsHub: [@notsuperganang](https://dagshub.com/notsuperganang)

## 🙏 Acknowledgments

- Dicoding Indonesia - MLOps Course
- UCI Machine Learning Repository - Heart Disease Dataset
- DagsHub - MLflow Remote Tracking Platform

---

**Note:** This project demonstrates MLOps best practices including experiment tracking, model registry, hyperparameter tuning, and remote collaboration using DagsHub.
