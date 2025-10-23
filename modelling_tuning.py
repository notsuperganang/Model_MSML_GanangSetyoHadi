"""
Machine Learning Model Training with DagsHub Integration
Author: Ganang Setyo Hadi
Description: Train a heart disease prediction model with DagsHub remote tracking
Level: Advanced (4 pts) - DagsHub remote tracking + autolog metrics + additional custom metrics
"""

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    log_loss, matthews_corrcoef, balanced_accuracy_score,
    cohen_kappa_score, jaccard_score
)
import numpy as np
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# DagsHub Configuration (from environment variables)
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "notsuperganang")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "MSML-Dicoding")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", 
                                f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow")

# Set MLflow tracking URI to DagsHub
DAGSHUB_MLFLOW_URI = MLFLOW_TRACKING_URI

def setup_dagshub():
    """Setup DagsHub MLflow tracking with proper authentication"""
    print("Setting up DagsHub MLflow tracking...")
    print(f"DagsHub URI: {DAGSHUB_MLFLOW_URI}")
    print(f"Username: {DAGSHUB_USERNAME}")
    
    # IMPORTANT: Set credentials BEFORE setting tracking URI
    if DAGSHUB_TOKEN:
        os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
        print("Authentication configured with token")
    else:
        raise ValueError(
            "DAGSHUB_TOKEN is required! Please set it in your .env file.\n"
            "Get your token from: https://dagshub.com/user/settings/tokens"
        )
    
    # Set tracking URI to DagsHub AFTER credentials are set
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    
    print("DagsHub setup completed!")


def load_data():
    """Load preprocessed training and testing data"""
    print("Loading preprocessed data...")
    
    # Load training data
    X_train = pd.read_csv('heart_preprocessing/X_train.csv')
    y_train = pd.read_csv('heart_preprocessing/y_train.csv')
    
    # Load testing data
    X_test = pd.read_csv('heart_preprocessing/X_test.csv')
    y_test = pd.read_csv('heart_preprocessing/y_test.csv')
    
    # Convert to numpy arrays if needed
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    
    print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing data shape: X_test={X_test.shape}, y_test={y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_model_advanced(X_train, X_test, y_train, y_test):
    """Train Random Forest model with DagsHub tracking and comprehensive logging"""
    
    # Disable autolog - we'll use manual logging with more metrics
    mlflow.sklearn.autolog(disable=True)
    
    print("\nStarting MLflow run with DagsHub tracking...")
    
    with mlflow.start_run(run_name="RandomForest_DagsHub_Advanced"):
        
        # Start timing
        start_time = time.time()
        
        # Define hyperparameter grid for tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False]
        }
        
        print("Performing Grid Search with Cross-Validation...")
        print(f"Parameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")
        
        # Create base model
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        print(f"\nBest parameters found: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)
        y_train_pred = best_model.predict(X_train)
        
        # Calculate all standard metrics (autolog metrics)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Training set metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
        
        # Additional advanced metrics (beyond autolog)
        try:
            test_roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        except:
            test_roc_auc = 0.0
        
        try:
            test_logloss = log_loss(y_test, y_pred_proba)
        except:
            test_logloss = 0.0
            
        test_mcc = matthews_corrcoef(y_test, y_pred)
        test_balanced_acc = balanced_accuracy_score(y_test, y_pred)
        
        # ADVANCED: Additional custom metrics (2+ beyond autolog)
        test_cohen_kappa = cohen_kappa_score(y_test, y_pred)
        test_jaccard = jaccard_score(y_test, y_pred, average='weighted')
        
        # Calculate overfitting metrics
        overfit_gap_accuracy = train_accuracy - test_accuracy
        overfit_gap_f1 = train_f1 - test_f1
        
        training_time = time.time() - start_time
        
        # Manual logging of parameters (best params from grid search)
        print("\nLogging parameters to DagsHub...")
        for param_name, param_value in grid_search.best_params_.items():
            mlflow.log_param(param_name, param_value)
        
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_jobs", -1)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("tuning_method", "GridSearchCV")
        mlflow.log_param("total_combinations_tested", len(grid_search.cv_results_['params']))
        
        # Manual logging of all metrics
        print("Logging metrics to DagsHub...")
        
        # Standard metrics (same as autolog)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision_weighted", test_precision)
        mlflow.log_metric("test_recall_weighted", test_recall)
        mlflow.log_metric("test_f1_weighted", test_f1)
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        mlflow.log_metric("test_log_loss", test_logloss)
        
        # Training metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_precision_weighted", train_precision)
        mlflow.log_metric("train_recall_weighted", train_recall)
        mlflow.log_metric("train_f1_weighted", train_f1)
        
        # ADVANCED: Additional custom metrics (beyond autolog)
        mlflow.log_metric("test_matthews_corrcoef", test_mcc)
        mlflow.log_metric("test_balanced_accuracy", test_balanced_acc)
        mlflow.log_metric("test_cohen_kappa", test_cohen_kappa)
        mlflow.log_metric("test_jaccard_score", test_jaccard)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        mlflow.log_metric("training_time_seconds", training_time)
        mlflow.log_metric("overfit_gap_accuracy", overfit_gap_accuracy)
        mlflow.log_metric("overfit_gap_f1", overfit_gap_f1)
        
        # Log confusion matrix as text
        cm = confusion_matrix(y_test, y_pred)
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        
        # Log classification report
        report = classification_report(y_test, y_pred)
        mlflow.log_text(report, "classification_report.txt")
        
        # Log feature importances
        feature_names = X_train.columns.tolist()
        importances = best_model.feature_importances_
        feature_importance_dict = dict(zip(feature_names, importances))
        
        # Sort and log all features
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        importance_text = "\n".join([f"{feat}: {imp:.4f}" for feat, imp in sorted_features])
        mlflow.log_text(importance_text, "feature_importances.txt")
        
        # Log CV results summary
        cv_results_summary = f"Mean CV Score: {grid_search.best_score_:.4f}\n"
        cv_results_summary += f"Std CV Score: {grid_search.cv_results_['std_test_score'][grid_search.best_index_]:.4f}\n"
        mlflow.log_text(cv_results_summary, "cv_results_summary.txt")
        
        # Log the model
        print("Logging model to DagsHub...")
        mlflow.sklearn.log_model(
            best_model,
            "model",
            registered_model_name="HeartDiseaseRF_DagsHub"
        )
        
        # Log additional tags
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag("tuning_method", "GridSearchCV")
        mlflow.set_tag("level", "Advanced")
        mlflow.set_tag("tracking", "DagsHub")
        mlflow.set_tag("author", "Ganang Setyo Hadi")
        mlflow.set_tag("dataset", "Heart Disease UCI")
        
        print(f"\n{'='*70}")
        print("Model Performance Summary:")
        print(f"{'='*70}")
        print("\nTest Set Metrics:")
        print(f"  Accuracy:           {test_accuracy:.4f}")
        print(f"  Precision:          {test_precision:.4f}")
        print(f"  Recall:             {test_recall:.4f}")
        print(f"  F1-Score:           {test_f1:.4f}")
        print(f"  ROC-AUC:            {test_roc_auc:.4f}")
        print(f"  Log Loss:           {test_logloss:.4f}")
        print(f"  Matthews Corr:      {test_mcc:.4f}")
        print(f"  Balanced Accuracy:  {test_balanced_acc:.4f}")
        print(f"  Cohen's Kappa:      {test_cohen_kappa:.4f}")
        print(f"  Jaccard Score:      {test_jaccard:.4f}")
        
        print("\nTraining Set Metrics:")
        print(f"  Accuracy:           {train_accuracy:.4f}")
        print(f"  F1-Score:           {train_f1:.4f}")
        
        print("\nModel Diagnostics:")
        print(f"  Best CV Score:      {grid_search.best_score_:.4f}")
        print(f"  Training Time:      {training_time:.2f}s")
        print(f"  Overfit Gap (Acc):  {overfit_gap_accuracy:.4f}")
        print(f"  Overfit Gap (F1):   {overfit_gap_f1:.4f}")
        print(f"{'='*70}")
        
        print("\nConfusion Matrix:")
        print(cm)
        
        print("\nTop 5 Important Features:")
        for feat, imp in sorted_features[:5]:
            print(f"  {feat}: {imp:.4f}")
        
        print("\nModel training completed successfully!")
        print(f"Model and metrics saved to DagsHub: {mlflow.get_tracking_uri()}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        
        return best_model, grid_search.best_params_

def main():
    """Main function to orchestrate the training process"""
    print("=" * 70)
    print("Heart Disease Prediction Model Training")
    print("With DagsHub Remote Tracking (Advanced Level)")
    print("=" * 70)
    
    # Setup DagsHub
    setup_dagshub()
    
    # Set experiment name
    mlflow.set_experiment("Heart_Disease_Prediction_Advanced")
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train model with DagsHub tracking
    model, best_params = train_model_advanced(X_train, X_test, y_train, y_test)
    
    print("\n" + "=" * 70)
    print("Training process completed!")
    print(f"View results at: {DAGSHUB_MLFLOW_URI}")
    print("=" * 70)

if __name__ == "__main__":
    main()
