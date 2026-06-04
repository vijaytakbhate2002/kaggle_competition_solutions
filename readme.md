# Kaggle Competitions Repository

## F1 Pit Stop Prediction

### Competition Overview

**Playground Series - Season 6, Episode 5**

- **Final Position**: Top 22% (Rank 679 out of 3023)
- **Objective**: Predict whether a Formula 1 driver will make a pit stop in the next lap

### Final Solution Architecture

The final solution uses an **ensemble approach** combining multiple state-of-the-art models:

**Models Used:**

- **XGBoost**: Gradient boosting model with optimized hyperparameters (1387 estimators, max_depth=12)
- **LightGBM**: Fast gradient boosting with 3384 estimators and advanced regularization
- **CatBoost**: Categorical-aware gradient boosting (1635 iterations, depth=8)
- **RealMLP**: Advanced PyTorch-based neural network (16 ensemble networks, custom architecture)

### Key Experiments & Techniques

#### Feature Engineering

- Domain knowledge-based feature creation
- Arithmetic interactions (LapNumber/RaceProgress, TyreLife/LapNumber, etc.)
- Categorical feature engineering and encoding
- Count encoding for categorical variables
- Binning and discretization of numerical features
- Interaction feature combinations (Race-Compound, Race-Year)

#### Data Processing & Encoding

- Label encoding for categorical variables
- One-hot encoding for selected features
- Normalization of numerical features
- Optimal binning using OptimalBinning algorithm

#### Advanced Preprocessing

- **Target Encoding with OOF (Out-of-Fold)**: 5-fold cross-validation approach for target encoding
- **WoE (Weight of Evidence) Transformation**: For feature importance and monotonic relationships
- **Information Value (IV)**: Feature selection and ranking

#### Hyperparameter Optimization

- **Optuna-based optimization** for model parameters
- Grid search and Bayesian optimization
- Class weight balancing (handling imbalanced data)

#### Model Training & Validation

- Stratified K-Fold cross-validation
- Train/validation/test split strategy
- OOF predictions for ensemble blending
- Evaluation metric: ROC-AUC Score

#### Ensemble & Blending

- Model blending using multiple trained models
- Weighted averaging of predictions
- Out-of-time (OOT) validation on separate test set

### Solution Files Structure

- **Final Solution**: [`f1-pit-stop-prediction-xgb-lgb-cb-realmlp.ipynb`](f1_pit_stop_prediction/f1-pit-stop-prediction-xgb-lgb-cb-realmlp.ipynb)
- **Model Experiments**:
  - [`f1-pit-stop-prediction-catboost.ipynb`](f1_pit_stop_prediction/f1-pit-stop-prediction-catboost.ipynb)
  - [`f1-pit-stop-prediction-catboost_v2.ipynb`](f1_pit_stop_prediction/f1-pit-stop-prediction-catboost_v2.ipynb)
  - [`ps-s6-e5-realmlp-pytorch.ipynb`](f1_pit_stop_prediction/ps-s6-e5-realmlp-pytorch.ipynb)
- **Iteration Experiments**:
  - [`f1_pit_stop_prediction_23_05_2026.ipynb`](f1_pit_stop_prediction/f1_pit_stop_prediction_23_05_2026.ipynb)
  - [`f1_pit_stop_prediction_23_05_2026_EXP.ipynb`](f1_pit_stop_prediction/f1_pit_stop_prediction_23_05_2026_EXP.ipynb)
  - [`F1_PIT_STOP_24_05_2026_FINAL.ipynb`](f1_pit_stop_prediction/F1_PIT_STOP_24_05_2026_FINAL.ipynb)
  - [`F1_PIT_STOP_24_05_2026_FINAL_EXP.ipynb`](f1_pit_stop_prediction/F1_PIT_STOP_24_05_2026_FINAL_EXP.ipynb)
- **Ensemble & Blending**: [`Blender.ipynb`](f1_pit_stop_prediction/Blender.ipynb)

### Key Results

- XGBoost ROC-AUC: ~0.8900+ (on validation)
- LightGBM ROC-AUC: ~0.8850+ (on validation)
- CatBoost ROC-AUC: ~0.8800+ (on validation)
- RealMLP ROC-AUC: ~0.8750+ (on validation)
- **Ensemble Blended Score**: Achieved rank 679/3023 (Top 22%)

---

## Predicting Stellar Class

### Competition Overview

**Playground Series - Season 6, Episode 6** (Ongoing)

- **Status**: In Progress - Preparing first solution
- **Objective**: Classify stellar objects based on astronomical features

### Initial Approach & Experiments

#### Notebooks

- **Base Solution**: [`real_mlp_model_training_base.ipynb`](predicting_stellar_class/real_mlp_model_training_base.ipynb)
- **XGBoost Solution**: [`xgboost_solution.ipynb`](predicting_stellar_class/xgboost_solution.ipynb)

#### Planned Methodology

- Feature exploration and EDA
- Multiple model implementations (XGBoost, LightGBM, RealMLP)
- Hyperparameter tuning
- Ensemble methods
- Cross-validation strategies

**Note**: This competition is ongoing. More details will be added as the solution develops.

---

## Environment Setup

### Requirements

- Python 3.8+
- PyTorch (with CUDA support)
- XGBoost
- LightGBM
- CatBoost
- Scikit-learn
- Pandas & NumPy
- PyTabKit (for RealMLP)
- Optuna (for hyperparameter optimization)
- OptBinning (for WoE transformations)

See `requirements.txt` for complete dependencies list.

### Virtual Environment

A Python virtual environment (`comp_env/`) is configured in the workspace.

---

## Key Technologies Used

| Technology       | Purpose                                    |
| ---------------- | ------------------------------------------ |
| **XGBoost**      | Gradient boosting with custom parameters   |
| **LightGBM**     | Fast, memory-efficient gradient boosting   |
| **CatBoost**     | Handles categorical features natively      |
| **PyTorch**      | Deep learning framework for RealMLP        |
| **PyTabKit**     | Advanced tabular neural networks (RealMLP) |
| **Optuna**       | Automated hyperparameter optimization      |
| **Scikit-learn** | Data preprocessing and evaluation metrics  |
| **Pandas**       | Data manipulation and analysis             |

---

## Competition Strategy Summary

### Ranking Factors (F1 Pit Stop Prediction)

1. **Robust Feature Engineering**: Domain-specific features with mathematical relationships
2. **Advanced Preprocessing**: Target encoding, WoE transformation, optimal binning
3. **Diverse Model Selection**: Combining tree-based and neural network approaches
4. **Hyperparameter Optimization**: Systematic tuning using Optuna
5. **Ensemble Methods**: Effective blending of predictions from multiple models
6. **Cross-validation Strategy**: Stratified K-fold with OOF predictions

### Success Metrics

- Top 22% finish on F1 Pit Stop Prediction
- Effective ensemble performance
- Robust cross-validation preventing overfitting
