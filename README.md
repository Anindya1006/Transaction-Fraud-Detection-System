# Transaction Fraud Prediction System

This project trains and compares machine learning models for transaction fraud prediction, then serves predictions through a Streamlit web app.
## Dataset
Used from Kaggle.
## Project Files

- `fraud_detection.py` - Streamlit app for entering transaction details and selecting a model for prediction.
- `model_evaluation_app.py` - Python script for comparing all saved models.
- `analysis_model.ipynb` - Logistic Regression training notebook.
- `analysis_model 1.ipynb` - Random Forest training notebook.
- `analysis_model 2.ipynb` - XGBoost training notebook.
- `AIML Dataset.csv` - Dataset used for training and analysis. This file is not included in Git because it is large; download or place it locally before running evaluation or retraining.
- `fraud_detection_logistic_pipeline.pkl` - Saved Logistic Regression pipeline.
- `fraud_detection_random_forest_pipeline.pkl` - Saved Random Forest pipeline.
- `fraud_detection_xgboost_pipeline.pkl` - Saved XGBoost pipeline.

## Models

The app supports three algorithms:

1. Logistic Regression
2. Random Forest
3. XGBoost

Each model is saved as a separate pipeline file so the Streamlit app can load and compare them from the sidebar.

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

The Streamlit prediction app can run with the saved `.pkl` model files. The model evaluation script and notebooks also require `AIML Dataset.csv` in the project root.

## Run the App

Start the prediction app:

```bash
streamlit run fraud_detection.py
```

Then open the local URL shown in the terminal.

Run the model evaluation script:

```bash
python model_evaluation_app.py
```

## How to Use

1. Select a model from the sidebar.
2. Enter the transaction details.
3. Click `Predict`.
4. The app displays:
   - selected model
   - prediction result
   - fraud probability, when supported by the model

Prediction output:

- `0` means the transaction is predicted as not fraud.
- `1` means the transaction is predicted as fraud.

## Retraining Models

Run the notebooks to regenerate the model files:

- Run `analysis_model.ipynb` to create `fraud_detection_logistic_pipeline.pkl`.
- Run `analysis_model 1.ipynb` to create `fraud_detection_random_forest_pipeline.pkl`.
- Run `analysis_model 2.ipynb` to create `fraud_detection_xgboost_pipeline.pkl`.

After retraining, restart the Streamlit app so it loads the updated `.pkl` files.

## Notes

The app automatically calculates these engineered features from the user inputs:

- `balanceDiffOrig = oldbalanceOrg - newbalanceOrig`
- `balanceDiffDest = oldbalanceDest - newbalanceDest`

These features are used by the trained model pipelines during prediction.

## Model Evaluation App

The evaluation app compares all three saved models using:

- `Accuracy` - percentage of correct predictions.
- `F1 Score` - balance between fraud precision and fraud recall.
- `Latency / Prediction` - average prediction time in milliseconds.
- `Robustness Score` - prediction consistency after small random changes to numeric inputs.

The robustness score is calculated by adding small noise to transaction amounts and balances, recomputing the engineered balance-difference features, and checking how often the model keeps the same prediction.
