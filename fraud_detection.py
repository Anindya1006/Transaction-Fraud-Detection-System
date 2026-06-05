import streamlit as st
import pandas as pd 
import joblib
from pathlib import Path


MODEL_FILES = {
    "XGBoost": "fraud_detection_xgboost_pipeline.pkl",
    "Logistic Regression": "fraud_detection_logistic_pipeline.pkl",
    "Random Forest": "fraud_detection_random_forest_pipeline.pkl",
}


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


available_models = {
    model_name: model_path
    for model_name, model_path in MODEL_FILES.items()
    if Path(model_path).exists()
}

st.title("Transaction Fraud Prediction System")

st.markdown("Please enter the transaction details and use predict button")

st.divider()

if not available_models:
    st.error("No trained model file found. Please train and save a model pipeline first.")
    st.stop()

selected_model_name = st.sidebar.selectbox("Select Model", list(available_models.keys()))
model = load_model(available_models[selected_model_name])

if len(available_models) == 1:
    st.sidebar.info("Only one saved model was found.")

missing_models = [
    model_name
    for model_name, model_path in MODEL_FILES.items()
    if not Path(model_path).exists()
]

if missing_models:
    st.sidebar.caption("Train these models to show them here: " + ", ".join(missing_models))

transaction_type=st.selectbox("Transaction Type",["PAYMENT","TRANSFER","CASH_OUT","CASH_IN","DEBIT"])
amount=st.number_input("Amount",min_value=0.0,value=1000.0)
oldbalanceOrg=st.number_input("Old Balance (Sender)",min_value=0.0,value=1000.0)
newbalanceOrig=st.number_input("New Balance (Sender)",min_value=0.0,value=9000.0)
oldbalanceDest=st.number_input("Old Balance (Receiver)",min_value=0.0,value=0.0)
newbalanceDest=st.number_input("New Balance (Receiver)",min_value=0.0,value=0.0)

if st.button("Predict"):
    input_data=pd.DataFrame([{
        "type":transaction_type,
        "amount":amount,
        "oldbalanceOrg":oldbalanceOrg,
        "newbalanceOrig":newbalanceOrig,
        "oldbalanceDest":oldbalanceDest,
        "newbalanceDest":newbalanceDest,
        "balanceDiffOrig":oldbalanceOrg-newbalanceOrig,
        "balanceDiffDest":oldbalanceDest-newbalanceDest
        
    }])

    prediction=model.predict(input_data)[0]

    st.subheader(f"Model : {selected_model_name}")
    st.subheader(f"Prediction : '{int(prediction)}'")

    if hasattr(model, "predict_proba"):
        fraud_probability=model.predict_proba(input_data)[0][1]
        st.write(f"Fraud Probability: {fraud_probability:.2%}")

    if prediction==1:
        st.error("This transaction can be Fraud")
    else:
        st.success("This transaction is not a fraud")

