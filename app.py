import os
import pickle
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "breast_cancer_model.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)
st.subheader("Tumor Features")

# User inputs
mean_radius = st.number_input("Mean Radius", min_value=0.0, max_value=50.0, value=14.0, step=0.1)
mean_texture = st.number_input("Mean Texture", min_value=0.0, max_value=40.0, value=20.0, step=0.1)
mean_perimeter = st.number_input("Mean Perimeter", min_value=0.0, max_value=200.0, value=90.0, step=0.1)
mean_area = st.number_input("Mean Area", min_value=0.0, max_value=3000.0, value=600.0, step=0.1)
mean_smoothness = st.number_input("Mean Smoothness", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

# Prepare input for prediction
input_data = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]])
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Tumor Type"):
    prediction = model.predict(input_scaled)
    result = "Benign ✅" if prediction[0] == 1 else "Malignant ❌"
    st.success(f"Prediction: {result}")
