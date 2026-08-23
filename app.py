import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

st.set_page_config(page_title="Multi-Disease Prediction Platform", layout="centered")
st.title("🩺 Multi-Disease Prediction Platform")

disease = st.sidebar.selectbox(
    "Select a disease to check",
    ["Heart Disease", "Diabetes", "Liver Disease", "Diabetic Retinopathy", "Lumbar Spine (Scoliosis/Spondylolisthesis)"]
)

# ============================================================
# TABULAR DISEASES (Heart, Diabetes, Liver)
# ============================================================
def run_tabular_model(folder, title):
    model = joblib.load(f"models/{folder}/best_model.pkl")
    scaler = joblib.load(f"models/{folder}/scaler.pkl")
    encoders = joblib.load(f"models/{folder}/encoders.pkl")
    columns = joblib.load(f"models/{folder}/columns.pkl")
    model_name = joblib.load(f"models/{folder}/best_model_name.pkl")

    st.header(title)
    user_data = {}

    for col in columns:
        if col in encoders:
            options = list(encoders[col].classes_)
            val = st.selectbox(col, options, key=f"{folder}_{col}")
            user_data[col] = encoders[col].transform([val])[0]
        else:
            val = st.number_input(col, value=0.0, key=f"{folder}_{col}")
            user_data[col] = val

    if st.button("Predict Risk", key=f"{folder}_btn"):
        input_df = pd.DataFrame([user_data])[columns]
        if model_name == "logistic_regression":
            risk = model.predict_proba(scaler.transform(input_df))[0][1]
        else:
            risk = model.predict_proba(input_df)[0][1]
        risk_percent = round(risk * 100, 2)

        st.metric("Risk %", f"{risk_percent}%")
        if risk_percent > 70:
            st.error("High risk — please consult a doctor.")
        elif risk_percent > 40:
            st.warning("Moderate risk — monitor your health and consider a checkup.")
        else:
            st.success("Low risk — keep maintaining a healthy lifestyle.")

# ============================================================
# IMAGE-BASED DISEASES (Retinopathy, Lumbar Spine)
# ============================================================
def run_image_model(model_path, class_names_path, title, manual_rescale, normal_label=None):
    import tensorflow as tf

    st.header(title)
    uploaded_file = st.file_uploader("Upload an image (X-ray / retina scan)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded image", use_container_width=True)

        if st.button("Predict", key=title):
            model = tf.keras.models.load_model(model_path)
            with open(class_names_path) as f:
                class_names = [line.strip() for line in f]

            img_resized = img.resize((128, 128))
            arr = np.array(img_resized)
            if manual_rescale:
                arr = arr / 255.0
            arr = np.expand_dims(arr, axis=0)

            preds = model.predict(arr)[0]

            st.subheader("Prediction Results")
            for cn, p in zip(class_names, preds):
                st.write(f"{cn}: {round(p * 100, 2)}%")

            predicted_class = class_names[np.argmax(preds)]
            st.success(f"Most Likely Class: {predicted_class}")

            if normal_label and normal_label in class_names:
                normal_idx = class_names.index(normal_label)
                risk = round((1 - preds[normal_idx]) * 100, 2)
                st.metric("Overall Disease Risk", f"{risk}%")

# ============================================================
# ROUTING
# ============================================================
if disease == "Heart Disease":
    run_tabular_model("heart", "Heart Disease Risk Prediction")
elif disease == "Diabetes":
    run_tabular_model("diabetes", "Diabetes Risk Prediction")
elif disease == "Liver Disease":
    run_tabular_model("liver", "Liver Disease Risk Prediction")
elif disease == "Diabetic Retinopathy":
    run_image_model(
        "models/retinopathy/retinopathy_model.keras",
        "models/retinopathy/class_names.txt",
        "Diabetic Retinopathy Detection",
        manual_rescale=False,
        normal_label="No_DR"
    )
elif disease == "Lumbar Spine (Scoliosis/Spondylolisthesis)":
    run_image_model(
        "models/lumbar/lumbar_model.keras",
        "models/lumbar/class_names.txt",
        "Lumbar Spine X-ray Classification",
        manual_rescale=True,
        normal_label="NormalFinal"
    )