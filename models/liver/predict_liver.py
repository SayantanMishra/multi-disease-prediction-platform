# ============================
# INTERACTIVE LIVER DISEASE RISK PREDICTOR
# ============================
import joblib
import pandas as pd

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
columns = joblib.load("columns.pkl")
model_name = joblib.load("best_model_name.pkl")

print(f"Using model: {model_name}")
print("\n--- Enter your health details below ---\n")

user_data = {}

for col in columns:
    if col in encoders:
        options = list(encoders[col].classes_)
        print(f"{col} - Options: {options}")
        val = input(f"Enter {col}: ").strip()
        while val not in options:
            print("Invalid value. Please choose from:", options)
            val = input(f"Enter {col}: ").strip()
        user_data[col] = encoders[col].transform([val])[0]
    else:
        val = float(input(f"Enter {col} (number): "))
        user_data[col] = val

input_df = pd.DataFrame([user_data])[columns]

if model_name == "logistic_regression":
    input_scaled = scaler.transform(input_df)
    risk = model.predict_proba(input_scaled)[0][1]
else:
    risk = model.predict_proba(input_df)[0][1]

print(f"\n=== Liver Disease Risk: {round(risk*100, 2)}% ===")

if risk > 0.7:
    print("High risk — please consult a doctor.")
elif risk > 0.4:
    print("Moderate risk — monitor your health and consider a checkup.")
else:
    print("Low risk — keep maintaining a healthy lifestyle.")