# ============================
# LIVER DISEASE MODEL TRAINER (with Feature Selection)
# ============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ============================
# CONFIG
# ============================
TRAIN_CSV = "Training_indian_liver_disease_dataset.csv"
TEST_CSV = "Testing_indian_liver_disease_dataset.csv"
TARGET_COLUMN = "Liver_Disease_Type"
DROP_COLUMNS = ["Patient_ID"]
TOP_N_FEATURES = 12  

# ============================
# STEP 1: Load and Combine Data
# ============================
df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)
df = pd.concat([df_train, df_test], ignore_index=True)
print("Combined shape:", df.shape)

df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

# ============================
# STEP 2: Convert target to binary
# ============================
df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(lambda x: 0 if x == "Normal" else 1)

# ============================
# STEP 3: Generic Cleaning
# ============================
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
if TARGET_COLUMN in num_cols:
    num_cols.remove(TARGET_COLUMN)

cat_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ============================
# STEP 4: Feature Importance Calculation (NEW STEP)
# Train a quick Random Forest on ALL features just to rank them
# ============================
X_full = df.drop(columns=[TARGET_COLUMN])
y_full = df[TARGET_COLUMN]

print("\nCalculating feature importance...")
importance_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
importance_model.fit(X_full, y_full)

importance_df = pd.DataFrame({
    "feature": X_full.columns,
    "importance": importance_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n=== Feature Importance Ranking ===")
print(importance_df.to_string(index=False))

# Select top N features
selected_features = importance_df["feature"].head(TOP_N_FEATURES).tolist()
print(f"\nSelected Top {TOP_N_FEATURES} Features:", selected_features)

# ============================
# STEP 5: Reduce dataset to selected features only
# ============================
X = df[selected_features]
y = df[TARGET_COLUMN]

# ============================
# STEP 6: Split Data
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================
# STEP 7: Train Base Models (on selected features only)
# ============================
models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "xgboost": XGBClassifier(eval_metric="logloss", random_state=42)
}

results = {}
for name, model in models.items():
    if name == "logistic_regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    results[name] = {"model": model, "accuracy": acc, "auc": auc}
    print(f"\n{name} -> Accuracy: {round(acc*100,2)}%, AUC: {round(auc,3)}")
    print(classification_report(y_test, preds))

# ============================
# STEP 8: Pick Best Base Model
# ============================
best_name = max(results, key=lambda x: results[x]["auc"])
print(f"\nBest base model (by AUC): {best_name}")

# ============================
# STEP 9: Hyperparameter Tuning
# ============================
if best_name == "random_forest":
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                                param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    tuned_model = grid_search.best_estimator_
    tuned_preds = tuned_model.predict(X_test)
    tuned_probs = tuned_model.predict_proba(X_test)[:, 1]

elif best_name == "xgboost":
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2]
    }
    grid_search = GridSearchCV(XGBClassifier(eval_metric="logloss", random_state=42),
                                param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    tuned_model = grid_search.best_estimator_
    tuned_preds = tuned_model.predict(X_test)
    tuned_probs = tuned_model.predict_proba(X_test)[:, 1]

else:
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs']
    }
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000),
                                param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    tuned_model = grid_search.best_estimator_
    tuned_preds = tuned_model.predict(X_test_scaled)
    tuned_probs = tuned_model.predict_proba(X_test_scaled)[:, 1]

print("Best params:", grid_search.best_params_)
print("Best CV F1 score:", round(grid_search.best_score_, 3))

final_model = tuned_model
final_preds = tuned_preds
final_probs = tuned_probs

tuned_acc = accuracy_score(y_test, final_preds)
tuned_auc = roc_auc_score(y_test, final_probs)
print(f"\nTuned {best_name} -> Accuracy: {round(tuned_acc*100,2)}%, AUC: {round(tuned_auc,3)}")

# ============================
# STEP 10: Confusion Matrix
# ============================
cm = confusion_matrix(y_test, final_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Liver Disease', 'Liver Disease'],
            yticklabels=['No Liver Disease', 'Liver Disease'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_name} (tuned)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# ============================
# STEP 11: ROC Curve
# ============================
fpr, tpr, _ = roc_curve(y_test, final_probs)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {tuned_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - {best_name} (tuned)")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()

# Also save feature importance chart
plt.figure(figsize=(8, 6))
sns.barplot(data=importance_df.head(TOP_N_FEATURES), x="importance", y="feature")
plt.title("Top Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

print("\nConfusion matrix saved as confusion_matrix.png")
print("ROC curve saved as roc_curve.png")
print("Feature importance chart saved as feature_importance.png")

# ============================
# STEP 12: Save Everything
# ============================
joblib.dump(final_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(list(X.columns), "columns.pkl")   
joblib.dump(best_name, "best_model_name.pkl")

print("\nModel, scaler, encoders saved! Now you can run predict_liver.py.")