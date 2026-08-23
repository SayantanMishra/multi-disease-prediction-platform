# ============================
# GENERIC HEART DISEASE MODEL TRAINER
# Works on any sim  ilar tabular dataset if you set TARGET_COLUMN correctly
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
# CONFIG - change these if dataset changes
CSV_FILE = "heart_disease_uci.csv"
TARGET_COLUMN = "num"             # column that tells disease or not
DROP_COLUMNS = ["id", "dataset"]  # columns that are useless for prediction

#Load Data
df = pd.read_csv(CSV_FILE)
print("Original shape:", df.shape)

df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

# STEP 2: Make target binary (0 = no disease, 1 = disease)
df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(lambda x: 1 if x > 0 else 0)

# STEP 3: Generic Cleaning (works for ANY dataset)
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
if TARGET_COLUMN in num_cols:
    num_cols.remove(TARGET_COLUMN)

cat_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()

print("Numeric columns detected:", num_cols)
print("Categorical columns detected:", cat_cols)

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# STEP 4: Split Data
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# STEP 5: Train Base Models
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
    results[name] = {"model": model, "accuracy": acc, "auc": auc,
                      "preds": preds, "probs": probs}
    print(f"\n{name} -> Accuracy: {round(acc*100,2)}%, AUC: {round(auc,3)}")
    print(classification_report(y_test, preds))


# STEP 6: Pick Best Base Model
best_name = max(results, key=lambda x: results[x]["auc"])
print(f"\nBest base model (by AUC): {best_name}")


# STEP 7: Hyperparameter Tuning with GridSearchCV
# Currently tunes Random Forest and XGBoost. Add more grids if needed.
tuned_model = None

if best_name == "random_forest":
    print("\nRunning GridSearchCV on Random Forest...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=5, scoring='f1', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    tuned_model = grid_search.best_estimator_
    tuned_preds = tuned_model.predict(X_test)
    tuned_probs = tuned_model.predict_proba(X_test)[:, 1]

elif best_name == "xgboost":
    print("\nRunning GridSearchCV on XGBoost...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    grid_search = GridSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=42),
        param_grid, cv=5, scoring='f1', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    tuned_model = grid_search.best_estimator_
    tuned_preds = tuned_model.predict(X_test)
    tuned_probs = tuned_model.predict_proba(X_test)[:, 1]

else:  # logistic_regression
    print("\nRunning GridSearchCV on Logistic Regression...")
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs']
    }
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid, cv=5, scoring='f1', n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    tuned_model = grid_search.best_estimator_
    tuned_preds = tuned_model.predict(X_test_scaled)
    tuned_probs = tuned_model.predict_proba(X_test_scaled)[:, 1]
print("Best params:", grid_search.best_params_)
print("Best CV F1 score:", round(grid_search.best_score_, 3))

tuned_acc = accuracy_score(y_test, tuned_preds)
tuned_auc = roc_auc_score(y_test, tuned_probs)
print(f"\nTuned {best_name} -> Accuracy: {round(tuned_acc*100,2)}%, AUC: {round(tuned_auc,3)}")
print(classification_report(y_test, tuned_preds))

# Use tuned model as final model (it's usually equal or better than the base one)
final_model = tuned_model
final_preds = tuned_preds
final_probs = tuned_probs


# STEP 8: Confusion Matrix (saved as image, not shown as popup)
cm = confusion_matrix(y_test, final_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_name} (tuned)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()


# STEP 9: ROC Curve (saved as image)
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

print("\nConfusion matrix saved as confusion_matrix.png")
print("ROC curve saved as roc_curve.png")


# STEP 10: Save everything needed for prediction later     
joblib.dump(final_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(list(X.columns), "columns.pkl")
joblib.dump(best_name, "best_model_name.pkl")
print("\nModel, scaler, encoders saved! you can now run predict.py")