# %% [markdown]
# # Notebook 2 - Model Training, Evaluation & Prediction
# **Student Stress & Burnout Predictor**
#
# Trains three classifiers, compares them with cross-validation,
# evaluates the best one on the test set, and includes a predict
# function you can call directly from this notebook.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

LABEL_NAMES = ["Low", "Medium", "High"]
SEED = 42

# %%
X_train = pd.read_csv("../data/X_train.csv")
X_test  = pd.read_csv("../data/X_test.csv")
y_train = pd.read_csv("../data/y_train.csv").squeeze()
y_test  = pd.read_csv("../data/y_test.csv").squeeze()
FEATURES = joblib.load("../models/feature_names.pkl")

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

# %% [markdown]
# ## 1. Define Models

# %%
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=SEED),
    "KNN":                 KNeighborsClassifier(n_neighbors=7, weights="distance"),
}

# %% [markdown]
# ## 2. Cross-Validation
#
# 5-fold stratified CV. Using weighted F1 alongside accuracy
# to account for any class imbalance.

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_results = {}

print(f"{'Model':<22} {'Accuracy':>12}  {'F1 weighted':>13}")
print("-" * 50)

for name, model in models.items():
    scores = cross_validate(model, X_train, y_train, cv=cv,
                            scoring=["accuracy", "f1_weighted"])
    cv_results[name] = {
        "acc": scores["test_accuracy"],
        "f1":  scores["test_f1_weighted"],
    }
    acc = scores["test_accuracy"]
    f1  = scores["test_f1_weighted"]
    print(f"{name:<22} {acc.mean():.4f} ± {acc.std():.4f}   {f1.mean():.4f} ± {f1.std():.4f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
model_names = list(cv_results.keys())
bar_colors  = ["#2E75B6", "#E74C3C", "#27AE60"]

for ax, metric, title in zip(axes, ["acc", "f1"], ["CV Accuracy", "CV F1 (weighted)"]):
    means = [cv_results[m][metric].mean() for m in model_names]
    stds  = [cv_results[m][metric].std()  for m in model_names]
    bars  = ax.bar(model_names, means, yerr=stds, capsize=5,
                   color=bar_colors, edgecolor="white", alpha=0.85, width=0.45)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_title(title)
    ax.set_xticklabels(model_names, rotation=10, ha="right")

plt.suptitle("Cross-validation comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Train Best Model & Test Set Evaluation
#
# Random Forest wins CV — training it on the full training set
# and evaluating on the held-out test set.

# %%
best_model = RandomForestClassifier(n_estimators=200, random_state=SEED)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print(f"Test Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Test F1 (wtd) : {f1_score(y_test, y_pred, average='weighted'):.4f}\n")
print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

# %%
# confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                       display_labels=LABEL_NAMES).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix — Random Forest", fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Feature Importance

# %%
feat_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": best_model.feature_importances_
}).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(7, 6))
colors = ["#2E75B6" if v > feat_df["Importance"].median() else "#A8C8E8"
          for v in feat_df["Importance"]]
ax.barh(feat_df["Feature"], feat_df["Importance"],
        color=colors, edgecolor="white", height=0.6)
ax.axvline(feat_df["Importance"].mean(), color="red", linestyle="--",
           linewidth=1.2, label="mean")
ax.set_title("Feature Importance — Random Forest", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.show()

print("Top 5 features:")
print(feat_df.sort_values("Importance", ascending=False).head(5).to_string(index=False))

# %% [markdown]
# ## 5. Save Model

# %%
joblib.dump(best_model, "../models/burnout_classifier.pkl")
print("Model saved.")

# %% [markdown]
# ## 6. Predict on New Student Input
#
# Run the cell below to predict burnout risk for any student.
# Just update the values in the `student` dictionary.

# %%
def predict_burnout(student_data):
    model   = joblib.load("../models/burnout_classifier.pkl")
    scaler  = joblib.load("../models/scaler.pkl")
    features = joblib.load("../models/feature_names.pkl")

    # compute engineered features
    student_data["stress_load"] = (
        (student_data["study_hours"] + student_data["academic_pressure"])
        / (student_data["sleep_hours"] + 1)
    )
    student_data["lifestyle_score"] = (
        (student_data["mood_rating"] + student_data["social_activity"]
         + student_data["physical_activity"]) / 3
    )

    X = np.array([[student_data[f] for f in features]])
    X_sc = scaler.transform(X)
    pred  = model.predict(X_sc)[0]
    proba = model.predict_proba(X_sc)[0]

    labels = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    advice = {
        "LOW":    "Lifestyle indicators look healthy — keep it up.",
        "MEDIUM": "Some stress signs detected. Review your schedule and take breaks.",
        "HIGH":   "High burnout risk. Please talk to a counsellor or trusted person.",
    }

    result = labels[pred]
    print(f"\nPredicted Burnout Risk: {result}")
    print(f"Advice: {advice[result]}")
    print("\nConfidence:")
    for i, p in enumerate(proba):
        bar = "█" * int(p * 25)
        print(f"  {labels[i]:>8}: {bar:<25} {p*100:.1f}%")

    return result


# %%
# --- edit these values and run the cell ---

student = {
    "sleep_hours":        5.5,
    "study_hours":        8.0,
    "social_activity":    2.0,
    "physical_activity":  1.0,
    "cgpa":               7.2,
    "attendance_pct":     72.0,
    "academic_pressure":  8.0,
    "assignment_backlog": 5.0,
    "mood_rating":        4.0,
}

predict_burnout(student)
