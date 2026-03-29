# %% [markdown]
# # Notebook 1 - Data Analysis & Preprocessing
# **Student Stress & Burnout Predictor**
#
# This notebook covers everything before the model:
# loading the data, exploring it, cleaning it, and saving
# a ready-to-use train/test split for notebook 2.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

os.makedirs("../data", exist_ok=True)
os.makedirs("../models", exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
print("imports done")

# %% [markdown]
# ## 1. Load Data
#
# Survey was collected via Google Form. Exported as CSV and column names
# were shortened manually before loading.

# %%
df = pd.read_csv("../data/survey_responses.csv")
print(f"{len(df)} responses, {df.shape[1]} columns")
df.head()

# %%
df.info()

# %%
df.isnull().sum()

# %% [markdown]
# ## 2. Class Distribution
#
# Checking if the three burnout levels are reasonably balanced
# before doing anything else.

# %%
label_order = ["Low", "Medium", "High"]
colors = ["#4CAF50", "#FF9800", "#F44336"]
counts = df["burnout_level"].value_counts().reindex(label_order)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", width=0.5)
for i, (v, p) in enumerate(zip(counts.values, counts.values / len(df) * 100)):
    axes[0].text(i, v + 0.5, f"{p:.0f}%", ha="center", fontweight="bold")
axes[0].set_title("Students per burnout level")
axes[0].set_ylabel("Count")

axes[1].pie(counts.values, labels=counts.index, colors=colors,
            autopct="%1.1f%%", startangle=140, wedgeprops={"edgecolor": "white"})
axes[1].set_title("Proportion")

plt.suptitle("Burnout level distribution", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Feature Distributions

# %%
FEATURES = [
    "sleep_hours", "study_hours", "social_activity",
    "physical_activity", "cgpa", "attendance_pct",
    "academic_pressure", "assignment_backlog", "mood_rating"
]

fig, axes = plt.subplots(3, 3, figsize=(13, 9))
axes = axes.flatten()

for i, col in enumerate(FEATURES):
    axes[i].hist(df[col], bins=12, color="#2E75B6", edgecolor="white", alpha=0.8)
    axes[i].axvline(df[col].mean(), color="red", linestyle="--",
                    linewidth=1.2, label=f"mean={df[col].mean():.1f}")
    axes[i].set_title(col)
    axes[i].legend(fontsize=8)

plt.suptitle("Feature distributions", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Boxplots by Burnout Level
#
# Most useful EDA chart — features with clearly separated boxes
# across Low/Medium/High will be the strongest predictors.

# %%
palette = {"Low": "#4CAF50", "Medium": "#FF9800", "High": "#F44336"}

fig, axes = plt.subplots(3, 3, figsize=(13, 9))
axes = axes.flatten()

for i, col in enumerate(FEATURES):
    data = [df[df["burnout_level"] == lbl][col].dropna().values for lbl in label_order]
    bp = axes[i].boxplot(data, labels=label_order, patch_artist=True,
                         medianprops={"color": "black", "linewidth": 2})
    for patch, lbl in zip(bp["boxes"], label_order):
        patch.set_facecolor(palette[lbl])
        patch.set_alpha(0.7)
    axes[i].set_title(col)

plt.suptitle("Features split by burnout level", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# Sleep hours, academic pressure, CGPA and mood show the clearest
# class separation — these will likely be the top predictors.

# %% [markdown]
# ## 5. Correlation Heatmap

# %%
corr = df[FEATURES].corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, ax=ax, mask=np.triu(np.ones_like(corr, dtype=bool)))
ax.set_title("Feature correlations", fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# Nothing dangerously correlated — keeping all features.

# %% [markdown]
# ## 6. Cleaning & Imputation

# %%
imputer = SimpleImputer(strategy="median")
df[FEATURES] = imputer.fit_transform(df[FEATURES])
df = df.dropna(subset=["burnout_level"]).reset_index(drop=True)
print(f"After cleaning: {len(df)} rows")

# cap outliers using IQR
def cap_outliers(s, factor=1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return s.clip(lower=q1 - factor * iqr, upper=q3 + factor * iqr)

for col in FEATURES:
    df[col] = cap_outliers(df[col])

print("Outlier capping done.")

# %% [markdown]
# ## 7. Encode Target & Engineer Features

# %%
LABEL_MAP = {"Low": 0, "Medium": 1, "High": 2}
df["burnout_encoded"] = df["burnout_level"].map(LABEL_MAP)

# stress_load: high study + pressure relative to sleep = bad sign
# lifestyle_score: average of protective factors
df["stress_load"]     = (df["study_hours"] + df["academic_pressure"]) / (df["sleep_hours"] + 1)
df["lifestyle_score"] = (df["mood_rating"] + df["social_activity"] + df["physical_activity"]) / 3

FINAL_FEATURES = FEATURES + ["stress_load", "lifestyle_score"]
print(f"Final feature count: {len(FINAL_FEATURES)}")

# %% [markdown]
# ## 8. Train / Test Split & Scaling

# %%
X = df[FINAL_FEATURES]
y = df["burnout_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=FINAL_FEATURES)
X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=FINAL_FEATURES)

# %%
# save everything
X_train_sc.to_csv("../data/X_train.csv", index=False)
X_test_sc.to_csv("../data/X_test.csv",   index=False)
y_train.to_csv("../data/y_train.csv",     index=False)
y_test.to_csv("../data/y_test.csv",       index=False)

joblib.dump(scaler,         "../models/scaler.pkl")
joblib.dump(FINAL_FEATURES, "../models/feature_names.pkl")
joblib.dump(LABEL_MAP,      "../models/label_map.pkl")

print("Saved splits and scaler. Ready for notebook 2.")
