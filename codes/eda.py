# %% [markdown]
# # Notebook 1 - Exploratory Data Analysis
# **Student Stress & Burnout Predictor**
#
# So this is the first notebook. Goal here is just to understand the data
# before doing anything fancy with it. I collected responses from classmates
# using a Google Form and exported it as a CSV.
#
# Let's see what we're working with.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

print("imports done")

# %% [markdown]
# ## Loading the data
#
# Exported the Google Form responses as a CSV and renamed the columns
# to shorter names since the original question text was way too long.

# %%
df = pd.read_csv("../data/survey_responses.csv")
print(f"Got {len(df)} responses, {df.shape[1]} columns")
df.head()

# %%
df.info()

# %%
# check for missing values
df.isnull().sum()

# %% [markdown]
# ## Class distribution
#
# First thing I wanted to check — are the three burnout levels roughly balanced?
# If one class has way fewer samples it could mess with the model later.

# %%
label_order = ["Low", "Medium", "High"]
colors = ["#4CAF50", "#FF9800", "#F44336"]
counts = df["burnout_level"].value_counts().reindex(label_order)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", width=0.5)
for i, (val, pct) in enumerate(zip(counts.values, counts.values / len(df) * 100)):
    axes[0].text(i, val + 0.5, f"{pct:.0f}%", ha="center", fontweight="bold")
axes[0].set_title("Students per burnout level")
axes[0].set_ylabel("Count")

axes[1].pie(counts.values, labels=counts.index, colors=colors,
            autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor": "white"})
axes[1].set_title("Proportion")

plt.suptitle("Burnout level distribution", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../data/fig_class_dist.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# Looks reasonably balanced. Medium has the most responses which makes sense —
# most students are somewhere in the middle stress-wise.

# %% [markdown]
# ## Feature distributions

# %%
features = [
    "sleep_hours", "study_hours", "social_activity",
    "physical_activity", "cgpa", "attendance_pct",
    "academic_pressure", "assignment_backlog", "mood_rating"
]

fig, axes = plt.subplots(3, 3, figsize=(13, 9))
axes = axes.flatten()

for i, col in enumerate(features):
    axes[i].hist(df[col], bins=12, color="#2E75B6", edgecolor="white", alpha=0.8)
    axes[i].axvline(df[col].mean(), color="red", linestyle="--", linewidth=1.2,
                    label=f"mean={df[col].mean():.1f}")
    axes[i].set_title(col)
    axes[i].legend(fontsize=8)

plt.suptitle("Distribution of all features", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../data/fig_distributions.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Boxplots by burnout level
#
# This is the most useful chart here. If a feature's boxes are clearly separated
# across Low/Medium/High, it'll be a good predictor for the model.

# %%
fig, axes = plt.subplots(3, 3, figsize=(13, 9))
axes = axes.flatten()
palette = {"Low": "#4CAF50", "Medium": "#FF9800", "High": "#F44336"}

for i, col in enumerate(features):
    data = [df[df["burnout_level"] == lbl][col].dropna().values for lbl in label_order]
    bp = axes[i].boxplot(data, labels=label_order, patch_artist=True,
                         medianprops={"color": "black", "linewidth": 2})
    for patch, lbl in zip(bp["boxes"], label_order):
        patch.set_facecolor(palette[lbl])
        patch.set_alpha(0.7)
    axes[i].set_title(col)

plt.suptitle("Features split by burnout level", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../data/fig_boxplots.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# Sleep hours, academic pressure, CGPA and mood seem to separate the classes
# most clearly. Will prioritise these during analysis.

# %% [markdown]
# ## Correlation heatmap

# %%
corr = df[features].corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax,
            mask=np.triu(np.ones_like(corr, dtype=bool)))
ax.set_title("Feature correlation matrix", fontweight="bold")
plt.tight_layout()
plt.savefig("../data/fig_heatmap.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# Nothing too correlated so I'll keep all features going into preprocessing.
# Moving on to notebook 2.
