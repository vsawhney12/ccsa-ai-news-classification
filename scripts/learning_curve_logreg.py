import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef
)

# ==========================================================
# LOAD DATA
# ==========================================================

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    PROJECT_ROOT / "data" / "ArticlesV2_Blind_review_qwen.csv"
)

df = df[
    df["reviewed_primary_category"] != "Unknown"
].copy()

X = df["article_text"].astype(str)
y = df["reviewed_primary_category"].astype(str)

# ==========================================================
# SETTINGS
# ==========================================================

train_sizes = [
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    900
]

n_repeats = 20

results = []

# ==========================================================
# REPEATED LEARNING CURVE
# ==========================================================

for train_size in train_sizes:

    accuracy_scores = []
    macro_f1_scores = []
    weighted_f1_scores = []
    mcc_scores = []

    for seed in range(n_repeats):

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_size,
            stratify=y,
            random_state=seed
        )

        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    stop_words="english"
                )
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed
                )
            )
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        accuracy_scores.append(
            accuracy_score(y_test, y_pred)
        )

        macro_f1_scores.append(
            f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0
            )
        )

        weighted_f1_scores.append(
            f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )
        )

        mcc_scores.append(
            matthews_corrcoef(y_test, y_pred)
        )

    results.append({
        "Training": train_size,
        "Testing": len(X) - train_size,

        "Accuracy Mean": np.mean(accuracy_scores),
        "Accuracy SD": np.std(accuracy_scores, ddof=1),

        "Macro F1 Mean": np.mean(macro_f1_scores),
        "Macro F1 SD": np.std(macro_f1_scores, ddof=1),

        "Weighted F1 Mean": np.mean(weighted_f1_scores),
        "Weighted F1 SD": np.std(weighted_f1_scores, ddof=1),

        "MCC Mean": np.mean(mcc_scores),
        "MCC SD": np.std(mcc_scores, ddof=1)
    })

# ==========================================================
# OUTPUT
# ==========================================================

import matplotlib.pyplot as plt
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "learning_curve"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ----------------------------------------------------------
# SAVE CSV
# ----------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n====================================")
print("REPEATED LEARNING CURVE")
print("====================================\n")

print(results_df.round(3).to_string(index=False))

csv_path = OUTPUT_DIR / "learning_curve_repeated_results.csv"

results_df.to_csv(
    csv_path,
    index=False
)

print(f"\nSaved: {csv_path}")

# ==========================================================
# LEARNING CURVE VISUALIZATION
# ==========================================================


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
})

fig, axes = plt.subplots(
    2, 1,
    figsize=(8, 7),
    sharex=True,
    gridspec_kw={"hspace": 0.18}
)

x = results_df["Training"]

# ==========================================================
# PANEL A: ACCURACY
# ==========================================================

ax = axes[0]

ax.errorbar(
    x,
    results_df["Accuracy Mean"],
    yerr=results_df["Accuracy SD"],
    fmt="o-",
    color="#164A7B",
    linewidth=2,
    markersize=5,
    capsize=3,
    elinewidth=1,
    label="Mean ± SD",
    zorder=3,
)

ax.axvspan(
    600, 700,
    color="#DCEAF4",
    alpha=0.55,
    zorder=0,
)

ax.set_ylabel("Accuracy")
ax.set_ylim(0.78, 0.96)
ax.yaxis.set_major_locator(MultipleLocator(0.04))

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.25,
)

ax.text(
    0.02, 0.95,
    "A  |  Classification accuracy",
    transform=ax.transAxes,
    va="top",
    fontweight="bold",
)

ax.legend(
    loc="lower right",
    frameon=False,
    fontsize=9,
)

# ==========================================================
# PANEL B: MCC
# ==========================================================

ax = axes[1]

ax.errorbar(
    x,
    results_df["MCC Mean"],
    yerr=results_df["MCC SD"],
    fmt="s-",
    color="#A43B36",
    linewidth=2,
    markersize=5,
    capsize=3,
    elinewidth=1,
    label="Mean ± SD",
    zorder=3,
)

ax.axvspan(
    600, 700,
    color="#DCEAF4",
    alpha=0.55,
    zorder=0,
)

ax.set_ylabel("Matthews correlation coefficient")
ax.set_xlabel("Number of training articles")
ax.set_ylim(0.63, 0.91)
ax.yaxis.set_major_locator(MultipleLocator(0.05))

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.25,
)

ax.text(
    0.02, 0.95,
    "B  |  Matthews correlation coefficient",
    transform=ax.transAxes,
    va="top",
    fontweight="bold",
)

# ==========================================================
# AXIS FORMATTING
# ==========================================================

axes[1].set_xticks(results_df["Training"])
axes[1].set_xlim(75, 925)

for ax in axes:
    ax.tick_params(
        axis="both",
        direction="out",
        length=4,
        width=0.8,
    )

# ==========================================================
# SAVE FIGURE
# ==========================================================

plt.savefig(
    OUTPUT_DIR / "learning_curve.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print(
    f"Learning curve saved: "
    f"{OUTPUT_DIR / 'learning_curve.png'}"
)