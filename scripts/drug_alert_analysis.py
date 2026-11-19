
#!/usr/bin/env python3
"""Reproduce the Drug Alert qualitative error analysis.

Uses archived predictions and article texts.
Does not rerun any language models.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
audit_file = (
    ROOT / "results" / "model_comparison_expanded"
    / "comparison_article_level_audit.csv"
)

articles_file = ROOT / "data" / "ArticlesV2_Blind_review_qwen.csv"

audit = pd.read_csv(audit_file)
articles = pd.read_csv(articles_file)

eligible = audit.loc[
    audit["included_expanded"].eq(True)
    & audit["shared_valid_prediction"].eq(True)
].copy()

assert len(eligible) == 574, (
    f"Expected 574 articles, found {len(eligible)}"
)

models = ["Qwen", "Llama", "Mistral"]

for model in models:
    eligible[f"{model}_false_positive"] = (
        eligible[f"{model}_prediction"].eq("Drug Alert")
        & eligible["consensus_category"].ne("Drug Alert")
    )

eligible["false_positive_models"] = eligible.apply(
    lambda row: ", ".join(
        model for model in models
        if row[f"{model}_false_positive"]
    ),
    axis=1
)

# Keep articles flagged by at least one model.
false_positives = eligible.loc[
    eligible["false_positive_models"].ne("")
].copy()

false_positives = false_positives.merge(
    articles[["review_id", "article_text"]],
    on="review_id",
    validate="one_to_one"
)

# Descriptive qualitative categories assigned during
# the article-level review.
overdose_warning = {
    23, 75, 121, 143, 219, 249, 279, 316,
    329, 417, 429, 515, 629, 648, 656, 658
}

product_advisory = {
    37, 130, 241, 290, 618
}

enforcement = {
    8, 36, 38, 60, 66, 74, 133, 225,
    242, 276, 291, 292, 310, 336, 416,
    475, 505, 525, 569, 593, 604, 608, 628
}

other = {
    126, 255, 320, 333, 406, 446
}

def assign_theme(article_id):
    if article_id in overdose_warning:
        return "Overdose, toxicity or drug-risk reporting"
    if article_id in product_advisory:
        return "Unauthorized health-product advisory"
    if article_id in enforcement:
        return "Enforcement, seizure or legal reporting"
    if article_id in other:
        return "Other or mixed-context reporting"
    raise ValueError(
        f"Article {article_id} has no qualitative theme"
    )

false_positives["qualitative_theme"] = (
    false_positives["review_id"].map(assign_theme)
)

# Preserve exactly the text window supplied to the models.
false_positives["model_input_excerpt"] = (
    false_positives["article_text"]
    .fillna("")
    .str[:1200]
)

columns = [
    "review_id",
    "consensus_category",
    "consensus_basis",
    "Qwen_prediction",
    "Llama_prediction",
    "Mistral_prediction",
    "false_positive_models",
    "qualitative_theme",
    "model_input_excerpt"
]

output = RESULTS / "drug_alert_error_analysis.csv"

false_positives[columns].sort_values(
    "review_id"
).to_csv(output, index=False)

# Validate results against the manuscript.
assert len(false_positives) == 50

expected = {
    "Qwen": 28,
    "Llama": 14,
    "Mistral": 31
}

for model, count in expected.items():
    actual = int(
        false_positives[f"{model}_false_positive"].sum()
    )
    assert actual == count, (
        f"{model}: expected {count}, found {actual}"
    )

print("\nDRUG ALERT ERROR ANALYSIS")
print("-" * 35)
print(f"Shared evaluation: {len(eligible)} articles")
print(f"Unique false positives: {len(false_positives)}")

for model in models:
    count = false_positives[
        f"{model}_false_positive"
    ].sum()
    print(f"{model}: {count} false positives")

print("\nReference categories:")
print(false_positives["consensus_category"].value_counts())

print("\nQualitative themes:")
print(false_positives["qualitative_theme"].value_counts())

print(f"\nCSV saved to: {output}")
