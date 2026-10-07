"""
Compare Qwen, Llama, and Mistral predictions against expert consensus.

Steps:
1. Read labels from three expert reviewers.
2. Build the expert consensus reference standard.
3. Read predictions from all three models.
4. Keep only articles with valid predictions from all three models.
5. Calculate model metrics on the same shared sample.
6. Compare pairwise model correctness.
7. Save raw comparison results.

Expert consensus:
- Only articles reviewed by all three experts are evaluated.
- 3/3 agreement = Unanimous consensus.
- 2/3 agreement = Majority consensus.

Shared-sample rule:
- An article is included only when all three models produced a valid
  category prediction.
- This ensures that Qwen, Llama, and Mistral are compared on exactly
  the same articles.
"""

import argparse
import csv
import itertools
from collections import Counter
from pathlib import Path

from openpyxl import load_workbook
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


# ==========================================================
# SETTINGS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


LABELS = (
    "Drug Alert",
    "Public Health",
    "Public Safety",
)

SUBSETS = (
    "Unanimous (3/3)",
    "Majority (2/3)",
    "Combined consensus",
)

MODELS = (
    "Qwen",
    "Llama",
    "Mistral",
)

LABEL_NORMALIZATION = {
    "Public healh": "Public Health",
    "Drug alert": "Drug Alert",
}


# ==========================================================
# DATA CLEANING
# ==========================================================

def normalize_label(value):
    """Standardize known spelling/capitalization variants."""

    if value is None:
        return ""

    value = str(value).strip()
    return LABEL_NORMALIZATION.get(value, value)


def normalize_id(value):
    """Convert review IDs to a consistent string format."""

    if value is None:
        raise ValueError("Missing review_id")

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid review_id: {value!r}"
        ) from error

    if not number.is_integer():
        raise ValueError(
            f"review_id must be an integer: {value!r}"
        )

    return str(int(number))


# ==========================================================
# READ EXPERT LABELS
# ==========================================================

def read_reviewer(path):
    """Read review IDs and expert labels from one Excel file."""

    workbook = load_workbook(
        path,
        read_only=True,
        data_only=True,
    )

    matching_sheets = []

    for worksheet in workbook.worksheets:

        rows = worksheet.iter_rows(values_only=True)

        try:
            headers = next(rows)
        except StopIteration:
            continue

        headers = [
            str(value).strip() if value is not None else ""
            for value in headers
        ]

        required = {
            "review_id",
            "reviewed_primary_category",
        }

        if not required <= set(headers):
            continue

        id_column = headers.index("review_id")
        label_column = headers.index(
            "reviewed_primary_category"
        )

        records = {}

        for row_number, row in enumerate(rows, start=2):

            raw_id = row[id_column]
            raw_label = row[label_column]

            if raw_id is None and all(
                value is None for value in row
            ):
                continue

            if raw_id is None:
                raise ValueError(
                    f"{path}: row {row_number} has no review_id"
                )

            review_id = normalize_id(raw_id)
            label = normalize_label(raw_label)

            if label not in (*LABELS, "", "N/A"):
                raise ValueError(
                    f"{path}: unexpected category "
                    f"{label!r}, ID {review_id}"
                )

            if review_id in records:
                raise ValueError(
                    f"{path}: duplicate review_id {review_id}"
                )

            records[review_id] = label

        matching_sheets.append(records)

    workbook.close()

    if len(matching_sheets) != 1:
        raise ValueError(
            f"{path}: expected exactly one worksheet "
            "containing the required columns."
        )

    return matching_sheets[0]


# ==========================================================
# BUILD EXPERT CONSENSUS
# ==========================================================

def build_consensus(reviewers):
    """Build unanimous and majority expert consensus."""

    id_sets = [set(reviewer) for reviewer in reviewers]

    if (
        not id_sets[0]
        or any(ids != id_sets[0] for ids in id_sets[1:])
    ):
        raise ValueError(
            "The three reviewer files must contain "
            "the same review IDs."
        )

    records = []

    for review_id in sorted(id_sets[0], key=int):

        labels = [
            reviewer[review_id]
            for reviewer in reviewers
        ]

        valid_labels = [
            label for label in labels
            if label in LABELS
        ]

        outcome = "Incomplete"
        consensus = ""

        if len(valid_labels) == 3:

            counts = Counter(valid_labels)
            consensus, votes = counts.most_common(1)[0]

            if votes == 3:
                outcome = "Unanimous (3/3)"

            elif votes == 2:
                outcome = "Majority (2/3)"

            else:
                outcome = "All three different"
                consensus = ""

        records.append({
            "review_id": review_id,
            "R1": labels[0],
            "R2": labels[1],
            "R3": labels[2],
            "valid_reviewers": len(valid_labels),
            "outcome": outcome,
            "consensus_category": consensus,
        })

    return records


# ==========================================================
# READ MODEL PREDICTIONS
# ==========================================================

def read_predictions(path):
    """Read saved model predictions."""

    records = {}

    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as file:

        reader = csv.DictReader(file)

        required = {
            "review_id",
            "reviewed_primary_category",
        }

        if (
            reader.fieldnames is None
            or not required <= set(reader.fieldnames)
        ):
            raise ValueError(
                f"{path}: required columns are missing."
            )

        for row_number, row in enumerate(reader, start=2):

            review_id = normalize_id(
                row["review_id"]
            )

            prediction = normalize_label(
                row["reviewed_primary_category"]
            )

            if prediction not in (
                *LABELS,
                "",
                "N/A",
                "Unknown",
            ):
                raise ValueError(
                    f"{path}: unexpected prediction "
                    f"{prediction!r}, ID {review_id}"
                )

            if review_id in records:
                raise ValueError(
                    f"{path}: duplicate review_id {review_id}"
                )

            records[review_id] = prediction

    return records


# ==========================================================
# CALCULATE METRICS
# ==========================================================

def calculate_metrics(reference, predictions):
    """Calculate classification metrics using scikit-learn."""

    macro_p, macro_r, macro_f1, _ = (
        precision_recall_fscore_support(
            reference,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
    )

    _, _, weighted_f1, _ = (
        precision_recall_fscore_support(
            reference,
            predictions,
            labels=LABELS,
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "N": len(reference),
        "Accuracy": accuracy_score(
            reference,
            predictions,
        ),
        "Macro P": macro_p,
        "Macro R": macro_r,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Cohen's kappa": cohen_kappa_score(
            reference,
            predictions,
            labels=LABELS,
        ),
        "MCC": matthews_corrcoef(
            reference,
            predictions,
        ),
    }


# ==========================================================
# COMPARE MODELS
# ==========================================================

def compare_models(consensus_records, predictions):
    """Compare all models on one shared evaluable sample."""

    reviewer_ids = {
        row["review_id"]
        for row in consensus_records
    }

    for model in MODELS:

        if set(predictions[model]) != reviewer_ids:

            missing = reviewer_ids - set(predictions[model])
            extra = set(predictions[model]) - reviewer_ids

            raise ValueError(
                f"{model}: prediction IDs do not match reviewer IDs. "
                f"Missing: {sorted(missing, key=int)}; "
                f"extra: {sorted(extra, key=int)}"
            )

    # ------------------------------------------------------
    # ARTICLE-LEVEL AUDIT
    # ------------------------------------------------------

    audit = []

    for original in consensus_records:

        row = dict(original)
        invalid_models = []

        for model in MODELS:

            prediction = predictions[model][
                row["review_id"]
            ]

            row[f"{model}_prediction"] = prediction

            if prediction not in LABELS:
                invalid_models.append(model)

        has_consensus = (
            row["outcome"] in SUBSETS[:2]
        )

        row["included_in_comparison"] = (
            has_consensus and not invalid_models
        )

        if not has_consensus:

            row["exclusion_reason"] = (
                "No complete three-reviewer consensus"
            )

        elif invalid_models:

            row["exclusion_reason"] = (
                "Non-category prediction: "
                + ", ".join(invalid_models)
            )

        else:

            row["exclusion_reason"] = ""

        for model in MODELS:

            row[f"{model}_correct"] = (
                row[f"{model}_prediction"]
                == row["consensus_category"]
                if row["included_in_comparison"]
                else ""
            )

        audit.append(row)

    # ------------------------------------------------------
    # MODEL COMPARISON
    # ------------------------------------------------------

    model_results = []
    pairwise_results = []
    coverage_results = []

    for subset in SUBSETS:

        if subset == "Combined consensus":

            reference_rows = [
                row for row in audit
                if row["outcome"] in SUBSETS[:2]
            ]

        else:

            reference_rows = [
                row for row in audit
                if row["outcome"] == subset
            ]

        shared_rows = [
            row for row in reference_rows
            if row["included_in_comparison"]
        ]

        excluded_rows = [
            row for row in reference_rows
            if not row["included_in_comparison"]
        ]

        coverage_results.append({
            "Reference set": subset,
            "Reference N": len(reference_rows),
            "Shared N": len(shared_rows),
            "Excluded predictions": len(excluded_rows),
            "Shared coverage": (
                len(shared_rows) / len(reference_rows)
            ),
            "Excluded review IDs": "; ".join(
                row["review_id"]
                for row in excluded_rows
            ),
        })

        reference = [
            row["consensus_category"]
            for row in shared_rows
        ]

        if not shared_rows:
            raise ValueError(
                f"No shared evaluable articles in {subset}."
            )

        # Model metrics
        for model in MODELS:

            predicted = [
                row[f"{model}_prediction"]
                for row in shared_rows
            ]

            metrics = calculate_metrics(
                reference,
                predicted,
            )

            model_results.append({
                "Reference set": subset,
                "Model": model,
                **metrics,
            })

        # Pairwise correctness
        for model_a, model_b in itertools.combinations(
            MODELS,
            2,
        ):

            counts = Counter(
                (
                    row[f"{model_a}_correct"],
                    row[f"{model_b}_correct"],
                )
                for row in shared_rows
            )

            pairwise_results.append({
                "Reference set": subset,
                "Model A": model_a,
                "Model B": model_b,
                "Both correct": counts[(True, True)],
                "A only correct": counts[(True, False)],
                "B only correct": counts[(False, True)],
                "Both incorrect": counts[(False, False)],
            })

    return (
        model_results,
        pairwise_results,
        coverage_results,
        audit,
    )


# ==========================================================
# SAVE RESULTS
# ==========================================================

def write_csv(path, rows):
    """Write a list of dictionaries to CSV."""

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


def save_results(
    output_dir,
    model_results,
    pairwise_results,
    coverage,
    audit,
):
    """Save raw model-comparison outputs."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "model_comparison.csv": model_results,
        "pairwise_correctness.csv": pairwise_results,
        "comparison_coverage.csv": coverage,
        "comparison_article_level_audit.csv": audit,
    }

    for filename, rows in outputs.items():

        write_csv(
            output_dir / filename,
            rows,
        )


# ==========================================================
# MAIN
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--reviewers",
        nargs=3,
        type=Path,
        default=[
            DATA_DIR
            / (
                f"Reviewer {i} - "
                "Articles and Labels for review.xlsx"
            )
            for i in range(1, 4)
        ],
    )

    parser.add_argument(
        "--qwen",
        type=Path,
        default=(
            DATA_DIR
            / "ArticlesV2_Blind_review_qwen.csv"
        ),
    )

    parser.add_argument(
        "--llama",
        type=Path,
        default=(
            DATA_DIR
            / "ArticlesV2_Blind_review_llama.csv"
        ),
    )

    parser.add_argument(
        "--mistral",
        type=Path,
        default=(
            DATA_DIR
            / "ArticlesV2_Blind_review_mistral.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            RESULTS_DIR
            / "model_comparison_results"
        ),
    )

    args = parser.parse_args()

    try:

        reviewers = [
            read_reviewer(path)
            for path in args.reviewers
        ]

        consensus = build_consensus(reviewers)

        predictions = {
            "Qwen": read_predictions(args.qwen),
            "Llama": read_predictions(args.llama),
            "Mistral": read_predictions(args.mistral),
        }

        results = compare_models(
            consensus,
            predictions,
        )

        save_results(
            args.output_dir,
            *results,
        )

        print(
            f"Comparison complete. "
            f"Results saved to {args.output_dir}"
        )

    except (
        ValueError,
        OSError,
        csv.Error,
    ) as error:

        parser.exit(
            1,
            f"Error: {error}\n",
        )


if __name__ == "__main__":
    main()