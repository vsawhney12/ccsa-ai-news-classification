"""
Evaluate Llama predictions against expert consensus.

Steps:
1. Read labels from three expert reviewers.
2. Build the expert consensus reference standard.
3. Read Llama predictions.
4. Calculate classification metrics using scikit-learn.
5. Save raw results and an article-level audit.

Expert consensus:
- Only articles reviewed by all three experts are evaluated.
- 3/3 agreement = Unanimous consensus.
- 2/3 agreement = Majority consensus.

Prediction handling:
- Valid predictions: Drug Alert, Public Health, Public Safety.
- Unknown, blank, and N/A predictions are excluded from classification
  metrics and reported in the coverage and article-level audit files.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

from openpyxl import load_workbook
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
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

# Spelling/capitalization variants observed in the source files.
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
    """
    Read review IDs and expert labels from one reviewer Excel file.

    The worksheet is identified by the required column names rather
    than by sheet name or position.
    """

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

            # Ignore completely empty rows.
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
    """
    Build the expert consensus reference standard.

    Only articles with three valid expert labels can receive a
    consensus classification.
    """

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
# READ LLAMA PREDICTIONS
# ==========================================================

def read_predictions(path):
    """Read the previously generated Llama predictions."""

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

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            reference,
            predictions,
            labels=LABELS,
            average=None,
            zero_division=0,
        )
    )

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

    overall = {
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

    per_category = [
        {
            "Category": label,
            "Support": int(support[i]),
            "Precision": precision[i],
            "Recall": recall[i],
            "F1": f1[i],
        }
        for i, label in enumerate(LABELS)
    ]

    matrix = confusion_matrix(
        reference,
        predictions,
        labels=LABELS,
    )

    return overall, per_category, matrix


# ==========================================================
# EVALUATE LLAMA
# ==========================================================

def evaluate_llama(consensus_records, predictions):
    """Evaluate Llama against each expert-consensus subset."""

    reviewer_ids = {
        row["review_id"]
        for row in consensus_records
    }

    if reviewer_ids != set(predictions):

        missing = reviewer_ids - set(predictions)
        extra = set(predictions) - reviewer_ids

        raise ValueError(
            "Reviewer and prediction IDs do not match. "
            f"Missing predictions: {sorted(missing, key=int)}; "
            f"extra predictions: {sorted(extra, key=int)}"
        )

    # ------------------------------------------------------
    # ARTICLE-LEVEL AUDIT
    # ------------------------------------------------------

    audit = []

    for row in consensus_records:

        prediction = predictions[row["review_id"]]

        has_consensus = row["outcome"] in SUBSETS[:2]
        valid_prediction = prediction in LABELS

        included = has_consensus and valid_prediction

        if included:
            exclusion_reason = ""

        elif not has_consensus:
            exclusion_reason = (
                "Fewer than three valid reviewer labels"
                if row["valid_reviewers"] < 3
                else "No reviewer majority"
            )

        else:
            exclusion_reason = (
                "Non-category Llama prediction: "
                f"{prediction or '(blank)'}"
            )

        audit.append({
            **row,
            "llama_prediction": prediction,
            "included_in_evaluation": included,
            "exclusion_reason": exclusion_reason,
            "correct": (
                prediction == row["consensus_category"]
                if included
                else ""
            ),
        })

    # ------------------------------------------------------
    # SUBSET RESULTS
    # ------------------------------------------------------

    overall_results = []
    category_results = []
    confusion_results = []
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

        evaluated_rows = [
            row for row in reference_rows
            if row["included_in_evaluation"]
        ]

        excluded_rows = [
            row for row in reference_rows
            if not row["included_in_evaluation"]
        ]

        reference = [
            row["consensus_category"]
            for row in evaluated_rows
        ]

        predicted = [
            row["llama_prediction"]
            for row in evaluated_rows
        ]

        if not evaluated_rows:
            raise ValueError(
                f"No evaluable articles in {subset}."
            )

        overall, per_category, matrix = (
            calculate_metrics(
                reference,
                predicted,
            )
        )

        overall_results.append({
            "Reference set": subset,
            **overall,
        })

        for row in per_category:

            category_results.append({
                "Reference set": subset,
                **row,
            })

        for i, reference_label in enumerate(LABELS):

            confusion_results.append({
                "Reference set": subset,
                "Reference category": reference_label,
                **{
                    f"Predicted {label}":
                        int(matrix[i, j])
                    for j, label in enumerate(LABELS)
                },
            })

        coverage_results.append({
            "Reference set": subset,
            "Reference articles": len(reference_rows),
            "Evaluated articles": len(evaluated_rows),
            "Excluded predictions": len(excluded_rows),
            "Prediction coverage": (
                len(evaluated_rows)
                / len(reference_rows)
            ),
            "Excluded review IDs": "; ".join(
                row["review_id"]
                for row in excluded_rows
            ),
        })

    return (
        overall_results,
        category_results,
        confusion_results,
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
    overall,
    per_category,
    confusion,
    coverage,
    audit,
):
    """Save the five raw evaluation outputs."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "overall_metrics.csv": overall,
        "per_category_metrics.csv": per_category,
        "confusion_matrices.csv": confusion,
        "evaluation_coverage.csv": coverage,
        "llama_article_level_audit.csv": audit,
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
        "--llama",
        type=Path,
        default=(
            DATA_DIR
            / "ArticlesV2_Blind_review_llama.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "llama_results",
    )

    args = parser.parse_args()

    try:

        reviewers = [
            read_reviewer(path)
            for path in args.reviewers
        ]

        consensus = build_consensus(reviewers)

        predictions = read_predictions(args.llama)

        results = evaluate_llama(
            consensus,
            predictions,
        )

        save_results(
            args.output_dir,
            *results,
        )

        print(
            f"Evaluation complete. "
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