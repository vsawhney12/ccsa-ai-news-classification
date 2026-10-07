"""
Reproduce expert-reviewer consensus and reliability results.

Steps:
1. Read labels from three expert reviewers.
2. Summarize reviewer coverage and label availability.
3. Build three-reviewer consensus.
4. Calculate reviewer agreement and reliability.
5. Save raw CSV results and an article-level audit.
"""

import argparse
import csv
import itertools
from collections import Counter
from pathlib import Path

from openpyxl import load_workbook
from sklearn.metrics import cohen_kappa_score


# ==========================================================
# SETTINGS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

CATEGORIES = (
    "Public Safety",
    "Public Health",
    "Drug Alert",
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

            if label not in (*CATEGORIES, "", "N/A"):
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
# BUILD ARTICLE-LEVEL AUDIT
# ==========================================================

def build_audit(reviewers):
    """Combine the three reviewer files article by article."""

    id_sets = [set(reviewer) for reviewer in reviewers]

    if (
        not id_sets[0]
        or any(ids != id_sets[0] for ids in id_sets[1:])
    ):
        raise ValueError(
            "The three reviewer files must contain "
            "the same review IDs."
        )

    audit = []

    for review_id in sorted(id_sets[0], key=int):

        labels = [
            reviewer[review_id]
            for reviewer in reviewers
        ]

        valid_labels = [
            label
            for label in labels
            if label in CATEGORIES
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

        audit.append({
            "review_id": review_id,
            "R1": labels[0],
            "R2": labels[1],
            "R3": labels[2],
            "valid_reviewers": len(valid_labels),
            "outcome": outcome,
            "consensus_category": consensus,
        })

    return audit


# ==========================================================
# REVIEWER COVERAGE
# ==========================================================

def calculate_coverage(audit):
    """Summarize valid, blank, and N/A labels by reviewer."""

    results = []

    for reviewer_number in range(1, 4):

        field = f"R{reviewer_number}"

        counts = Counter(
            row[field]
            for row in audit
        )

        valid_count = sum(
            counts[category]
            for category in CATEGORIES
        )

        result = {
            "Reviewer": f"Reviewer {reviewer_number}",
            "Valid category": valid_count,
            "Blank": counts[""],
            "Explicit N/A": counts["N/A"],
        }

        for category in CATEGORIES:

            result[category] = counts[category]

            result[f"{category} percent"] = (
                counts[category] / valid_count
                if valid_count
                else 0.0
            )

        results.append(result)

    return results


# ==========================================================
# LABEL AVAILABILITY
# ==========================================================

def calculate_availability(audit):
    """Summarize how many reviewers supplied valid labels."""

    total = len(audit)
    results = []

    names = {
        3: "All three reviewers",
        2: "Exactly two reviewers",
        1: "Exactly one reviewer",
        0: "No valid primary category",
    }

    for number in (3, 2, 1, 0):

        count = sum(
            row["valid_reviewers"] == number
            for row in audit
        )

        results.append({
            "Availability pattern": names[number],
            "Articles": count,
            "Proportion": (
                count / total
                if total
                else 0.0
            ),
        })

    return results


# ==========================================================
# CONSENSUS SUMMARY
# ==========================================================

def calculate_consensus(audit):
    """Summarize consensus among complete three-reviewer cases."""

    complete = [
        row for row in audit
        if row["valid_reviewers"] == 3
    ]

    outcome_counts = Counter(
        row["outcome"]
        for row in complete
    )

    results = []

    for outcome in (
        "Unanimous (3/3)",
        "Majority (2/3)",
        "All three different",
    ):

        count = outcome_counts[outcome]

        results.append({
            "Agreement outcome": outcome,
            "Count": count,
            "Proportion": (
                count / len(complete)
                if complete
                else 0.0
            ),
        })

    for category in CATEGORIES:

        count = sum(
            row["consensus_category"] == category
            for row in complete
        )

        results.append({
            "Agreement outcome":
                f"Consensus category: {category}",
            "Count": count,
            "Proportion": (
                count / len(complete)
                if complete
                else 0.0
            ),
        })

    return results


# ==========================================================
# RELIABILITY
# ==========================================================

def calculate_fleiss_kappa(complete_labels):
    """Calculate Fleiss' kappa for three reviewers."""

    if not complete_labels:
        return None, None

    observed_agreement = sum(
        sum(
            count * (count - 1)
            for count in Counter(labels).values()
        ) / 6
        for labels in complete_labels
    ) / len(complete_labels)

    category_counts = Counter(
        label
        for labels in complete_labels
        for label in labels
    )

    total_labels = 3 * len(complete_labels)

    proportions = [
        category_counts[category] / total_labels
        for category in CATEGORIES
    ]

    expected_agreement = sum(
        proportion ** 2
        for proportion in proportions
    )

    if expected_agreement == 1:
        kappa = None
    else:
        kappa = (
            observed_agreement - expected_agreement
        ) / (
            1 - expected_agreement
        )

    return observed_agreement, kappa


def calculate_gwet_ac1(complete_labels):
    """Calculate unweighted Gwet's AC1."""

    if not complete_labels:
        return None

    observed_agreement = sum(
        sum(
            count * (count - 1)
            for count in Counter(labels).values()
        ) / 6
        for labels in complete_labels
    ) / len(complete_labels)

    category_counts = Counter(
        label
        for labels in complete_labels
        for label in labels
    )

    total_labels = 3 * len(complete_labels)

    proportions = [
        category_counts[category] / total_labels
        for category in CATEGORIES
    ]

    expected_agreement = sum(
        proportion * (1 - proportion)
        for proportion in proportions
    ) / (
        len(CATEGORIES) - 1
    )

    if expected_agreement == 1:
        return None

    return (
        observed_agreement - expected_agreement
    ) / (
        1 - expected_agreement
    )


def calculate_krippendorff_alpha(units):
    """Calculate nominal Krippendorff's alpha."""

    pairable = [
        labels
        for labels in units
        if len(labels) >= 2
    ]

    counts = Counter(
        label
        for labels in pairable
        for label in labels
    )

    total = sum(counts.values())

    if total < 2:
        return len(pairable), None

    observed_disagreement = sum(
        (
            len(labels) ** 2
            - sum(
                count ** 2
                for count in Counter(labels).values()
            )
        )
        / (len(labels) - 1)
        for labels in pairable
    ) / total

    expected_disagreement = (
        total ** 2
        - sum(
            count ** 2
            for count in counts.values()
        )
    ) / (
        total * (total - 1)
    )

    if expected_disagreement == 0:
        alpha = None
    else:
        alpha = (
            1
            - observed_disagreement
            / expected_disagreement
        )

    return len(pairable), alpha


def calculate_reliability(audit):
    """Calculate pairwise and multi-reviewer reliability."""

    results = []

    # Pairwise Cohen's kappa
    for left, right in itertools.combinations(
        range(1, 4),
        2,
    ):

        pairs = [
            (
                row[f"R{left}"],
                row[f"R{right}"],
            )
            for row in audit
            if (
                row[f"R{left}"] in CATEGORIES
                and row[f"R{right}"] in CATEGORIES
            )
        ]

        left_labels = [
            pair[0]
            for pair in pairs
        ]

        right_labels = [
            pair[1]
            for pair in pairs
        ]

        agreements = sum(
            left_label == right_label
            for left_label, right_label in pairs
        )

        raw_agreement = (
            agreements / len(pairs)
            if pairs
            else None
        )

        kappa = (
            cohen_kappa_score(
                left_labels,
                right_labels,
                labels=CATEGORIES,
            )
            if pairs
            else None
        )

        results.append({
            "Comparison": (
                f"Reviewer {left} vs. Reviewer {right}"
            ),
            "Eligible articles": len(pairs),
            "Agreements": agreements,
            "Raw agreement": raw_agreement,
            "Metric": "Cohen's kappa",
            "Coefficient": kappa,
        })

    # Complete three-reviewer cases
    complete_labels = [
        [
            row["R1"],
            row["R2"],
            row["R3"],
        ]
        for row in audit
        if row["valid_reviewers"] == 3
    ]

    observed, fleiss = calculate_fleiss_kappa(
        complete_labels
    )

    results.append({
        "Comparison": "Three-reviewer agreement",
        "Eligible articles": len(complete_labels),
        "Agreements": "",
        "Raw agreement": observed,
        "Metric": "Fleiss' kappa",
        "Coefficient": fleiss,
    })

    ac1 = calculate_gwet_ac1(
        complete_labels
    )

    results.append({
        "Comparison": "Prevalence-robust agreement",
        "Eligible articles": len(complete_labels),
        "Agreements": "",
        "Raw agreement": observed,
        "Metric": "Gwet's AC1",
        "Coefficient": ac1,
    })

    all_units = [
        [
            row[f"R{number}"]
            for number in range(1, 4)
            if row[f"R{number}"] in CATEGORIES
        ]
        for row in audit
    ]

    pairable, alpha = calculate_krippendorff_alpha(
        all_units
    )

    results.append({
        "Comparison":
            "Nominal reliability with missing labels",
        "Eligible articles": pairable,
        "Agreements": "",
        "Raw agreement": "",
        "Metric": "Krippendorff's alpha",
        "Coefficient": alpha,
    })

    return results


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
    coverage,
    availability,
    consensus,
    reliability,
    audit,
):
    """Save raw expert-consensus results."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "reviewer_coverage.csv": coverage,
        "label_availability.csv": availability,
        "consensus_summary.csv": consensus,
        "reviewer_reliability.csv": reliability,
        "article_level_audit.csv": audit,
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
                f"Reviewer {number} - "
                "Articles and Labels for review.xlsx"
            )
            for number in range(1, 4)
        ],
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            RESULTS_DIR
            / "consensus_results"
        ),
    )

    args = parser.parse_args()

    try:

        reviewers = [
            read_reviewer(path)
            for path in args.reviewers
        ]

        audit = build_audit(reviewers)

        coverage = calculate_coverage(audit)
        availability = calculate_availability(audit)
        consensus = calculate_consensus(audit)
        reliability = calculate_reliability(audit)

        save_results(
            args.output_dir,
            coverage,
            availability,
            consensus,
            reliability,
            audit,
        )

        print(
            "Expert consensus analysis complete. "
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