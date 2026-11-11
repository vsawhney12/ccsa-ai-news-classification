"""
Article-level missingness audit for 172 two-reviewer agreement cases.

This script does not modify expert consensus or model predictions.

Outputs:
1. two_reviewer_missingness_audit.csv
2. two_reviewer_missingness_summary.csv

Missingness categories:
- Documented relevance concern
- Likely French-language article
- Outside observed annotation range
- Unexplained

Important:
Language and annotation-range findings are possible explanations
for missing labels, not verified reasons for reviewer nonresponse.
"""

import csv
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from langdetect import detect_langs, DetectorFactory

from expert_consensus import (
    CATEGORIES,
    normalize_id,
    read_reviewer,
    read_reviewer_notes,
    identify_relevance_concern,
)


# ==========================================================
# SETTINGS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RESULTS_DIR = PROJECT_ROOT / "results" / "consensus_results"

ARTICLE_FILE = DATA_DIR / "ArticlesV2_Blind_review.csv"

TWO_REVIEWER_FILE = (
    RESULTS_DIR / "two_reviewer_agreement_cases.csv"
)

REVIEWER_FILES = {
    f"R{i}": DATA_DIR / (
        f"Reviewer {i} - Articles and Labels for review.xlsx"
    )
    for i in range(1, 4)
}

AUDIT_FILE = (
    RESULTS_DIR / "two_reviewer_missingness_audit.csv"
)

SUMMARY_FILE = (
    RESULTS_DIR / "two_reviewer_missingness_summary.csv"
)

DetectorFactory.seed = 0


# ==========================================================
# LANGUAGE DETECTION
# ==========================================================

def detect_language(text):
    """
    Detect the likely article language.

    Returns:
        detected_language
        detection_confidence
        language_evidence

    Detection is probabilistic and does not establish
    why a reviewer omitted an article.
    """

    if pd.isna(text) or not str(text).strip():
        return "Unknown", None, "Article text unavailable"

    text = str(text).strip()

    # Use up to 3,000 characters for language identification.
    sample = text[:3000]

    try:
        predictions = detect_langs(sample)

        if not predictions:
            return "Unknown", None, "No language prediction"

        prediction = predictions[0]

        language = prediction.lang
        confidence = float(prediction.prob)

        return (
            language,
            confidence,
            f"langdetect={language}; probability={confidence:.4f}",
        )

    except Exception as error:
        return (
            "Unknown",
            None,
            f"Language detection failed: {error}",
        )


# ==========================================================
# REVIEWER COVERAGE
# ==========================================================

def calculate_reviewer_ranges(reviewer_labels):
    """
    Identify each reviewer's observed valid-label coverage.

    Minimum and maximum IDs are descriptive boundaries,
    not proof of assigned review responsibilities.
    """

    ranges = {}

    for reviewer, labels in reviewer_labels.items():

        valid_ids = sorted(
            int(review_id)
            for review_id, label in labels.items()
            if label in CATEGORIES
        )

        if not valid_ids:
            ranges[reviewer] = None
            continue

        ranges[reviewer] = {
            "minimum": min(valid_ids),
            "maximum": max(valid_ids),
            "valid_count": len(valid_ids),
        }

    return ranges


def check_annotation_range(review_id, reviewer, ranges):
    """
    Check whether an article falls outside the observed
    valid-label ID range for its missing reviewer.
    """

    reviewer_range = ranges.get(reviewer)

    if reviewer_range is None:
        return False, "No valid-label range available"

    article_id = int(review_id)

    minimum = reviewer_range["minimum"]
    maximum = reviewer_range["maximum"]

    outside = (
        article_id < minimum
        or article_id > maximum
    )

    evidence = (
        f"Article ID={article_id}; "
        f"{reviewer} observed valid-label range="
        f"{minimum}-{maximum}; "
        f"valid labels={reviewer_range['valid_count']}"
    )

    return outside, evidence


# ==========================================================
# LOAD ARTICLE DATA
# ==========================================================

def load_articles():

    articles = pd.read_csv(ARTICLE_FILE)

    required = {"review_id", "article_text"}

    missing = required - set(articles.columns)

    if missing:
        raise ValueError(
            f"Article dataset missing columns: {missing}"
        )

    articles["review_id"] = articles["review_id"].apply(
        normalize_id
    )

    if articles["review_id"].duplicated().any():
        raise ValueError(
            "Duplicate review_id values in article dataset"
        )

    return articles.set_index("review_id")


# ==========================================================
# BUILD MISSINGNESS AUDIT
# ==========================================================

def build_missingness_audit():

    articles = load_articles()

    two_reviewer_cases = pd.read_csv(
        TWO_REVIEWER_FILE,
        dtype={"review_id": str},
        keep_default_na=False,
    )

    if len(two_reviewer_cases) != 172:
        raise ValueError(
            "Expected 172 two-reviewer agreement cases; "
            f"found {len(two_reviewer_cases)}"
        )

    if two_reviewer_cases["review_id"].duplicated().any():
        raise ValueError(
            "Duplicate review IDs in two-reviewer cases"
        )

    reviewer_labels = {
        reviewer: read_reviewer(path)
        for reviewer, path in REVIEWER_FILES.items()
    }

    reviewer_notes = {
        reviewer: read_reviewer_notes(path)
        for reviewer, path in REVIEWER_FILES.items()
    }

    reviewer_ranges = calculate_reviewer_ranges(
        reviewer_labels
    )

    results = []

    for _, row in two_reviewer_cases.iterrows():

        review_id = normalize_id(row["review_id"])

        if review_id not in articles.index:
            raise ValueError(
                f"Article ID {review_id} not found in dataset"
            )

        labels = {
            reviewer: reviewer_labels[reviewer][review_id]
            for reviewer in REVIEWER_FILES
        }

        valid_reviewers = [
            reviewer
            for reviewer, label in labels.items()
            if label in CATEGORIES
        ]

        missing_reviewers = [
            reviewer
            for reviewer, label in labels.items()
            if label not in CATEGORIES
        ]

        if len(valid_reviewers) != 2 or len(missing_reviewers) != 1:
            raise ValueError(
                f"Unexpected reviewer availability for ID {review_id}"
            )

        first_label = labels[valid_reviewers[0]]
        second_label = labels[valid_reviewers[1]]

        if first_label != second_label:
            raise ValueError(
                f"Two reviewers disagree for ID {review_id}"
            )

        missing_reviewer = missing_reviewers[0]

        missing_label = labels[missing_reviewer]

        reviewer_note = reviewer_notes[
            missing_reviewer
        ].get(review_id, "")

        relevance_flag = identify_relevance_concern(
            reviewer_note
        )

        article_text = articles.loc[
            review_id,
            "article_text"
        ]

        language, confidence, language_evidence = (
            detect_language(article_text)
        )

        likely_french = (
            language == "fr"
            and confidence is not None
            and confidence >= 0.90
        )

        outside_range, range_evidence = (
            check_annotation_range(
                review_id,
                missing_reviewer,
                reviewer_ranges,
            )
        )

        # --------------------------------------------------
        # Assign descriptive missingness category
        # --------------------------------------------------

        if relevance_flag:

            category = "Documented relevance concern"

            evidence_status = "Documented reviewer note"

            evidence = (
                f"Reviewer note: {reviewer_note}"
            )

        elif likely_french:

            category = "Likely French-language article"

            evidence_status = (
                "Automated language inference"
            )

            evidence = language_evidence

        elif outside_range:

            category = (
                "Outside observed annotation range"
            )

            evidence_status = (
                "Inferred from reviewer coverage"
            )

            evidence = range_evidence

        else:

            category = "Unexplained"

            evidence_status = (
                "No supporting explanation identified"
            )

            evidence = (
                "No explicit relevance concern; "
                "French language not detected with "
                "high confidence; article ID within "
                "observed valid-label range."
            )

        results.append({
            "review_id": review_id,

            "missing_reviewer": missing_reviewer,

            "missing_label": (
                missing_label
                if missing_label
                else "Blank"
            ),

            "agreeing_reviewers": (
                f"{valid_reviewers[0]}, "
                f"{valid_reviewers[1]}"
            ),

            "agreed_category": first_label,

            "reviewer_note": reviewer_note,

            "documented_relevance_concern": (
                relevance_flag
            ),

            "detected_language": language,

            "language_detection_confidence": (
                confidence
            ),

            "likely_french": likely_french,

            "language_evidence": language_evidence,

            "outside_observed_range": outside_range,

            "annotation_range_evidence": (
                range_evidence
            ),

            "missingness_category": category,

            "evidence_status": evidence_status,

            "evidence": evidence,
        })

    return pd.DataFrame(results)


# ==========================================================
# SUMMARY
# ==========================================================

def create_summary(audit):

    counts = Counter(
        audit["missingness_category"]
    )

    categories = [
        "Documented relevance concern",
        "Likely French-language article",
        "Outside observed annotation range",
        "Unexplained",
    ]

    summary = []

    for category in categories:

        count = counts[category]

        summary.append({
            "Missingness category": category,
            "Articles": count,
            "Percentage": round(
                count / len(audit) * 100,
                2,
            ),
        })

    return pd.DataFrame(summary)


# ==========================================================
# MAIN
# ==========================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit = build_missingness_audit()

    summary = create_summary(audit)

    audit.to_csv(
        AUDIT_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print("\nMISSINGNESS AUDIT COMPLETE")
    print("-" * 45)

    print(f"Articles audited: {len(audit)}")

    print("\nMISSINGNESS SUMMARY")

    print(
        summary.to_string(index=False)
    )

    print("\nFiles saved:")
    print(AUDIT_FILE)
    print(SUMMARY_FILE)

    print(
        "\nNote: Language and annotation-range "
        "categories are inferred explanations, "
        "not confirmed reasons for missing labels."
    )


if __name__ == "__main__":
    main()