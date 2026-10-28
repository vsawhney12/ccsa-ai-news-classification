import unittest
from collections import Counter
from pathlib import Path
import importlib.util

from statsmodels.stats.inter_rater import fleiss_kappa


# ==========================================================
# IMPORT EXISTING RELIABILITY FUNCTIONS
# ==========================================================

SCRIPT_DIR = Path(__file__).resolve().parent

SOURCE_FILE = SCRIPT_DIR / "expert_consensus.py"

spec = importlib.util.spec_from_file_location(
    "expert_consensus",
    SOURCE_FILE,
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

calculate_fleiss_kappa = module.calculate_fleiss_kappa
calculate_gwet_ac1 = module.calculate_gwet_ac1
calculate_krippendorff_alpha = module.calculate_krippendorff_alpha

CATEGORIES = module.CATEGORIES


# ==========================================================
# INDEPENDENT REFERENCE CALCULATIONS
# ==========================================================

def reference_fleiss_kappa(labels):
    """Compare against statsmodels."""

    table = [
        [row.count(category) for category in CATEGORIES]
        for row in labels
    ]

    return fleiss_kappa(table, method="fleiss")


def reference_gwet_ac1(labels):
    """Calculate nominal AC1 independently."""

    n = len(labels)
    k = len(CATEGORIES)

    agreements = []

    for row in labels:
        matching_pairs = sum(
            row[i] == row[j]
            for i in range(3)
            for j in range(i + 1, 3)
        )

        agreements.append(matching_pairs / 3)

    observed = sum(agreements) / n

    all_labels = [
        label
        for row in labels
        for label in row
    ]

    proportions = [
        all_labels.count(category) / len(all_labels)
        for category in CATEGORIES
    ]

    expected = sum(
        p * (1 - p)
        for p in proportions
    ) / (k - 1)

    return (observed - expected) / (1 - expected)


def reference_krippendorff_alpha(units):
    """Calculate nominal alpha using pairwise disagreements."""

    units = [
        row for row in units
        if len(row) >= 2
    ]

    labels = [
        label
        for row in units
        for label in row
    ]

    n = len(labels)

    if n < 2:
        return None

    observed_numerator = 0

    for row in units:
        disagreements = sum(
            left != right
            for left in row
            for right in row
        )

        observed_numerator += (
            disagreements / (len(row) - 1)
        )

    observed = observed_numerator / n

    counts = Counter(labels)

    expected = (
        n * n - sum(c * c for c in counts.values())
    ) / (n * (n - 1))

    if expected == 0:
        return None

    return 1 - observed / expected


# ==========================================================
# VALIDATION TESTS
# ==========================================================

class TestReliability(unittest.TestCase):

    def setUp(self):

        self.sample = [
            ["Public Safety", "Public Safety", "Public Safety"],
            ["Public Health", "Public Health", "Public Health"],
            ["Drug Alert", "Drug Alert", "Drug Alert"],
            ["Public Safety", "Public Safety", "Public Health"],
            ["Public Health", "Public Health", "Drug Alert"],
            ["Drug Alert", "Drug Alert", "Public Safety"],
            ["Public Safety", "Public Health", "Drug Alert"],
        ]

    # ------------------------------------------------------
    # FLEISS' KAPPA
    # ------------------------------------------------------

    def test_fleiss_against_statsmodels(self):

        _, actual = calculate_fleiss_kappa(self.sample)

        expected = reference_fleiss_kappa(self.sample)

        self.assertAlmostEqual(actual, expected, places=10)

    def test_fleiss_perfect_agreement(self):

        labels = [
            ["Public Safety"] * 3,
            ["Public Health"] * 3,
            ["Drug Alert"] * 3,
        ]

        observed, kappa = calculate_fleiss_kappa(labels)

        self.assertAlmostEqual(observed, 1.0)
        self.assertAlmostEqual(kappa, 1.0)

    # ------------------------------------------------------
    # GWET'S AC1
    # ------------------------------------------------------

    def test_gwet_against_reference(self):

        actual = calculate_gwet_ac1(self.sample)

        expected = reference_gwet_ac1(self.sample)

        self.assertAlmostEqual(actual, expected, places=10)

    def test_gwet_perfect_agreement(self):

        labels = [
            ["Public Safety"] * 3,
            ["Public Health"] * 3,
            ["Drug Alert"] * 3,
        ]

        actual = calculate_gwet_ac1(labels)

        self.assertAlmostEqual(actual, 1.0)

    # ------------------------------------------------------
    # KRIPPENDORFF'S ALPHA
    # ------------------------------------------------------

    def test_alpha_against_reference(self):

        _, actual = calculate_krippendorff_alpha(self.sample)

        expected = reference_krippendorff_alpha(self.sample)

        self.assertAlmostEqual(actual, expected, places=10)

    def test_alpha_missing_annotations(self):

        labels = [
            ["Public Safety", "Public Safety", "Public Safety"],
            ["Public Health", "Public Health"],
            ["Drug Alert", "Public Health"],
            ["Public Safety"],
        ]

        count, actual = calculate_krippendorff_alpha(labels)

        expected = reference_krippendorff_alpha(labels)

        self.assertEqual(count, 3)
        self.assertAlmostEqual(actual, expected, places=10)

    def test_alpha_perfect_agreement(self):

        labels = [
            ["Public Safety"] * 3,
            ["Public Health"] * 3,
            ["Drug Alert"] * 3,
        ]

        _, actual = calculate_krippendorff_alpha(labels)

        self.assertAlmostEqual(actual, 1.0)

    # ------------------------------------------------------
    # EDGE CASES
    # ------------------------------------------------------

    def test_empty_inputs(self):

        observed, kappa = calculate_fleiss_kappa([])

        self.assertIsNone(observed)
        self.assertIsNone(kappa)

        self.assertIsNone(calculate_gwet_ac1([]))

        count, alpha = calculate_krippendorff_alpha([])

        self.assertEqual(count, 0)
        self.assertIsNone(alpha)

    def test_single_category(self):

        labels = [
            ["Public Safety"] * 3,
            ["Public Safety"] * 3,
        ]

        _, kappa = calculate_fleiss_kappa(labels)

        self.assertIsNone(kappa)

        _, alpha = calculate_krippendorff_alpha(labels)

        self.assertIsNone(alpha)


# ==========================================================
# RUN TESTS
# ==========================================================

if __name__ == "__main__":

    unittest.main(verbosity=2)