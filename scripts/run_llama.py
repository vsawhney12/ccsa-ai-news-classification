import os
import time
import pandas as pd
import ollama
from pathlib import Path

# ==========================================================
# SETTINGS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "ArticlesV2_Blind_review.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "ArticlesV2_Blind_review_llama.csv"

MODEL = "llama3.1:8b"

SAVE_EVERY = 25

# ==========================================================
# LOAD DATA
# ==========================================================

if os.path.exists(OUTPUT_FILE):
    print("Loading previous progress...")
    df = pd.read_csv(OUTPUT_FILE)

else:
    print("Loading original dataset...")
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

df.columns = df.columns.str.strip()

df["reviewed_primary_category"] = (
    df["reviewed_primary_category"]
    .fillna("")
    .astype(str)
)

if "reviewed_primary_category" not in df.columns:
    df["reviewed_primary_category"] = ""

# ==========================================================
# PROMPT
# ==========================================================

def build_prompt(article):

    article = article[:1200]

    return f"""
You are an expert classifier for Canadian drug-related news.

Classify the article into EXACTLY ONE category.

==================================================
CATEGORY 1: Drug Alert
==================================================

Official warnings or advisories about the
unregulated drug supply.

Examples:

- drug alert
- drug advisory
- drug bulletin
- contaminated drugs
- adulterated drugs
- unexpected substances
- novel psychoactive substances
- dangerous drug supply

==================================================
CATEGORY 2: Public Safety
==================================================

Crime, policing and justice.

Examples:

- police
- trafficking
- seizures
- investigations
- organized crime
- arrests
- charges
- convictions
- production
- manufacturing
- smuggling
- violence
- homicide
- shootings
- impaired driving

==================================================
CATEGORY 3: Public Health
==================================================

Health impacts and public health response.

Examples:

- overdose
- toxic drug poisoning
- harm reduction
- treatment
- addiction
- substance use
- hospitalization
- emergency department
- surveillance
- safer supply
- public health monitoring
- overdose prevention

==================================================
STRICT RULES
==================================================

1. Drug Alert overrides everything else.

2. If primarily about policing/crime,
choose Public Safety.

3. Otherwise choose Public Health.

4. Ignore secondary topics.

5. Output ONLY ONE of:

Drug Alert

Public Safety

Public Health

ARTICLE:

{article}

ANSWER:
"""

# ==========================================================
# EXTRACT LABEL
# ==========================================================

def extract_label(response):

    text = response.lower()

    if "drug alert" in text:
        return "Drug Alert"

    if "public safety" in text:
        return "Public Safety"

    if "public health" in text:
        return "Public Health"

    return "Unknown"

# ==========================================================
# CLASSIFY
# ==========================================================

total = len(df)

for i in range(total):

    if df.loc[i, "reviewed_primary_category"].strip() != "":
        continue

    article = str(df.loc[i, "article_text"])

    prompt = build_prompt(article)

    success = False

    while not success:

        try:

            response = ollama.chat(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response["message"]["content"]

            label = extract_label(answer)

            df.loc[i, "reviewed_primary_category"] = label

            print(f"{i+1}/{total} -> {label}")

            success = True

        except Exception as e:

            print(e)
            print("Retrying in 5 seconds...")

            time.sleep(5)

    if (i + 1) % SAVE_EVERY == 0:

        df.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"\nProgress saved ({i+1}/{total})\n")

# ==========================================================
# SAVE
# ==========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n===================================")
print("DONE!")
print("===================================")

print(df["reviewed_primary_category"].value_counts())