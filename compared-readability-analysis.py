import textstat
import pandas as pd
import matplotlib.pyplot as plt

def load_text(file_path):
    """Loads text from a file and returns the full text as a string."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def get_readability_scores(text):
    """
    Splits the text into paragraphs and calculates multiple readability metrics
    for each paragraph. Returns a pandas DataFrame.
    """
    # Split text into paragraphs (based on double newlines)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    readability_results = []
    for i, paragraph in enumerate(paragraphs):
        # Calculate readability scores
        flesch_reading_ease = textstat.flesch_reading_ease(paragraph)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(paragraph)
        gunning_fog = textstat.gunning_fog(paragraph)
        smog_index = textstat.smog_index(paragraph)
        dale_chall_score = textstat.dale_chall_readability_score(paragraph)

        readability_results.append({
            "Paragraph Index": i,
            "Flesch Reading Ease": flesch_reading_ease,
            "Flesch-Kincaid Grade": flesch_kincaid_grade,
            "Gunning Fog Index": gunning_fog,
            "SMOG Index": smog_index,
            "Dale-Chall Score": dale_chall_score
        })

    df = pd.DataFrame(readability_results)
    return df

# 1. LOAD TEXTS
guardian_text = load_text("guardian_psilocybin.txt")
nyt_text = load_text("nyt_psilocybin.txt")

# 2. COMPUTE READABILITY DATAFRAMES
df_guardian = get_readability_scores(guardian_text)
df_nyt = get_readability_scores(nyt_text)

# 3. PRINT SUMMARIES
print("\n=== Guardian Readability Summary ===")
print(df_guardian.describe())

print("\n=== NYT Readability Summary ===")
print(df_nyt.describe())

# 4. SAVE EACH DATAFRAME TO CSV
df_guardian.to_csv("readability_scores_guardian.csv", index=False)
df_nyt.to_csv("readability_scores_nyt.csv", index=False)

# 5. VISUALIZE IN COMPARISON PLOTS

# --- FLESCH-KINCAID GRADE ---
plt.figure(figsize=(10, 5))
plt.plot(
    df_guardian["Paragraph Index"], 
    df_guardian["Flesch-Kincaid Grade"], 
    marker='o', 
    color="blue", 
    label="Guardian"
)
plt.plot(
    df_nyt["Paragraph Index"], 
    df_nyt["Flesch-Kincaid Grade"], 
    marker='o', 
    linestyle='--',   # Dashed line
    color="orange", 
    label="NYT"
)
plt.title("Flesch-Kincaid Grade Level Comparison")
plt.xlabel("Paragraph Index")
plt.ylabel("Grade Level")
plt.axhline(8, color="green", linestyle="--", label="Grade 8 (General Audience)")
plt.axhline(12, color="red", linestyle="--", label="Grade 12 (More Advanced)")
plt.legend()
plt.show()

# --- FLESCH READING EASE ---
plt.figure(figsize=(10, 5))
plt.plot(
    df_guardian["Paragraph Index"], 
    df_guardian["Flesch Reading Ease"], 
    marker='o', 
    color="blue", 
    label="Guardian"
)
plt.plot(
    df_nyt["Paragraph Index"], 
    df_nyt["Flesch Reading Ease"], 
    marker='o', 
    linestyle='--',   # Dashed line
    color="orange", 
    label="NYT"
)
plt.title("Flesch Reading Ease Comparison")
plt.xlabel("Paragraph Index")
plt.ylabel("Reading Ease Score")
plt.axhline(60, color="green", linestyle="--", label="60 (Easy to Read)")
plt.axhline(30, color="red", linestyle="--", label="30 (Difficult to Read)")
plt.legend()
plt.show()
