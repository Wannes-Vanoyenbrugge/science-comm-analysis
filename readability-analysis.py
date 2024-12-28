import textstat
import pandas as pd

# Load the article text
def load_text(file_path):
    """Loads the text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

# Path to your text file
file_path = "nyt_psilocybin.txt"
article_text = load_text(file_path)

# Split the article into paragraphs
paragraphs = [p.strip() for p in article_text.split("\n\n") if p.strip()]

# Initialize an empty list to store results
readability_results = []

# Calculate readability metrics for each paragraph
for i, paragraph in enumerate(paragraphs):
    # Calculate readability scores
    flesch_reading_ease = textstat.flesch_reading_ease(paragraph)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(paragraph)
    gunning_fog = textstat.gunning_fog(paragraph)
    smog_index = textstat.smog_index(paragraph)
    dale_chall_score = textstat.dale_chall_readability_score(paragraph)

    # Append results to list
    readability_results.append({
        "Paragraph Index": i,
        "Flesch Reading Ease": flesch_reading_ease,
        "Flesch-Kincaid Grade": flesch_kincaid_grade,
        "Gunning Fog Index": gunning_fog,
        "SMOG Index": smog_index,
        "Dale-Chall Score": dale_chall_score
    })

# Convert results to a DataFrame
df_readability = pd.DataFrame(readability_results)

# Display summary statistics
print("\n=== Readability Summary ===")
print(df_readability.describe())

# Save results to a CSV file
df_readability.to_csv("readability_scores.csv", index=False)

# Visualize readability scores
import matplotlib.pyplot as plt

# Plot Flesch-Kincaid Grade Level
plt.figure(figsize=(10, 5))
plt.plot(df_readability["Paragraph Index"], df_readability["Flesch-Kincaid Grade"], marker='o')
plt.title("Flesch-Kincaid Grade Level Across Paragraphs")
plt.xlabel("Paragraph Index")
plt.ylabel("Grade Level")
plt.axhline(8, color="green", linestyle="--", label="Grade 8 (General Audience)")
plt.axhline(12, color="red", linestyle="--", label="Grade 12 (More Advanced)")
plt.legend()
plt.show()

# Plot Flesch Reading Ease
plt.figure(figsize=(10, 5))
plt.plot(df_readability["Paragraph Index"], df_readability["Flesch Reading Ease"], marker='o')
plt.title("Flesch Reading Ease Across Paragraphs")
plt.xlabel("Paragraph Index")
plt.ylabel("Reading Ease Score")
plt.axhline(60, color="green", linestyle="--", label="60 (Easy to Read)")
plt.axhline(30, color="red", linestyle="--", label="30 (Difficult to Read)")
plt.legend()
plt.show()
