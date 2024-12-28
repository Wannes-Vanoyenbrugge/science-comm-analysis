########################################
# STEP 0: INSTALL / IMPORT LIBRARIES
########################################

# If you haven't already installed these, uncomment and run:
# !pip install nltk textblob nrclex matplotlib pandas

import nltk
import matplotlib.pyplot as plt
import pandas as pd

# Sentiment / Emotion libraries
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nrclex import NRCLex

########################################
# STEP 1: DOWNLOAD NLTK RESOURCES (VADER LEXICON)
########################################

# Only need to run once in a new environment:
nltk.download('vader_lexicon')

########################################
# STEP 2: LOAD AND SPLIT THE ARTICLE TEXT
########################################

def load_paragraphs_from_file(file_path):
    """Loads the text file and splits into paragraphs based on double newlines."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Split by double newlines
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs

file_path = "nyt_psilocybin.txt"
paragraphs = load_paragraphs_from_file(file_path)

# Quick sanity check
print(f"Loaded {len(paragraphs)} paragraphs.")
for i, p in enumerate(paragraphs[:3]):
    print(f"\nParagraph {i+1}:\n{p[:200]}{'...' if len(p) > 200 else ''}")

########################################
# STEP 3: SENTIMENT ANALYSIS WITH VADER
########################################

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

sentiment_results = []

for i, paragraph in enumerate(paragraphs):
    scores = sid.polarity_scores(paragraph)
    sentiment_results.append({
        "paragraph_index": i,
        "text": paragraph,
        "neg": scores["neg"],
        "neu": scores["neu"],
        "pos": scores["pos"],
        "compound": scores["compound"]
    })

########################################
# STEP 4: EMOTION ANALYSIS WITH NRC LEXICON
########################################

emotion_results = []

for i, paragraph in enumerate(paragraphs):
    nrc_object = NRCLex(paragraph)
    emotion_dict = nrc_object.raw_emotion_scores
    
    emotion_results.append({
        "paragraph_index": i,
        "text": paragraph,
        "emotions": emotion_dict
    })

########################################
# STEP 5: VISUALIZE VADER SENTIMENT
########################################

# Create lists for plotting
paragraph_indices = [res["paragraph_index"] for res in sentiment_results]
compound_scores = [res["compound"] for res in sentiment_results]

plt.figure(figsize=(10, 5))
plt.plot(paragraph_indices, compound_scores, marker='o')
plt.title("VADER Compound Sentiment Across Paragraphs")
plt.xlabel("Paragraph Index")
plt.ylabel("Compound Sentiment Score")
plt.axhline(0, color='red', linestyle='--')  # zero line for neutral sentiment
plt.show()

########################################
# STEP 6: VISUALIZE KEY EMOTIONS
########################################

# Convert emotion_results into a Pandas DataFrame
rows = []
for item in emotion_results:
    paragraph_index = item["paragraph_index"]
    for emotion, count in item["emotions"].items():
        rows.append({
            "paragraph_index": paragraph_index,
            "emotion": emotion,
            "count": count
        })

df_emotions = pd.DataFrame(rows)

# Pivot so each emotion is a column, paragraphs are rows
df_emotions_pivot = df_emotions.pivot_table(
    index="paragraph_index",
    columns="emotion",
    values="count",
    aggfunc="sum"
).fillna(0)

# OPTIONAL: If the article is long, focusing on a few key emotions can be clearer
selected_emotions = ["anger", "fear", "trust", "joy", "anticipation", "sadness"]
df_selected = df_emotions_pivot[selected_emotions].copy()

# Plot the selected emotions as a stacked bar chart (per paragraph)
ax = df_selected.plot(
    kind="bar", 
    stacked=True, 
    figsize=(12, 6),
    colormap="Set3",
    width=0.7
)
plt.title("NRC Emotion Scores per Paragraph (Selected Emotions)")
plt.xlabel("Paragraph Index")
plt.ylabel("Emotion Score")
plt.legend(title="Emotion")
plt.tight_layout()
plt.show()

########################################
# STEP 7: OPTIONAL EXPLORATION
########################################

# 1. Print out paragraphs with extreme positive/negative sentiment
print("\n=== Paragraphs with Most Positive Sentiment ===")
top_pos = sorted(sentiment_results, key=lambda x: x["compound"], reverse=True)[:3]
for item in top_pos:
    print(f"\nParagraph {item['paragraph_index']} (compound={item['compound']}):")
    print(item['text'][:300], "...\n")

print("\n=== Paragraphs with Most Negative Sentiment ===")
top_neg = sorted(sentiment_results, key=lambda x: x["compound"])[:3]
for item in top_neg:
    print(f"\nParagraph {item['paragraph_index']} (compound={item['compound']}):")
    print(item['text'][:300], "...\n")

# 2. Print out paragraphs with the highest 'fear' or 'joy'
df_emotions_pivot["fear"] = df_emotions_pivot.get("fear", 0)
df_emotions_pivot["joy"] = df_emotions_pivot.get("joy", 0)

max_fear_paragraph = df_emotions_pivot["fear"].idxmax()
max_joy_paragraph = df_emotions_pivot["joy"].idxmax()

print(f"\nParagraph with Highest 'fear': {max_fear_paragraph}")
print(paragraphs[max_fear_paragraph], "\n")

print(f"Paragraph with Highest 'joy': {max_joy_paragraph}")
print(paragraphs[max_joy_paragraph], "\n")
