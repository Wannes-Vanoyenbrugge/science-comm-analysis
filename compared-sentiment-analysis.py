########################################
# STEP 0: INSTALL / IMPORT LIBRARIES
########################################

# If you haven't already installed these, uncomment and run:
# !pip install nltk textblob nrclex matplotlib pandas

import nltk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sentiment / Emotion libraries
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nrclex import NRCLex

########################################
# STEP 1: DOWNLOAD NLTK RESOURCES (VADER LEXICON)
########################################

# Only need to run once in a new environment:
nltk.download('vader_lexicon')

########################################
# STEP 2: HELPER FUNCTIONS
########################################

def load_paragraphs_from_file(file_path):
    """Loads the text file and splits into paragraphs based on double newlines."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs

def get_vader_sentiment(paragraphs):
    """
    Given a list of paragraphs, compute VADER sentiment scores for each
    and return a DataFrame with columns: paragraph_index, neg, neu, pos, compound.
    """
    sid = SentimentIntensityAnalyzer()
    results = []
    for i, paragraph in enumerate(paragraphs):
        scores = sid.polarity_scores(paragraph)
        results.append({
            "paragraph_index": i,
            "neg": scores["neg"],
            "neu": scores["neu"],
            "pos": scores["pos"],
            "compound": scores["compound"]
        })
    df = pd.DataFrame(results)
    return df

def get_nrc_emotions(paragraphs):
    """
    Given a list of paragraphs, use NRCLex to compute emotion scores.
    Returns a pivoted DataFrame with each emotion as a column, indexed by paragraph_index.
    """
    rows = []
    for i, paragraph in enumerate(paragraphs):
        nrc_object = NRCLex(paragraph)
        emotion_dict = nrc_object.raw_emotion_scores
        for emotion, count in emotion_dict.items():
            rows.append({"paragraph_index": i, "emotion": emotion, "count": count})

    # Create DataFrame and pivot
    df_emotions = pd.DataFrame(rows)
    if df_emotions.empty:
        # If text is empty or no emotions found, return an empty DataFrame
        return pd.DataFrame()

    df_emotions_pivot = df_emotions.pivot_table(
        index="paragraph_index",
        columns="emotion",
        values="count",
        aggfunc="sum"
    ).fillna(0)
    return df_emotions_pivot

########################################
# STEP 3: LOAD DATA
########################################

guardian_paragraphs = load_paragraphs_from_file("guardian_psilocybin.txt")
nyt_paragraphs = load_paragraphs_from_file("nyt_psilocybin.txt")

print(f"Loaded {len(guardian_paragraphs)} paragraphs from Guardian.")
print(f"Loaded {len(nyt_paragraphs)} paragraphs from NYT.")

########################################
# STEP 4: PERFORM VADER SENTIMENT ANALYSIS
########################################

df_guardian_sentiment = get_vader_sentiment(guardian_paragraphs)
df_nyt_sentiment = get_vader_sentiment(nyt_paragraphs)

# Make sure paragraph_index is integer
df_guardian_sentiment["paragraph_index"] = df_guardian_sentiment["paragraph_index"].astype(int)
df_nyt_sentiment["paragraph_index"] = df_nyt_sentiment["paragraph_index"].astype(int)

# Print out some basic stats
print("\n=== Guardian VADER Sentiment (head) ===")
print(df_guardian_sentiment.head())
print("\n=== NYT VADER Sentiment (head) ===")
print(df_nyt_sentiment.head())

########################################
# STEP 5: PERFORM NRC EMOTION ANALYSIS
########################################

df_guardian_emotions = get_nrc_emotions(guardian_paragraphs)
df_nyt_emotions = get_nrc_emotions(nyt_paragraphs)

########################################
# STEP 6: VISUALIZE VADER COMPOUND SENTIMENT (COMPARISON)
########################################

plt.figure(figsize=(10, 5))

# Guardian (solid line)
plt.plot(
    df_guardian_sentiment["paragraph_index"],
    df_guardian_sentiment["compound"],
    marker='o',
    color="blue",
    label="Guardian (Compound)",
)

# NYT (dashed line)
plt.plot(
    df_nyt_sentiment["paragraph_index"],
    df_nyt_sentiment["compound"],
    marker='o',
    linestyle='--',
    color="orange",
    label="NYT (Compound)",
)

plt.title("VADER Compound Sentiment Across Paragraphs: Guardian vs. NYT")
plt.xlabel("Paragraph Index")
plt.ylabel("Compound Sentiment Score")
plt.axhline(0, color='red', linestyle='--', label="Neutral (0)")
plt.legend()
plt.tight_layout()
plt.show()

########################################
# STEP 7: VISUALIZE KEY EMOTIONS AS STACKED BAR
########################################

# Choose a few emotions to compare
selected_emotions = ["anger", "fear", "trust", "joy", "anticipation", "sadness"]

# -- Guardian emotions --
if not df_guardian_emotions.empty:
    df_guardian_subset = df_guardian_emotions[selected_emotions].fillna(0)
    ax = df_guardian_subset.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
        colormap="Set3",
        width=0.7
    )
    plt.title("Guardian: NRC Emotion Scores per Paragraph (Selected Emotions)")
    plt.xlabel("Paragraph Index")
    plt.ylabel("Emotion Score")
    plt.legend(title="Emotion")
    plt.tight_layout()
    plt.show()
else:
    print("\nNo Guardian emotion data found.")

# -- NYT emotions --
if not df_nyt_emotions.empty:
    df_nyt_subset = df_nyt_emotions[selected_emotions].fillna(0)
    ax = df_nyt_subset.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
        colormap="Set3",
        width=0.7
    )
    plt.title("NYT: NRC Emotion Scores per Paragraph (Selected Emotions)")
    plt.xlabel("Paragraph Index")
    plt.ylabel("Emotion Score")
    plt.legend(title="Emotion")
    plt.tight_layout()
    plt.show()
else:
    print("\nNo NYT emotion data found.")

########################################
# STEP 8: OPTIONAL EXPLORATION
########################################
'''
# 1. Print out paragraphs with extreme positive/negative sentiment for Guardian
print("\n=== Guardian Extreme Sentiment Examples ===")
top_pos_guardian = df_guardian_sentiment.nlargest(3, "compound")
top_neg_guardian = df_guardian_sentiment.nsmallest(3, "compound")

print("\n--- Most Positive (Guardian) ---")
for idx, row in top_pos_guardian.iterrows():
    print(f"Paragraph {row['paragraph_index']} (compound={row['compound']}):")
    print(guardian_paragraphs[row['paragraph_index']][:300], "...\n")

print("\n--- Most Negative (Guardian) ---")
for idx, row in top_neg_guardian.iterrows():
    print(f"Paragraph {row['paragraph_index']} (compound={row['compound']}):")
    print(guardian_paragraphs[row['paragraph_index']][:300], "...\n")

# 2. Print out paragraphs with extreme positive/negative sentiment for NYT
print("\n=== NYT Extreme Sentiment Examples ===")
top_pos_nyt = df_nyt_sentiment.nlargest(3, "compound")
top_neg_nyt = df_nyt_sentiment.nsmallest(3, "compound")

print("\n--- Most Positive (NYT) ---")
for idx, row in top_pos_nyt.iterrows():
    print(f"Paragraph {row['paragraph_index']} (compound={row['compound']}):")
    print(nyt_paragraphs[row['paragraph_index']][:300], "...\n")

print("\n--- Most Negative (NYT) ---")
for idx, row in top_neg_nyt.iterrows():
    print(f"Paragraph {row['paragraph_index']} (compound={row['compound']}):")
    print(nyt_paragraphs[row['paragraph_index']][:300], "...\n")

# 3. Example: find paragraphs with highest 'fear' or 'joy' in Guardian
if not df_guardian_emotions.empty:
    guardian_fear = df_guardian_emotions.get("fear", pd.Series([0]*len(df_guardian_emotions)))
    guardian_joy = df_guardian_emotions.get("joy", pd.Series([0]*len(df_guardian_emotions)))
    max_fear_g = guardian_fear.idxmax()
    max_joy_g = guardian_joy.idxmax()

    print(f"\nGuardian paragraph with highest 'fear': {max_fear_g}")
    print(guardian_paragraphs[max_fear_g], "\n")

    print(f"Guardian paragraph with highest 'joy': {max_joy_g}")
    print(guardian_paragraphs[max_joy_g], "\n")

# 4. Example: find paragraphs with highest 'fear' or 'joy' in NYT
if not df_nyt_emotions.empty:
    nyt_fear = df_nyt_emotions.get("fear", pd.Series([0]*len(df_nyt_emotions)))
    nyt_joy = df_nyt_emotions.get("joy", pd.Series([0]*len(df_nyt_emotions)))
    max_fear_n = nyt_fear.idxmax()
    max_joy_n = nyt_joy.idxmax()

    print(f"\nNYT paragraph with highest 'fear': {max_fear_n}")
    print(nyt_paragraphs[max_fear_n], "\n")

    print(f"NYT paragraph with highest 'joy': {max_joy_n}")
    print(nyt_paragraphs[max_joy_n], "\n")
'''
########################################
# STEP 9: ADDITIONAL VISUALIZATIONS
########################################

# 9A. CUMULATIVE OR ROLLING ANALYSIS FOR A SELECTED EMOTION
emotion_to_analyze = "fear"  # Example emotion

if not df_guardian_emotions.empty and emotion_to_analyze in df_guardian_emotions.columns:
    guardian_emotion_series = df_guardian_emotions[emotion_to_analyze].copy()
else:
    guardian_emotion_series = pd.Series([0]*len(guardian_paragraphs))

if not df_nyt_emotions.empty and emotion_to_analyze in df_nyt_emotions.columns:
    nyt_emotion_series = df_nyt_emotions[emotion_to_analyze].copy()
else:
    nyt_emotion_series = pd.Series([0]*len(nyt_paragraphs))

# ---- CUMULATIVE SUM ----
guardian_fear_cum = guardian_emotion_series.cumsum()
nyt_fear_cum = nyt_emotion_series.cumsum()

plt.figure(figsize=(10, 5))
plt.plot(guardian_fear_cum, label="Guardian (cumulative fear)", color="blue")
plt.plot(nyt_fear_cum, label="NYT (cumulative fear)", color="orange", linestyle="--")
plt.xlabel("Paragraph Index")
plt.ylabel("Cumulative Fear Score")
plt.title("Cumulative Fear Over Paragraphs")
plt.legend()
plt.tight_layout()
plt.show()

# ---- ROLLING MEAN (5-paragraph window) ----
window_size = 5
guardian_fear_roll = guardian_emotion_series.rolling(window=window_size, min_periods=1).mean()
nyt_fear_roll = nyt_emotion_series.rolling(window=window_size, min_periods=1).mean()

plt.figure(figsize=(10, 5))
plt.plot(guardian_fear_roll, label=f"Guardian rolling mean (window={window_size})", color="blue")
plt.plot(nyt_fear_roll, label=f"NYT rolling mean (window={window_size})", color="orange", linestyle="--")
plt.xlabel("Paragraph Index")
plt.ylabel("Rolling Mean Fear Score")
plt.title("Rolling Mean Fear Over Paragraphs")
plt.legend()
plt.tight_layout()
plt.show()

# 9B. RADAR CHART OF SUMMED EMOTIONS (GUARDIAN VS. NYT)
#    We'll compare all columns in df_guardian_emotions and df_nyt_emotions

def create_radar_chart(categories, values_list, labels, title):
    """
    categories: list of emotion labels (e.g. ["anger", "fear", "joy", ...])
    values_list: list of lists, each sublist is the emotion sums for a dataset
    labels: which name to associate with each values_list entry
    title: chart title
    """
    # Number of variables
    N = len(categories)

    # Determine the angle of each axis in the plot
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    # Initialize radar chart
    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(8, 8))

    # Plot each dataset
    for idx, values in enumerate(values_list):
        # close the polygon by repeating the first value
        v = values.tolist() + values.tolist()[:1]
        ax.plot(angles, v, label=labels[idx])
        ax.fill(angles, v, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # Adjust radial limits to max of data (or manually set)
    ax.set_rlabel_position(0)
    ax.set_title(title, y=1.1)

    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()

if not df_guardian_emotions.empty or not df_nyt_emotions.empty:
    # Union of all emotions across both DataFrames
    all_emotions = set(df_guardian_emotions.columns).union(df_nyt_emotions.columns)
    # Convert to a sorted list for consistent ordering
    all_emotions = sorted(list(all_emotions))

    # Sum each emotion for Guardian
    guardian_sums = df_guardian_emotions.reindex(columns=all_emotions, fill_value=0).sum()
    # Sum each emotion for NYT
    nyt_sums = df_nyt_emotions.reindex(columns=all_emotions, fill_value=0).sum()

    # Create the radar chart
    create_radar_chart(
        categories=all_emotions,
        values_list=[guardian_sums, nyt_sums],
        labels=["Guardian", "NYT"],
        title="Radar Chart of Summed Emotions"
    )
