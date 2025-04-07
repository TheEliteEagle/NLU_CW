import nlu_utils
import pandas as pd
import numpy as np
from scipy.stats import zscore

from collections import Counter

def fetch_dataset() -> nlu_utils.UnencodedNLUDataset:
    df = pd.read_csv("./Data/train.csv")

    params = {
        "lemmatise",
        "lowercase"
    }

    print(params)

    return nlu_utils.get_clean_dataset(df, params)

def get_word_freq(text, vocabulary):
    word_counts = Counter(text)
    return np.array([word_counts[word] if word in word_counts else 0 for word in vocabulary])

def burrows_delta(text1, text2, vocabulary):
    freq1 = get_word_freq(text1, vocabulary)
    freq2 = get_word_freq(text2, vocabulary)

    print(f"Freq1: {freq1}, Freq2: {freq2}")

    combined = np.vstack([freq1, freq2])
    z_scores = zscore(combined, axis=0, ddof=1)

    print(f"Z Scores: {z_scores}")

    delta = np.mean(np.abs(z_scores[0] - z_scores[1]))
    return delta

def classify_authorship(text1, text2, vocabulary, threshold=0.5):
    delta = burrows_delta(text1, text2, vocabulary)
    return delta < threshold

ds = fetch_dataset()
top_500_vocab = list(ds.vocabulary)

df = pd.read_csv("./Data/train.csv")
dl = nlu_utils.get_dataloader(ds, 1, False)

for s1, s2, label in dl:
    s1 = s1
    s2 = s2
    label = label

    print(s1)
    print(s2)
    print(label)

    print(burrows_delta(s1, s2, top_500_vocab))
    print(classify_authorship(s1, s2, top_500_vocab))
    break

# Print the sentence that became size 0

