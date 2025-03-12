'''
    Solution 1: Unsupervised Learning Approach to Authorship Verification

    Step 1: Feature Extraction
    - Tokenize the text (use nlu_utils.get_df() to get total dataframe)
    - Three types of features: lexical, syntactic, character-level
    - Lexical: word length distribution, sentence length distribution, vocabulary richness (type-token ratio)
    - Syntactic: POS tag distribution, function word usage, punctuation usage
    - Character-level: character n-grams, character distribution
    - Will result in a feature vector for each text (sparse or dense to be determined)

    Step 2: Similarity or Clustering
    - Use similarity metric to compare the feature vectors (naive approach) (cos_sim, Jensen-Shannon, etc.)
    - Use clustering to group similar texts together (based on feature vectors)
    - Dense vectors would be far better for clustering
    - Use KMeans or DBSCAN to cluster the texts (k to be determined)
    - DBSCAN is likely better for this task, as the number of authors is unknown
    - If the texts are clustered together, they are likely written by the same author

    Step 3: Evaluation
    - Use the dataset to evaluate the model
    - Compare the predicted authorship with the actual authorship
    - Use metrics like accuracy, precision, recall, F1-score, etc.
    - Use confusion matrix to visualize the results
'''

import numpy as np
import pandas as pd
import nltk
import string
import torch
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlu_utils import get_df, get_dataset, get_dataloader

def get_feature_vector(text: str, tensor: torch.Tensor, dense: bool = True) -> np.ndarray:
    # Lexical features
    words = text.split()
    sents = nltk.sent_tokenize(text)

    word_lengths = [len(word) for word in words]
    word_len_mean = np.mean(word_lengths)
    word_len_std = np.std(word_lengths)

    sentence_lengths = [len(sent.split()) for sent in sents]
    sent_len_mean = np.mean(sentence_lengths)
    sent_len_std = np.std(sentence_lengths)

    vocab_richness = len(set(words)) / len(words)
    print("Lexical features:")
    print(f"Word length mean: {word_len_mean}")
    print(f"Word length std: {word_len_std}")
    print(f"Sentence length mean: {sent_len_mean}")
    print(f"Sentence length std: {sent_len_std}")
    print(f"Vocabulary richness: {vocab_richness}")

    # Syntactic features
    pos_tags = nltk.pos_tag(words)
    pos_counts = Counter(tag for word, tag in pos_tags)
    pos_dist = [pos_counts[tag] / len(words) for tag in pos_counts]
    punctuation_dist = [text.count(p) / len(text) for p in string.punctuation]
    function_words = ["a", "an", "the", "and", "but", "or", "for", "nor", "so", "yet"]
    function_word_usage = sum(words.count(w) for w in function_words) / len(words)
    print("Syntactic features:")
    print(f"POS tag distribution: {pos_dist}")
    print(f"Punctuation distribution: {punctuation_dist}")
    print(f"Function word usage: {function_word_usage}")

    feat_vec = np.array([])
    feat_vec = np.append(feat_vec, word_len_mean)
    feat_vec = np.append(feat_vec, word_len_std)
    feat_vec = np.append(feat_vec, sent_len_mean)
    feat_vec = np.append(feat_vec, sent_len_std)
    feat_vec = np.append(feat_vec, vocab_richness)
    feat_vec = np.append(feat_vec, pos_dist)
    feat_vec = np.append(feat_vec, punctuation_dist)
    feat_vec = np.append(feat_vec, function_word_usage)

    return feat_vec

def main():
    train_file = "Data/train.csv"
    # Get the dataset
    df = get_df(train_file)
    print(df.head())
    # Get dataloader
    dataset = get_dataset(train_file)
    dataloader = get_dataloader(dataset, 1, False)
    i = 0
    for s1, s2, label in dataloader:
        s1_fv = get_feature_vector(df["text_1"][i], s1, True)
        s2_fv = get_feature_vector(df["text_2"][i], s2, True)
        print(s1_fv.shape)
        print(s2_fv.shape)
        print(cosine_similarity(s1_fv, s2_fv))
        i += 1
        break


if __name__ == '__main__':
    main()