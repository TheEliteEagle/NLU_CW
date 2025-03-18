'''
    Solution 1: Unsupervised Learning Approach to Authorship Verification

    Step 1: Feature Extraction
    - Tokenize the text (use nlu_utils.get_df() to get total dataframe)
    - Many features to extract:
        - Function word usage (%age and distribution)
        - Punctuation usage (%age and distribution)
        - Complexity of punctuation (e.g. ; or ! compared to . and ,)
        - Determiner usage (%age and distribution)
            - DT:NN ratio
        - Average/Range of word lengths
        - Average/Range of sentence lengths
        - Capitalisation of words (for proper nouns and start of sentences)
        - Typographical errors (e.g. "teh" instead of "the", likely as a ratio)
        - Abbreviations (e.g. "LMFAO" and "LOL" etc.)
    - Use NLTK for tokenization, POS tagging, and sentence splitting
    - Use CountVectorizer to get the word frequencies
    - Use the above features to create a feature vector for each text

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
import sys

import numpy as np
import pandas as pd
import nltk
import string
import torch
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn

from nlu_utils import get_df, get_dataset, get_dataloader, get_feat_dataset

import spellchecker.spellchecker as spellchecker

nltk.download('punkt')
nltk.download("stopwords")

class BinaryClassifier(nn.Module):
    def __init__(self, input_size: int):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        # return self.sigmoid(x)
        # print(x)
        return x


def get_feature_vector(text: str, tensor: torch.Tensor, dense: bool = True, verbose: bool = False) -> np.ndarray:
    # Lexical features
    words = text.split()
    sents = nltk.sent_tokenize(text)

    feat_vec = np.ndarray(0, dtype=np.float32)

    # 1. Function word usage

    # 2. Punctuation usage
    all_puncts = string.punctuation
    simple_puncts = [",", ".", "!", "?", '"', "'"]
    punct_count = 0
    simple_punct_count = 0
    for word in words:
        if word in all_puncts:
            punct_count += 1
        if word in simple_puncts:
            simple_punct_count += 1

    complex_punct_count = punct_count - simple_punct_count
    if punct_count == 0:
        punct_complexity = 0
    else:
        punct_complexity = complex_punct_count / punct_count

    punct_dist = np.zeros(len(all_puncts))
    for char in all_puncts:
        punct_dist[all_puncts.index(char)] = text.count(char)

    if sum(punct_dist) == 0:
        punct_dist = np.zeros(len(all_puncts))
    else:
        punct_dist = punct_dist / sum(punct_dist)

    if verbose:
        print(f"Punctuation count: {punct_count}")
        print(f"Punctuation distribution: {punct_dist}")
        print(f"Punctuation complexoty: {punct_complexity}")

    feat_vec = np.append(feat_vec, punct_dist)
    feat_vec = np.append(feat_vec, punct_complexity)

    # 3. Determiner usage
    tagged = nltk.pos_tag(words, tagset="universal")
    det_count = 0
    nn_count = 0
    for word in tagged:
        if word[1] == "DET":
            det_count += 1
        if word[1] == "NOUN":
            nn_count += 1
    if nn_count == 0:
        dt_nn_ratio = 0
    else:
        dt_nn_ratio = det_count / nn_count

    if verbose:
        print(f"DT count: {det_count}")
        print(f"NN count: {nn_count}")
        print(f"DT:NN ratio: {dt_nn_ratio}")

    feat_vec = np.append(feat_vec, dt_nn_ratio)

    # 4. Range and quartiles of word lengths
    word_lengths = [len(word) for word in words]
    word_len_range = max(word_lengths) - min(word_lengths)
    # word_len_q1 = np.percentile(word_lengths, 25)
    word_len_q3 = np.percentile(word_lengths, 75)

    if verbose:
        print(f"Word length range: {word_len_range}")
        print(f"Word length Q3: {word_len_q3}")

    feat_vec = np.append(feat_vec, word_len_range)
    # feat_vec = np.append(feat_vec, word_len_q1)
    feat_vec = np.append(feat_vec, word_len_q3)

    # 5. Range and quartiles of sentence lengths
    sent_lengths = [len(nltk.word_tokenize(sent)) for sent in sents]
    sent_len_range = max(sent_lengths) - min(sent_lengths)
    # sent_len_q1 = np.percentile(sent_lengths, 25)
    sent_len_q3 = np.percentile(sent_lengths, 75)

    if verbose:
        print(f"Sentence length range: {sent_len_range}")
        print(f"Sentence length Q3: {sent_len_q3}")

    feat_vec = np.append(feat_vec, sent_len_range)
    # feat_vec = np.append(feat_vec, sent_len_q1)
    feat_vec = np.append(feat_vec, sent_len_q3)

    # 6. Capitalisation of words (Proper nouns and start of sentences)
    tagged = nltk.pos_tag(words)
    proper_nouns = [word for word in tagged if word[1] == "NNP"]
    correctly_capitalised_proper_nouns = 0
    for word in proper_nouns:
        if word[0][0].isupper():
            correctly_capitalised_proper_nouns += 1
    if len(proper_nouns) == 0:
        prop_noun_cap_ratio = 0
    else:
        prop_noun_cap_ratio = correctly_capitalised_proper_nouns / len(proper_nouns)

    correctly_capitalised_start_of_sent = 0
    for sent in sents:
        if sent[0][0].isupper():
            correctly_capitalised_start_of_sent += 1
    start_of_sent_cap_ratio = correctly_capitalised_start_of_sent / len(sents)

    if verbose:
        print(f"Proper noun capitalisation ratio: {prop_noun_cap_ratio}")
        print(f"Start of sentence capitalisation ratio: {start_of_sent_cap_ratio}")

    feat_vec = np.append(feat_vec, prop_noun_cap_ratio)
    feat_vec = np.append(feat_vec, start_of_sent_cap_ratio)

    # 7. Typographical errors
    spell = spellchecker.SpellChecker("en")
    misspelled = spell.unknown(words)
    typo_ratio = len(misspelled) / len(words)

    if verbose:
        print(f"Typo ratio: {typo_ratio}")

    feat_vec = np.append(feat_vec, typo_ratio)

    # 8. Abbreviations


    scaler = StandardScaler()
    feat_vec = scaler.fit_transform(feat_vec.reshape(-1, 1))

    if not dense:
        return feat_vec

    return feat_vec.reshape(1, -1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python solution_1.py <1 for creating vectors, 0 otherwise>")
        exit(1)

    feat_vec_size = 41

    create_vecs = int(sys.argv[1])
    if create_vecs:
        train_file = "Data/train.csv"
        # Get the dataset
        df = get_df(train_file)
        print(df.head())
        # Get dataloader
        dataset = get_dataset(train_file)
        dataloader = get_dataloader(dataset, 1, False)
        i = 0

        sims = []
        true_labels = []

        col_names = [f"fv_1_{i}" for i in range(feat_vec_size)] + [f"fv_2_{i}" for i in range(feat_vec_size)] + ["label"]

        feat_vec_df = pd.DataFrame(columns=col_names)

        for s1, s2, label in dataloader:
            if i % 500 == 0:
                print(f"Processing pair {i}/{len(df)}")
            s1_fv = get_feature_vector(df["text_1"][i], s1, True).flatten()
            s2_fv = get_feature_vector(df["text_2"][i], s2, True).flatten()

            row = np.concatenate((s1_fv, s2_fv, label.numpy()), axis=0)

            feat_vec_df = feat_vec_df._append(pd.DataFrame([row], columns=col_names), ignore_index=True)

            i += 1

        # Export to CSV for easy access later
        feat_vec_df.to_csv("feature_vectors.csv", index=False)

        feat_vec_df = pd.read_csv("feature_vectors.csv")
        feat_vec_train_ds, feat_vec_test_ds = get_feat_dataset(feat_vec_df)
    else:
        feat_vec_df = pd.read_csv("feature_vectors.csv")
        feat_vec_train_ds, feat_vec_test_ds = get_feat_dataset(feat_vec_df)

    print(f"train ds size: {len(feat_vec_train_ds)}")

    feat_vec_train_dl = get_dataloader(feat_vec_train_ds, 4, True)
    feat_vec_test_dl = get_dataloader(feat_vec_test_ds, 1, False)

    # Train model
    model = BinaryClassifier(feat_vec_size * 2)
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    print(f"train size: {len(feat_vec_train_dl)}")

    epochs = 50
    threshold = 0.5

    for i in range(epochs):
        print(f"Training Epoch {i}")
        loss_list = []
        for fv1, fv2, label in feat_vec_train_dl:
            optimizer.zero_grad()
            output = model(torch.cat((fv1, fv2), 1))
            loss = criterion(output, label)
            loss_list.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()

        print(f"Mean Loss: {np.mean(loss_list)}")
        loss_list = []
        preds = []
        true_labels = []

        print(f"Testing Epoch {i}")
        for fv1, fv2, label in feat_vec_test_dl:
            output = model(torch.cat((fv1, fv2), 1))
            loss = criterion(output, label)
            loss_list.append(loss.detach().numpy())
            # print(f"Output: {output}")
            preds.append(1 if output[0] > threshold else 0)
            true_labels.append(label.detach().numpy())

        # print(preds)
        correct = 0
        TPs = 0
        FPs = 0
        FNs = 0
        TNs = 0
        for i in range(len(true_labels)):
            if true_labels[i] == preds[i]:
                correct += 1
                if true_labels[i] == 1:
                    TPs += 1
                else:
                    TNs += 1
            else:
                if true_labels[i] == 1:
                    FNs += 1
                else:
                    FPs += 1
        accuracy = correct / len(true_labels)
        print(f"Accuracy: {accuracy}")
        print(f"Correct: {correct}")
        print(f"TPs: {TPs}")
        print(f"FPs: {FPs}")
        print(f"FNs: {FNs}")
        print(f"TNs: {TNs}")
        if TPs + FNs == 0:
            recall = 0
        else:
            recall = TPs / (TPs + FNs)
        print(f"Recall: {recall}")
        if TPs + FPs == 0:
            precision = 0
        else:
            precision = TPs / (TPs + FPs)
        print(f"Precision: {precision}")
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        print(f"F1-score: {f1}")
        print(f"Mean Loss: {np.mean(loss_list)}")


if __name__ == '__main__':
    main()