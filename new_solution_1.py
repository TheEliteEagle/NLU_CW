import nlu_utils

import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader

class TFIDFNLUDataset(Dataset):
    '''
    Dataset for the TF-IDF vectors
    '''

    VOCAB_SIZE = 10000
    SEQ_LEN = 1000

    def __init__(self, data: pd.DataFrame, transform=None):
        self.vocab = self.build_vocab(data)
        self.data = self.tf_idf_encode(data)
        pass

    def __len__(self) -> int:
        return len(self.data)

    def tf_idf_encode(self, data: pd.DataFrame):
        tf_df = self.get_tf(data)
        idf_df = self.get_idf(data)

        tf_idf_df = pd.DataFrame(columns=[[word + "_1" for word, _ in self.vocab.items()] + [word + "_2" for word, _ in self.vocab.items()]])
        for i in range(len(data)):
            for j in range(2):

        return tf_idf_df

    def get_tf(self, data: pd.DataFrame):
        tf_df = pd.DataFrame(columns=[word for word, _ in self.vocab.items()])
        for i in range(len(data)):
            for j in range(2):
                for word in data.iloc[i, j]:
                    if word in tf_df.columns:
                        tf_df.loc[2*i+j, word] += 1
        return tf_df

    def get_idf(self, data: pd.DataFrame):
        idf_dict = {word: 0 for word in self.vocab.keys()}
        for i in range(len(data)):
            for j in range(2):
                for word in set(data.iloc[i, j]):
                    if word in idf_dict:
                        idf_dict[word] += 1

        idf_df = pd.DataFrame(idf_dict.items(), columns=["Word", "IDF"])
        idf_df["IDF"] = np.log(len(data) / idf_df["IDF"])
        return idf_df

    def build_vocab(self, data):
        word_freq_dict = {}
        for i in range(len(data)):
            for j in range(2):
                for word in data.iloc[i, j]:
                    if word in word_freq_dict:
                        word_freq_dict[word] += 1
                    else:
                        word_freq_dict[word] = 1

        word_freq_list = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)

        if len(word_freq_list) > self.VOCAB_SIZE:
            word_freq_list = word_freq_list[:self.VOCAB_SIZE]

        vocab_dict = {word_freq_list[i][0]: i for i in range(len(word_freq_list))}

        vocab_dict["<PAD>"] = len(vocab_dict)
        vocab_dict["<UNK>"] = len(vocab_dict)
        vocab_dict["<SOS>"] = len(vocab_dict)
        vocab_dict["<EOS>"] = len(vocab_dict)

        return vocab_dict

data_path = "./Data/train.csv"
df = pd.read_csv(data_path)

ds = nlu_utils.get_clean_dataset(df, {"lemmatise", "lowercase", "stop words"})
dl = nlu_utils.get_dataloader(ds, 1, False)

# Calculate tf-idf for each document
tfidf = TFIDFNLUDataset(df)
