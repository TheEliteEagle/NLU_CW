'''
    The main file for common functions between the two solutions. This comprises mainly the loading, transformation,
    and conversion of the data (to a PyTorch DataLoader/Dataset)
'''

import pandas as pd
import nltk
import torch
from torch.utils.data import Dataset, DataLoader

class NLUDataset(Dataset):
    VOCAB_SIZE = 10000  # Can adjust these variables whenever - these are placeholders
    SEQ_LEN = 128

    def __init__(self, csv_file: str, transform=None, preprocessing_params:set = None):
        '''
        Initialises the dataset
        :param csv_file:
        :param transform: (not used at the moment, there for later)

        Program flow:
            Read data csv
            Preprocess data (lemmatisation etc.)
            Build the vocabulary (see above for max VOCAB_SIZE)
            Encode the data (convert words to indices, pad to SEQ_LEN, removes end if too long)
        '''
        data = pd.read_csv(csv_file)
        # Apply preprocessing to columns 0 and 1
        data = data.apply(lambda x: self.preprocess_row(x, preprocessing_params), axis=1)
        self.vocabulary = self.build_vocab(data)
        self.encodings = self.encode_data(data)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Returns the data at the given index
        :param idx:
        :return: s1, s2, label

        s1: torch.Tensor
            The first sentence
        s2: torch.Tensor
            The second sentence
        label: torch.Tensor
            The label

        Can iterate over the dataloader using:
        for s1, s2, l in dataloader:
            ...
        '''
        sentence_1 = self.encodings[idx]["text_1"]
        sentence_2 = self.encodings[idx]["text_2"]
        label = self.encodings[idx]["label"]
        return sentence_1, sentence_2, label

    def preprocess_row(self, row: pd.Series, preprocessing_params:set) -> pd.Series:
        '''
        Preprocesses a row of the dataframe
        :param row:
        :return pd.Series for new row:
        '''
        sentence_1 = row["text_1"]
        sentence_2 = row["text_2"]
        label = row["label"]

        def preprocess_sent(sent: str) -> list[str]:

            # call external functions
            processed_tokens = preprocess_line(sent, preprocessing_params)
            return processed_tokens

        return pd.Series([preprocess_sent(sentence_1), preprocess_sent(sentence_2), label])

    def build_vocab(self, data) -> dict:
        '''
        Builds the vocabulary from the data
        :param data:
        :return dict:

        Program flow:
            Creates a word freq dict (for easy lookup)
            Sorts the dict by frequency
            If the dict is too large, truncates it
            Adds special tokens for padding, unknown words, start of sentence, and end of sentence
                N.B. <SOS> and <EOS> may not be used yet but are there in case of future use
            Returns the new dictionary, with words ordered by frequency
        '''
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

        # Add special tokens
        vocab_dict["<PAD>"] = len(vocab_dict)
        vocab_dict["<UNK>"] = len(vocab_dict)
        vocab_dict["<SOS>"] = len(vocab_dict)
        vocab_dict["<EOS>"] = len(vocab_dict)

        return vocab_dict

    def encode_data(self, data: pd.DataFrame) -> list[dict[str, torch.Tensor]]:
        '''
        Encodes the data to convert words to indices
        :param data:
        :return list of dicts containing encoded data:

        Program flow:
            for each row in the data:
                encode the first sentence
                encode the second sentence
                convert the label to a tensor
                append the new dict to the list
            return the list
        '''
        new_ds = []

        # print(self.vocabulary)

        def encode_sent(sent: list[str]) -> torch.Tensor:
            '''
            Encodes a sentence
            :param sent:
            :return torch.Tensor:

            Program flow:
                For each word in the sentence:
                    Get the index of the word in the vocabulary (if not found, use <UNK>)
                    Append the index to the encoded list
                If the encoded list is too short, pad it
                If the encoded list is too long, remove the end
                Convert the list to a tensor
                Return the tensor
            '''
            encoded = [self.vocabulary.get(word, self.vocabulary["<UNK>"]) for word in sent]
            while len(encoded) < self.SEQ_LEN:
                encoded.append(self.vocabulary["<PAD>"])
            if len(encoded) > self.SEQ_LEN:
                encoded = encoded[:self.SEQ_LEN]
            encoded_tensor = torch.tensor(encoded, dtype=torch.float)
            return encoded_tensor

        for i in range(len(data)):
            encoded_1 = encode_sent(data.iloc[i, 0])
            encoded_2 = encode_sent(data.iloc[i, 1])
            label = torch.tensor(data.iloc[i, 2], dtype=torch.float)

            new_ds.append({
                "text_1": encoded_1,
                "text_2": encoded_2,
                "label": label
            })

        return new_ds

def get_dataset(csv_file: str, preprocessing_params:set = None) -> NLUDataset:
    return NLUDataset(csv_file, preprocessing_params=preprocessing_params)

def get_dataloader(dataset: NLUDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_df(csv_file: str) -> pd.DataFrame:
    return pd.read_csv(csv_file)

def preprocess_line(line: str, params: set) -> list[str]:
    
    if "trim email" in params: #stub code for now
        line = detect_and_trim_emails(line)
    
    tokens = nltk.tokenize.word_tokenize(line)
    operations = {
        "stop words": lambda tokens: [token for token in tokens if token.lower() not in nltk.corpus.stopwords.words('english')],
        "stem": lambda tokens: [nltk.PorterStemmer().stem(token) for token in tokens],
        "lemmatise": lambda tokens: [nltk.WordNetLemmatizer().lemmatize(token) for token in tokens],
        "alphanumeric": lambda tokens: [token for token in tokens if token.isalnum()],
        "lowercase": lambda tokens: [token.lower() for token in tokens]
    }

    # Apply each operation if we define it in the params set
    for key, action in operations.items():
        if key in params:
            tokens = action(tokens)
    
    return tokens

def detect_and_trim_emails(line: str):
    
    '''
    If line contains an email, trims out the email header
    some emails have text before the email starts e.g "look at this  ----- Forwarded by..." with variable numbers of -
    some emails cut off before reaching the subject
    '''

    if "-- Forwarded by" in line:
        before_email = line.split("-")[0].strip()
        email = "-" + line.split("-", 1)[1] if "-" in line else ""
        email_subject = email.split("Subject:")[-1].strip() # if theres no subject, this keeps the whole email
        line = before_email + " " + email_subject
        line = line.strip()
    return line
