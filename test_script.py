import nlu_utils
import pandas as pd

data_path = "./Data/train.csv"
df = pd.read_csv(data_path)
text_one = df.iloc[:, 0]
text_two = df.iloc[:, 1]
labels = df.iloc[:, 2]

text_one_avg_len = 0
text_two_avg_len = 0
text_one_max_len = 0
text_two_max_len = 0
for i in range(len(text_one)):
    text_one_avg_len += len(text_one.iloc[i])
    text_two_avg_len += len(text_two.iloc[i])
    if len(text_one.iloc[i]) > text_one_max_len:
        text_one_max_len = len(text_one.iloc[i])
    if len(text_two.iloc[i]) > text_two_max_len:
        text_two_max_len = len(text_two.iloc[i])

text_one_avg_len /= len(text_one)
text_two_avg_len /= len(text_two)

print(text_one_avg_len, text_two_avg_len)
print(text_one_max_len, text_two_max_len)