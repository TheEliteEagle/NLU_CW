import nlu_utils

data_path = "./Data/train.csv"
ds = nlu_utils.get_dataset(data_path)
dl = nlu_utils.get_dataloader(ds, batch_size=1, shuffle=False)

for s1, s2, label in dl:
    print(s1)
    print(s2)
    print(label)
    break