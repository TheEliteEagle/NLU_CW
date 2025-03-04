import pandas as pd

def readFile(path="Data/AV_trial.csv"): #TODO change default path as data is released
    df = pd.read_csv(path, index_col=False)
    return df