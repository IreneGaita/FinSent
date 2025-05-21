import pandas as pd

df = pd.read_csv("sentiment_data/all-dataset.csv", encoding='latin1', names=['label', 'text'])
print(df)
