import pandas as pd

df = pd.read_csv("sentiment_data/all-data.csv", encoding='latin1', names=['label', 'text'])
print(df)
