import pandas as pd 
from transformers import pipeline

## Import and read CSV
df = pd.read_csv('songdata.csv') 

# Create empty list to store the lyrics
lyrics_list = []

# Iterate through each row and extract the lyrics
for index, row in df.iterrows():
    lyrics = row['lyrics'] 
    lyrics_list.append(lyrics)



## Classify text in row

classifier = pipeline("zero-shot-classification")

res = classifier("women are less than men tbh", candidate_labels=["misogyny", "empowering", "friendly"]),

print(res)

## Output determinaitons into new cell in data sheet