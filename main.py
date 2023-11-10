import pandas as pd
from transformers import pipeline

# Load song data CSV into dataframe
df = pd.read_csv('songData.csv')

## Define classifier
classifier = pipeline("zero-shot-classification")

# Create empty lists for storing results in columns
result_labels = []
result_scores = []

## Classify the lyrics and save the results
# Iterate through each row in dataframe
for index, row in df.iterrows():
    lyrics_to_classify = row['lyrics']

    # Classify the lyrics
    raw_results = classifier(lyrics_to_classify, candidate_labels=["misogyny", "non-misogyny"])

    # Extract labels and scores; append to results list to later save
    label = raw_results['labels'][0]
    score = raw_results['scores'][0]
    result_labels.append(label)
    result_scores.append(score)

## Create a CSV with the new classificaitons and metadata
# Add new columns to dataframe
df['label'] = result_labels
df['score'] = result_scores

# Save the updated DataFrame to a new CSV file
df.to_csv('songClassification_results.csv', index=False)