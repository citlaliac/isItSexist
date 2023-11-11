import pandas as pd
from transformers import pipeline

# Load song data CSV into dataframe
df = pd.read_csv('songDataNew.csv')

## Define classifier
classifier = pipeline("zero-shot-classification")

# Create empty lists for storing results in columns
result_labels = []
result_scores = []

## Classify the lyrics and save the results
# Iterate through each row in dataframe
for index, row in df.iterrows():
    try:
        lyrics_to_classify = row['lyrics']

        # Classify the lyrics
        # TODO consider better candidate labels
        # TODO Keep taking this further; consider more tuning
        raw_results = classifier(lyrics_to_classify, candidate_labels=["misogyny", "non-misogyny"])

        # Extract labels and scores; append to results list to later save
        labels = raw_results['labels'][0]
        scores = raw_results['scores'][0]

        # Append labels and scores to result lists
        result_labels.append(labels)
        result_scores.append(scores)
        if index % 100 == 0:
            print(index)
    except Exception as e:
        print(f"Error processing row {index}: {e} for classification")
        # If an error occurs, add a placeholder value to the result lists
        result_labels.append("error")
        result_scores.append(-1)


# Check lengths
print("Length of result_labels:", len(result_labels))
print("Length of result_scores:", len(result_scores))
print("Length of result_scores:", result_scores[2])
print("Length of DataFrame index:", len(df))
result_df = pd.DataFrame(index=range(len(df)))


## Create a CSV with the new classificaitons and metadata
# Add new columns to dataframe

# result_df['labels'] = result_labels
# result_df['scores'] = result_scores
df['labels'] = result_labels
df['scores'] = result_scores

# Save the updated DataFrame to a new CSV file
df.to_csv('songClassification_results.csv', index=False)