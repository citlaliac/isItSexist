import pandas as pd 
from transformers import pipeline

## Define classifier type
classifier = pipeline("zero-shot-classification")

## Variables
allResults= []

## Import and read CSV
df = pd.read_csv('songData.csv') 
column_name = "lyrics"

# TODO create as own more reusable method
## Function to read and classify from CSV
# For each lyrics in sheet classify the lyrics and add to results list
for lyricsToClassify in df[column_name]:
    # Use transformer to classify lyrics
    # TODO consider better candidate labels
    # TODO Keep taking this further; consider more tuning
    results = classifier(lyricsToClassify, candidate_labels=["misogyny", "non-misogyny"])
    allResults.append([lyricsToClassify, results])
    print(results)


# TODO create as own more reusable method
## Save data to CSV
# Save results of lyric classifcation to a data frame
newDf = pd.DataFrame(allResults)

# Append existing data fram with dataframe containing classifications
# TODO update headers of csv to not be 0, 1
for col in df.columns:
    if col != column_name:
        newDf[col] = df[col]

# Save dataframe containing all relevant classifications and metadata to CSV called 'determinations.csv' (in docker*)
newDf.to_csv('output.csv', index=False)


