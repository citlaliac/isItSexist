import pandas as pd 
from transformers import pipeline


## Import and read CSV
df = pd.read_csv('songdata.csv') 

## Iterate through each row in CSV to run lyrics through transformer

## Use classifier and save reults into next row in CSV
classifier = pipeline("zero-shot-classification")
results = classifier("American woman, stay away from me American woman, mamma let me be Don't come hanging round my door I don't want to see your face no more I got more important things to do Than spend my time growing' old with you Now woman, stay away American woman, listen when I say American woman, get away from me American woman, mamma let me be Don't come knocking on my door I don't want to see your shadow no more Colored lights can hypnotize Sparkle someone else's eyes Now woman, get away American woman, listen when I say American woman, I said get away American woman, listen when I say Don't come hanging' round my door Don't want to see your face no more I don't need your warm machines I don't need your ghetto scenes Colored lights can hypnotize Sparkle someone else's eyes Now woman, get away American woman, listen when I say American woman, stay away from me American woman, mamma let me be I got to go, I got to get away Babe, I got to go I want to fly away I'm going to leave you woman I'm going to leave you woman I'm going to leave you woman I'm going to leave you woman Bye, bye Bye, bye Bye, bye Bye, bye American woman You're no good for me And I'm no good for you American woman I'm looking at you right in the eye Tell you what I'm going to do I'm going to leave you woman You know I got to go I'm going to leave you woman I got to go American woman I got to go I got to go American woman, yeah", candidate_labels=["misogyny", "non-misogyny"]),

# Print results (for inital testing)
print(results)

## Output determinaitons into new cell in data sheet