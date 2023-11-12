
## Goals
The goal of this project is to look at popular music over the last few decased and use AI to determine if the songs are sexist against women. With that data I plan to look for trends by over time by genre (loosely, genre is subjective), and artist gender (whether or not any of the song's artists are women).

My goal is to use AI and transformers to do this. 

This is an updated version of earlier projects from 2018: [misoBot](https://github.com/citlaliac/misoBot) and [Music Sentiment Analysis](https://github.com/citlaliac/SentimentAnalysisOfMusic)

## How to run it!
1. Download the files locally
2. Ensure Docker is installed locally
3. Run `docker build -t isItSexist_image:latest .` in terminal
4. Run `docker run -it isItSexist_image:latest ` in terminal, and see results print!