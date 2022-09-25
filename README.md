# Project name
Twitter User Analysis Tool
## Team Members
- Shivansh Sethi
- Rohit

## Tool Description

The Main purpose of the tool is for users to analyse user behaviour on recent tweets.

## Installation

### As this repo contains some git lfs objects and the git lfs bandwidth is limited you have to follow the below steps to doenload the repo.

1. Fork the Following Repository  https://github.com/shivansh-sethi-2000/Twitter-User-Analysis.git

2. Download the cardiffnlp and universal-sentence-encoder_4 folders from the link : https://drive.google.com/drive/folders/1mOe2WVAit0AakFINY3k1iaVLoP4ExO8n?usp=sharing and place the cardiffnlp and universal-sentence-encoder_4 folders in the same directory.

3. Move to the tool's directory and install the required Packages

        pip install -r req.txt

4. add your API keys and sceret to my_tokens.py file

## Usage
1. To run the Script use the following command

        streamlit run User_Script.py

2. It will Take you to your Web Page If not you can use the Url showing in your terminal

3. The HomePage will Look Like this
    ![What is this](images/main.png)

## Additional Information
- Next Steps would include improving network graphs of Users for better analysis and image tweet analysis.
- there are some restrictions in using twitter API and same are applied here. Also, the text pre processing might take a liitle time for medium-large datasets.
- the network graph will currently contain only latest 500 connections rather than all of them.
