import streamlit as st 

import joblib 

import nltk 

from nltk.sentiment.vader import SentimentIntensityAnalyzer 

 

Load sentiment analyzer 

nltk.download('vader_lexicon') 

analyzer = SentimentIntensityAnalyzer() 

 

App title 

st.title("Sentiment Analysis App") 

 

User input 

user_input = st.text_input("Enter your text here:") 

if st.button("Analyze Sentiment"): 

    if user_input: 

        score = analyzer.polarity_scores(user_input) 

        sentiment = max(score, key=score.get) 

        st.write(f"*Predicted Sentiment:* {sentiment.capitalize()}") 

        st.write(f"*Confidence Scores:* {score}") 

 
