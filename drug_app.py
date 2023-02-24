# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
from sklearn.svm import SVC

# Load the dataset
df = pd.read_csv("drug_data_with_sentiment.csv")

# Clean the data
df = df.dropna()
df = df.drop_duplicates()
df["rating"] = df["rating"].astype(int)

# Create the feature matrix
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
reviews = vectorizer.fit_transform(df["review"])

# Train the model
model = SVC(kernel="linear")
model.fit(reviews, df["condition"])

# Write a function that returns the predicted condition and recommended drugs
def predict_condition(review):
    review = vectorizer.transform([review])
    condition = model.predict(review)[0]
    return condition
def recommend_drugs(condition):
    # Filter the drug dataset for the given condition
    condition_data = df[df['condition'] == condition]

    # Group the data by drug and calculate the average rating for each drug
    drug_ratings = condition_data.groupby('drugName')['rating'].mean().reset_index()

    # Sort the drugs by rating in descending order and return the top 3 drugs
    top_drugs = drug_ratings.sort_values(by='rating', ascending=False)['drugName'][:3].tolist()

    return top_drugs

# Write a Streamlit app that allows the user to enter a review and receive recommendations
st.title("Patient's Condition based on Drug Reviews and Recommend Drugs")
review = st.text_input("Enter a patient review:")

if st.button("Predict Condition"):
    condition = predict_condition(review)
    drugs = recommend_drugs(condition)
    st.write('The best 3 drugs for the "' +condition+ '" are:',drugs)
    for i, drug in enumerate(drugs):
        st.write(f'{i+1}. {drug}')




