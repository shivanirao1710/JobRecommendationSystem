import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_data.csv")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["combined_features"])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save the model
os.makedirs("models", exist_ok=True)
joblib.dump((df, cosine_sim), "models/job_recommender.pkl")

print("✅ Model training complete. File saved as 'models/job_recommender.pkl'.")
