import mysql.connector
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

# Connect to MySQL to fetch cleaned data
conn = mysql.connector.connect(
    host="localhost",
    user="root",  # Your MySQL username
    password="shivanirao1710",  # Your MySQL password
    database="jobtaxonomy"  # Your database name
)

# Fetch cleaned data from job_data__cleaned
query = "SELECT job_role, company_name, company_type, knowledge_cleaned, skills_cleaned, combined_features FROM job_data_cleaned"
df = pd.read_sql(query, con=conn)
conn.close()

# TF-IDF Vectorization on combined_features
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["combined_features"])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save the model and vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump((df, cosine_sim, vectorizer), "models/job_recommender.pkl")

print("✅ Model training complete. File saved as 'models/job_recommender.pkl'.")
