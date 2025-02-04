import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import os

# Download NLTK stopwords if not present
nltk.download('stopwords')

# Load dataset (handle encoding issues)
df = pd.read_csv("dataset/job_data.csv", encoding="ISO-8859-1")

# Fill missing values with empty strings
df.fillna("", inplace=True)

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])  # Remove stopwords
    return text

# Apply cleaning to relevant columns
df["knowledge_cleaned"] = df["knowledge"].apply(clean_text)
df["skills_cleaned"] = df["skills"].apply(clean_text)

# Combine important features
df["combined_features"] = df["job_role"] + " " + df["company_type"] + " " + df["knowledge_cleaned"] + " " + df["skills_cleaned"]

# Save cleaned dataset
os.makedirs("dataset", exist_ok=True)
df.to_csv("dataset/cleaned_data.csv", index=False)

print("✅ Data preprocessing complete. File saved as 'dataset/cleaned_data.csv'.")
