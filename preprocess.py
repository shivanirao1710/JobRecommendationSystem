import mysql.connector
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re

# Download NLTK stopwords if not present
nltk.download('stopwords')

# Connect to MySQL database
conn = mysql.connector.connect(
    host="localhost", 
    user="root", 
    password="shivanirao1710", 
    database="jobtaxonomy"
)

# Fetch data from jobroles table
query = "SELECT job_role, company_name, company_type, knowledge, skills FROM jobroles"
df = pd.read_sql(query, con=conn)
conn.close()

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

# Insert cleaned data into jobroles_cleaned table
conn = mysql.connector.connect(
    host="localhost", 
    user="root", 
    password="shivanirao1710", 
    database="jobtaxonomy"
)

cursor = conn.cursor()

# Insert data into job_data_cleaned table
for _, row in df.iterrows():
    cursor.execute("""
        INSERT INTO job_data_cleaned (job_role, company_name, company_type, knowledge_cleaned, skills_cleaned, combined_features)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (row['job_role'], row['company_name'], row['company_type'], row['knowledge_cleaned'], row['skills_cleaned'], row['combined_features']))

conn.commit()
conn.close()

print("✅ Data preprocessing and insertion complete.")
