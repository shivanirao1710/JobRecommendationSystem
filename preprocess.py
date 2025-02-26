import mysql.connector
import re
import string
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# MySQL database connection details
db_config = {
    'host': 'localhost',
    'user': 'root',  # Replace with your MySQL username
    'password': 'shivanirao1710',  # Replace with your MySQL password
    'database': 'jobtaxonomy'  # Replace with your database name
}

# Initialize SentenceTransformer model (you can replace this with a different model)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to clean text (remove punctuation, lowercasing, etc.)
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    return text.strip()

# Function to get job data from the jobroles table
def get_job_data_from_mysql():
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)
    
    # Query to fetch data from jobroles table
    cursor.execute("SELECT job_role, company_name, company_type, knowledge, skills FROM jobroles")
    job_data = cursor.fetchall()
    
    cursor.close()
    connection.close()
    
    return job_data

# Function to save cleaned job data to job_data_cleaned table
def save_cleaned_data_to_mysql(cleaned_data):
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    
    # Insert cleaned data into the job_data_cleaned table
    for job in cleaned_data:
        cursor.execute("""
            INSERT INTO job_data_cleaned (job_role, company_name, company_type, knowledge_cleaned, skills_cleaned, combined_features, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (job['job_role'], job['company_name'], job['company_type'], job['knowledge_cleaned'], job['skills_cleaned'], job['combined_features'], job['embedding']))
    
    connection.commit()
    cursor.close()
    connection.close()

# Function to preprocess job data and generate the cleaned data
def preprocess_data():
    # Step 1: Get job data from MySQL
    job_data = get_job_data_from_mysql()

    # Step 2: Preprocess the job data
    cleaned_data = []
    for job in job_data:
        job_role = job['job_role']
        company_name = job['company_name']
        company_type = job['company_type']
        
        # Clean knowledge and skills text
        knowledge_cleaned = clean_text(job['knowledge'])
        skills_cleaned = clean_text(job['skills'])
        
        # Combine knowledge and skills
        combined_features = knowledge_cleaned + " " + skills_cleaned
        
        # Generate embedding using the SentenceTransformer model
        embedding = model.encode([combined_features])[0]  # Get the embedding vector
        
        # Convert embedding (list) to JSON string
        embedding_json = json.dumps(embedding.tolist())  # Serialize the embedding list to JSON
        
        # Store the cleaned data in a dictionary
        cleaned_data.append({
            'job_role': job_role,
            'company_name': company_name,
            'company_type': company_type,
            'knowledge_cleaned': knowledge_cleaned,
            'skills_cleaned': skills_cleaned,
            'combined_features': combined_features,
            'embedding': embedding_json  # Store as JSON string
        })
    
    # Step 3: Save the cleaned data to MySQL job_data_cleaned table
    save_cleaned_data_to_mysql(cleaned_data)

    print(f"Processed {len(cleaned_data)} job entries and saved them to the database.")

# Run the preprocessing
if __name__ == "__main__":
    preprocess_data()
