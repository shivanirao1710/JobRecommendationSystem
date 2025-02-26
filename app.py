import psycopg2
from psycopg2.extras import DictCursor  # Import DictCursor for dictionary-based results
import json
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request, redirect, url_for, session
import faiss
import numpy as np

# PostgreSQL database connection details
db_config = {
    'host': 'localhost',  # Change if necessary
    'user': 'postgres',  # Replace with your PostgreSQL username
    'password': 'shivanirao1710',  # Replace with your PostgreSQL password
    'database': 'jobtaxonomy'  # Replace with your database name
}

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Function to get job data from the job_data_cleaned table (preprocessed data)
def get_job_data_from_postgresql():
    try:
        connection = psycopg2.connect(**db_config)
        # Use DictCursor to get results as a dictionary
        cursor = connection.cursor(cursor_factory=DictCursor)
        
        # Query to fetch data from job_data_cleaned table (preprocessed data)
        cursor.execute("SELECT job_role, company_name, company_type, knowledge_cleaned, skills_cleaned, combined_features, embedding FROM job_data_cleaned")
        job_data = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return job_data
    except psycopg2.Error as err:
        print(f"Error: {err}")
        return []

# Function to prepare FAISS index for semantic search
def prepare_faiss_index(job_data):
    job_embeddings = []
    job_titles = []
    for job in job_data:
        embedding = job.get('embedding', '')
        if embedding:
            try:
                embedding = json.loads(embedding)  # Convert JSON string back to list
                job_embeddings.append(embedding)
                job_titles.append(job['job_role'])
            except json.JSONDecodeError:
                print(f"Error decoding embedding for job {job['job_role']} at {job['company_name']}")
                continue
    
    # Convert embeddings to numpy array (needed for FAISS)
    job_embeddings = np.array(job_embeddings).astype('float32')

    # Create FAISS index
    faiss_index = faiss.IndexFlatL2(job_embeddings.shape[1])  # L2 distance for similarity search
    faiss_index.add(job_embeddings)  # Add embeddings to the index

    return faiss_index, job_titles

# Function to find job roles by skills using FAISS for semantic search
def find_job_roles_by_skills(skills, top_n=5):
    skills = skills.lower().split(",")
    skills = [skill.strip() for skill in skills]  # Remove extra spaces
    query = " ".join(skills)
    
    # Get embedding for the user's query
    query_embedding = model.encode([query])[0]
    
    # Fetch job data from PostgreSQL
    job_data = get_job_data_from_postgresql()

    # Prepare FAISS index
    faiss_index, job_titles = prepare_faiss_index(job_data)
    
    # Perform the search using FAISS
    query_embedding = np.array(query_embedding).reshape(1, -1).astype('float32')
    distances, indices = faiss_index.search(query_embedding, top_n)

    # Get the top recommended jobs based on the FAISS search results
    recommended_jobs = []
    for index in indices[0]:
        job = next((job for i, job in enumerate(job_data) if job_titles[i] == job_titles[index]), None)
        if job:
            recommended_jobs.append(job)
    
    return recommended_jobs

# Function to find job roles by job role name using FAISS for semantic search
def find_job_roles_by_job_role(job_role, top_n=5):
    job_role = job_role.lower().strip()
    
    # Get embedding for the user's job role query
    query_embedding = model.encode([job_role])[0]
    
    # Fetch job data from PostgreSQL
    job_data = get_job_data_from_postgresql()

    # Prepare FAISS index
    faiss_index, job_titles = prepare_faiss_index(job_data)
    
    # Perform the search using FAISS
    query_embedding = np.array(query_embedding).reshape(1, -1).astype('float32')
    distances, indices = faiss_index.search(query_embedding, top_n)

    # Get the top recommended jobs based on the FAISS search results
    recommended_jobs = []
    for index in indices[0]:
        job = next((job for i, job in enumerate(job_data) if job_titles[i] == job_titles[index]), None)
        if job:
            recommended_jobs.append(job)
    
    return recommended_jobs

# Home Route for search functionality
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    search_type = None
    if request.method == "POST":
        search_type = request.form.get("search_type", "skills")
        query = request.form.get("skills", "").strip()
        
        if not query:
            return render_template("index.html", recommendations=[], error="Please enter skills or job role.")
        
        if search_type == "skills":
            recommendations = find_job_roles_by_skills(query)
        elif search_type == "job_role":
            recommendations = find_job_roles_by_job_role(query)
        
        # Store the recommendations and search type in session
        session['search_type'] = search_type
        session['recommendations'] = recommendations
        session['query'] = query
    
    # If there are recommendations stored in session, use them
    if 'recommendations' in session:
        recommendations = session['recommendations']
        search_type = session['search_type']
    
    return render_template("index.html", recommendations=recommendations, search_type=search_type)

# Job Details Route (for viewing specific job details)
@app.route("/job/<job_role>/<company_name>")
def job_details(job_role, company_name):
    job_data = get_job_data_from_postgresql()
    
    # Look for the specific job with matching job_role and company_name
    job_details = next((job for job in job_data if job['job_role'] == job_role and job['company_name'] == company_name), None)
    
    # If the job is found, render the job details page
    if job_details:
        return render_template("job_details.html", job=job_details)
    else:
        # If the job isn't found, show an error message
        return render_template("job_details.html", error="Job not found.")

# Reset Route to clear session data
@app.route("/reset")
def reset():
    session.pop('recommendations', None)
    session.pop('search_type', None)
    session.pop('query', None)
    return redirect(url_for('index'))

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)
