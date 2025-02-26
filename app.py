from flask import Flask, render_template, request, redirect, url_for, session
import mysql.connector
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# MySQL database connection details
db_config = {
    'host': 'localhost',
    'user': 'root',  # Your MySQL username
    'password': 'shivanirao1710',  # Your MySQL password
    'database': 'jobtaxonomy'  # Your database name
}

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to get job data from MySQL
def get_job_data_from_mysql():
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)
        
        cursor.execute("SELECT job_role, company_name, company_type, knowledge_cleaned, skills_cleaned, combined_features, embedding FROM job_data_cleaned")
        job_data = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return job_data
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return []

# Function to find job roles by skills
def find_job_roles_by_skills(skills, top_n=5):
    skills = skills.lower().split(",")
    skills = [skill.strip() for skill in skills]  # Remove extra spaces
    query = " ".join(skills)
    
    # Get embedding for the user's query
    query_embedding = model.encode([query])[0]
    
    # Fetch job data from MySQL
    job_data = get_job_data_from_mysql()
    
    # Calculate cosine similarities
    cosine_similarities = []
    for job in job_data:
        embedding = job.get('embedding', '')
        if embedding:
            try:
                embedding = json.loads(embedding)  # Convert JSON string back to list
                cosine_sim = cosine_similarity([query_embedding], [embedding])[0][0]
                cosine_similarities.append((job, cosine_sim))
            except json.JSONDecodeError:
                print(f"Error decoding embedding for job {job['job_role']} at {job['company_name']}")
                continue
        else:
            continue
    
    # Sort by similarity and return the top N jobs
    cosine_similarities.sort(key=lambda x: x[1], reverse=True)
    recommended_jobs = [job for job, sim in cosine_similarities[:top_n]]
    
    return recommended_jobs

# Function to find job roles by job role name
def find_job_roles_by_job_role(job_role, top_n=5):
    job_role = job_role.lower().strip()
    
    # Get embedding for the user's job role query
    query_embedding = model.encode([job_role])[0]
    
    # Fetch job data from MySQL
    job_data = get_job_data_from_mysql()
    
    # Calculate cosine similarities
    cosine_similarities = []
    for job in job_data:
        embedding = job.get('embedding', '')
        if embedding:
            try:
                embedding = json.loads(embedding)  # Convert JSON string back to list
                cosine_sim = cosine_similarity([query_embedding], [embedding])[0][0]
                cosine_similarities.append((job, cosine_sim))
            except json.JSONDecodeError:
                print(f"Error decoding embedding for job {job['job_role']} at {job['company_name']}")
                continue
        else:
            continue
    
    # Sort by similarity and return the top N jobs
    cosine_similarities.sort(key=lambda x: x[1], reverse=True)
    recommended_jobs = [job for job, sim in cosine_similarities[:top_n]]
    
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
    job_data = get_job_data_from_mysql()
    
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
