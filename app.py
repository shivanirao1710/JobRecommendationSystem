from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import joblib

app = Flask(__name__)

# Secret key for session management (ensure to keep it secure)
app.secret_key = 'your_secret_key_here'

# Load trained model (TF-IDF and cosine similarity)
try:
    model_data = joblib.load("models/job_recommender.pkl")
    

    df = model_data[0]  # The DataFrame
    cosine_sim = model_data[1]  # The cosine similarity matrix
    print("Columns in the DataFrame:", df.columns)  # Print columns to debug
except Exception as e:
    print(f"Error loading model: {e}")
    df, cosine_sim = None, None

# Function to find job roles based on skills
def find_job_roles_by_skills(skills, top_n=5):
    if df is None:
        return []

    skills = skills.lower().split(",")  # Split skills by comma
    skills = [skill.strip() for skill in skills]  # Remove extra spaces

    # Debug: Print the skills being searched
    print(f"Searching for jobs with skills: {skills}")

    # Check the column names in the DataFrame
    print("Columns in DataFrame:", df.columns)

    # If 'skills_cleaned' is not found, you can replace it with the correct column name
    matched_jobs = df[df["skills_cleaned"].apply(lambda x: any(skill in x.lower() for skill in skills))]

    # Sort by the number of matched skills
    matched_jobs["matched_skills"] = matched_jobs["skills_cleaned"].apply(lambda x: sum(skill in x.lower() for skill in skills))
    matched_jobs = matched_jobs.sort_values(by="matched_skills", ascending=False).head(top_n)

    # Debug: Print the matched jobs
    print(f"Matched jobs: {matched_jobs[['job_role', 'company_name']]}")

    return matched_jobs[["job_role", "company_name", "company_type", "skills_cleaned"]].to_dict(orient="records")

# Function to find job roles based on job role name
def find_job_roles_by_job_role(job_role, top_n=5):
    if df is None:
        return []

    # Clean the search query to remove extra spaces
    job_role = job_role.strip()

    # Debug: Print the job role being searched
    print(f"Searching for job role: '{job_role}'")

    # Filter jobs that match the job role name (case-insensitive)
    matched_jobs = df[df["job_role"].str.contains(job_role, case=False, na=False)]

    # Debug: Print the number of matched jobs and their titles
    print(f"Number of matched jobs: {len(matched_jobs)}")
    print(f"Matched job roles: {matched_jobs[['job_role', 'company_name']].head()}")

    return matched_jobs[["job_role", "company_name", "company_type", "skills_cleaned"]].to_dict(orient="records")

# Home Route
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    search_type = None
    if request.method == "POST":
        search_type = request.form.get("search_type", "skills")  # Default to "skills" if not provided
        query = request.form.get("skills", "").strip()  # Get skills/job role from the form
        
        if not query:
            # If no skills/job role were provided, show an error or a message to the user
            return render_template("index.html", recommendations=[], error="Please enter skills or job role.")
        
        if search_type == "skills":
            recommendations = find_job_roles_by_skills(query)  # Get recommendations based on skills
        elif search_type == "job_role":
            recommendations = find_job_roles_by_job_role(query)  # Get recommendations based on job role

        # Store the recommendations and search type in session
        session['search_type'] = search_type
        session['recommendations'] = recommendations
        session['query'] = query  # Store the query for future use

    # If there are recommendations stored in session, use them
    if 'recommendations' in session:
        recommendations = session['recommendations']
        search_type = session['search_type']
    
    return render_template("index.html", recommendations=recommendations, search_type=search_type)

# Job Details Route
@app.route("/job/<job_role>/<company_name>")
def job_details(job_role, company_name):
    if df is None:
        return render_template("job_details.html", error="Model data not found.")
    
    # Find the job by job_role and company_name
    job_details = df[(df["job_role"] == job_role) & (df["company_name"] == company_name)].iloc[0]
    
    return render_template("job_details.html", job=job_details)

# Clear session on logout or reset
@app.route("/reset")
def reset():
    session.pop('recommendations', None)
    session.pop('search_type', None)
    session.pop('query', None)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
