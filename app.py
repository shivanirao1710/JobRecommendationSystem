from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import joblib

app = Flask(__name__)

# Secret key for session management (ensure to keep it secure)
app.secret_key = 'your_secret_key_here'

# Load trained model
df, cosine_sim = joblib.load("models/job_recommender.pkl")

# Function to find job roles based on skills
def find_job_roles_by_skills(skills, top_n=5):
    skills = skills.lower().split(",")  # Split skills by comma
    skills = [skill.strip() for skill in skills]  # Remove extra spaces
    
    # Filter jobs that match the skills
    matched_jobs = df[df["skills"].apply(lambda x: any(skill in x.lower() for skill in skills))]
    
    # Sort by the number of matched skills
    matched_jobs["matched_skills"] = matched_jobs["skills"].apply(lambda x: sum(skill in x.lower() for skill in skills))
    matched_jobs = matched_jobs.sort_values(by="matched_skills", ascending=False).head(top_n)
    
    # Return only the columns we need
    return matched_jobs[["job_role", "company_name", "company_type", "skills"]].to_dict(orient="records")

# Function to find job roles based on job role name
def find_job_roles_by_job_role(job_role, top_n=5):
    # Filter jobs that match the job role name
    matched_jobs = df[df["job_role"].str.contains(job_role, case=False, na=False)]
    
    return matched_jobs[["job_role", "company_name", "company_type", "skills"]].to_dict(orient="records")

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
