from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
df, cosine_sim = joblib.load("models/job_recommender.pkl")

# Function to find job roles by skills or job role
def find_job_roles(query, search_type='skills', top_n=5):
    query = query.lower()
    
    if search_type == 'skills':
        skills = query.split(",")  # Split skills by comma
        skills = [skill.strip() for skill in skills]  # Remove extra spaces
        
        # Filter jobs that match the skills
        matched_jobs = df[df["skills"].apply(lambda x: any(skill in x.lower() for skill in skills))]
        
        # Sort by the number of matched skills
        matched_jobs["matched_skills"] = matched_jobs["skills"].apply(lambda x: sum(skill in x.lower() for skill in skills))
        matched_jobs = matched_jobs.sort_values(by="matched_skills", ascending=False).head(top_n)
        
        return matched_jobs[["job_role", "company_name", "company_type", "skills"]].to_dict(orient="records")

    elif search_type == 'job_role':
        # Find jobs matching the job role
        matched_jobs = df[df["job_role"].str.contains(query, case=False, na=False)]
        return matched_jobs[["job_role", "company_name", "company_type", "skills"]].to_dict(orient="records")

# Home Route
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    search_type = 'skills'
    if request.method == "POST":
        skills = request.form["query"]  # Get query (skills or job role) from the form
        search_type = request.form["search_type"]  # Determine if searching by 'skills' or 'job_role'
        recommendations = find_job_roles(skills, search_type)  # Get recommendations based on input
        
    return render_template("index.html", recommendations=recommendations, search_type=search_type)

if __name__ == "__main__":
    app.run(debug=True)
