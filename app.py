from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

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
    
    return matched_jobs[["job_role", "company_name", "company_type", "skills"]].to_dict(orient="records")

# Home Route
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        skills = request.form["skills"]  # Get skills from the form
        recommendations = find_job_roles_by_skills(skills)  # Get recommendations based on skills
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)