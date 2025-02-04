from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
df, cosine_sim = joblib.load("models/job_recommender.pkl")

# Recommendation function
def get_recommendations(job_role, top_n=5):
    if job_role not in df["job_role"].values:
        return []

    idx = df[df["job_role"] == job_role].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    job_indices = [i[0] for i in sim_scores]

    return df.iloc[job_indices][["job_role", "company_name", "company_type","skills"]].to_dict(orient="records")

# Home Route
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        job_role = request.form["job_role"]
        recommendations = get_recommendations(job_role)
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
