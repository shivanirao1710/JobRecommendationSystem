import psycopg2
from psycopg2.extras import DictCursor
import json
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request, redirect, url_for, session
import faiss
import numpy as np
from transformers import pipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# PostgreSQL database connection details
db_config = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'shivanirao1710',
    'database': 'jobtaxonomy'
}

# Initialize SentenceTransformer model for semantic understanding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize GPT-2 for text generation (for rephrasing or fixing grammar)
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Initialize Hugging Face LLM pipeline for simple conversation
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'


def get_job_data_from_postgresql():
    """Fetch job data from PostgreSQL database"""
    try:
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor(cursor_factory=DictCursor)
        cursor.execute("SELECT job_role, company_name, company_type, knowledge_cleaned, skills_cleaned, combined_features, embedding FROM job_data_cleaned")
        job_data = cursor.fetchall()
        cursor.close()
        connection.close()
        return job_data
    except psycopg2.Error as err:
        print(f"Error: {err}")
        return []


def prepare_faiss_index(job_data):
    """Prepare FAISS index for semantic search"""
    job_embeddings = []
    job_titles = []
    for job in job_data:
        embedding = job.get('embedding', '')
        if embedding:
            try:
                embedding = json.loads(embedding)
                job_embeddings.append(embedding)
                job_titles.append(job['job_role'])
            except json.JSONDecodeError:
                continue
    job_embeddings = np.array(job_embeddings).astype('float32')
    faiss_index = faiss.IndexFlatL2(job_embeddings.shape[1])  # L2 distance for similarity search
    faiss_index.add(job_embeddings)
    return faiss_index, job_titles


def find_job_roles_by_skills(skills, top_n=5):
    """Find job roles based on skills"""
    skills_query = skills.lower().split(",")  # Split skills by comma and strip
    skills_query = [skill.strip() for skill in skills_query]
    query = " ".join(skills_query)
    
    query_embedding = model.encode([query])[0]
    job_data = get_job_data_from_postgresql()

    faiss_index, job_titles = prepare_faiss_index(job_data)
    query_embedding = np.array(query_embedding).reshape(1, -1).astype('float32')
    distances, indices = faiss_index.search(query_embedding, top_n)

    recommended_jobs = []
    for index in indices[0]:
        job = next((job for i, job in enumerate(job_data) if job_titles[i] == job_titles[index]), None)
        if job:
            recommended_jobs.append(job)
    return recommended_jobs


def find_job_roles_by_job_role(job_role, top_n=5):
    """Find job roles by job role name"""
    job_role = job_role.lower().strip()
    query_embedding = model.encode([job_role])[0]
    job_data = get_job_data_from_postgresql()

    faiss_index, job_titles = prepare_faiss_index(job_data)
    query_embedding = np.array(query_embedding).reshape(1, -1).astype('float32')
    distances, indices = faiss_index.search(query_embedding, top_n)

    recommended_jobs = []
    for index in indices[0]:
        job = next((job for i, job in enumerate(job_data) if job_titles[i] == job_titles[index]), None)
        if job:
            recommended_jobs.append(job)
    return recommended_jobs


def find_job_roles_by_company(company_name, top_n=5):
    """Find job roles based on company name"""
    company_name = company_name.lower().strip()
    job_data = get_job_data_from_postgresql()
    
    # Filter job data based on the company name
    filtered_jobs = [job for job in job_data if job['company_name'].lower() == company_name]
    
    return filtered_jobs


def correct_grammar_and_generate_response(text):
    """Generate a well-formed response using GPT-2 for grammar correction"""
    inputs = gpt_tokenizer.encode(text, return_tensors='pt')
    
    # Use max_new_tokens to handle long input
    outputs = gpt_model.generate(
        inputs,
        max_new_tokens=150,  # Limit the generation to 150 new tokens
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.92,
        temperature=0.7
    )
    
    response = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def extract_job_role_from_input(user_input):
    """Extract job role from the user's input."""
    job_role_keywords = ["tell me about", "what is", "role of", "job of", "describe", "give me information about"]
    for keyword in job_role_keywords:
        if user_input.lower().startswith(keyword):
            return user_input[len(keyword):].strip()
    return user_input.strip()


@app.route("/chatbot", methods=["GET", "POST"])
def chatbot_route():
    """Handle chatbot interaction"""
    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        
        if not user_input:
            return render_template("chatbot.html", user_input=user_input, response="Please ask me something!")

        # Check if the user is asking about job roles
        job_role_keywords = ['job', 'role', 'position', 'career']
        skills_keywords = ['skills', 'technology', 'tools']

        response = ""

        # If the user input contains keywords related to job roles
        if any(keyword in user_input.lower() for keyword in job_role_keywords):
            job_role = extract_job_role_from_input(user_input)
            job_info = get_job_data_from_postgresql()
            job_details = next((job for job in job_info if job['job_role'].lower() == job_role), None)
            
            if job_details:
                response = f"Job Role: {job_details['job_role']} at {job_details['company_name']}\n"
                response += f"Skills Needed: {job_details['skills_cleaned']}\n"
                response += f"Knowledge Required: {job_details['knowledge_cleaned']}\n"
                response += f"Company Type: {job_details['company_type']}\n"
                response += f"Description: {job_details.get('combined_features', 'No description available')}"
            else:
                response = f"Sorry, I couldn't find details for the job role '{job_role}'. Can you try another one?"
        
        # If the user input contains keywords related to skills
        elif any(keyword in user_input.lower() for keyword in skills_keywords):
            skills_query = user_input.lower().strip()
            recommended_jobs = find_job_roles_by_skills(skills_query)
            
            if recommended_jobs:
                response = "I found some jobs related to the skills you're looking for:\n"
                for job in recommended_jobs:
                    response += f"- {job['job_role']} at {job['company_name']} (Skills Needed: {job['skills_cleaned']})\n"
            else:
                response = f"I couldn't find any jobs with the skills '{skills_query}'. Can you try different skills?"

        # General conversation: Use the chatbot for free-form conversations
        else:
            response = chatbot(user_input, max_length=50, num_return_sequences=1)[0]['generated_text']
        
        # Use GPT-2 to correct grammar and rephrase the response
        response = correct_grammar_and_generate_response(response)
        
        return render_template("chatbot.html", user_input=user_input, response=response)
    
    return render_template("chatbot.html", user_input="", response="Ask me about job roles, skills, or any other questions!")


# Home Route for search functionality
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    search_type = None
    if request.method == "POST":
        search_type = request.form.get("search_type", "skills")
        query = request.form.get("skills", "").strip()
        
        if not query:
            return render_template("index.html", recommendations=[], error="Please enter skills, job role, or company name.")
        
        if search_type == "skills":
            recommendations = find_job_roles_by_skills(query)
        elif search_type == "job_role":
            recommendations = find_job_roles_by_job_role(query)
        elif search_type == "company_name":
            recommendations = find_job_roles_by_company(query)
        
        session['search_type'] = search_type
        session['recommendations'] = recommendations
        session['query'] = query
    
    if 'recommendations' in session:
        recommendations = session['recommendations']
        search_type = session['search_type']
    
    return render_template("index.html", recommendations=recommendations, search_type=search_type)


# Job Details Route (for viewing specific job details)
@app.route("/job/<job_role>/<company_name>")
def job_details(job_role, company_name):
    job_data = get_job_data_from_postgresql()
    
    job_details = next((job for job in job_data if job['job_role'] == job_role and job['company_name'] == company_name), None)
    
    if job_details:
        return render_template("job_details.html", job=job_details)
    else:
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
