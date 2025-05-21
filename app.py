from flask import Flask, request, render_template, redirect, url_for, session
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os
import joblib
from dotenv import load_dotenv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
import requests
from datetime import datetime

# Load environment variables
load_dotenv()

# Define custom transformers in the main module to match the pickled model
skill_keywords = [
    "python", "java", "c", "c++", "c#", "r", "javascript", "typescript",
    "html", "css", "react", "angular", "vue", "node.js", "flask", "django", "php", "bootstrap",
    "machine learning", "deep learning", "data analysis", "data visualization", "nlp", "computer vision",
    "tensorflow", "keras", "pytorch", "opencv", 
    "sql", "mysql", "postgresql", "mongodb", "oracle", "excel", "power bi", "tableau", "hadoop", "spark",
    "big data", "etl", "data warehouse", "data lake", "cloud", "aws", "azure", "google cloud", "docker",
    "kubernetes", "ci/cd", "jenkins", "linux", "windows", "unix", "networking", "tcp/ip", "cybersecurity",
    "firewall", "vpn", "communication", "problem solving", "critical thinking", "teamwork", "leadership",
    "adaptability", "git", "github", "jupyter", "visual studio", "android studio", "vscode", "unity",
    "matlab", "jira", "agile"
]

class SkillExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.skill_keywords = skill_keywords
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None):
        skills = X['Skills'].apply(self.extract_skills)
        self.mlb.fit(skills)
        return self

    def transform(self, X):
        skills = X['Skills'].apply(self.extract_skills)
        return self.mlb.transform(skills)

    def extract_skills(self, text):
        text = str(text).lower()
        return list(set(skill for skill in self.skill_keywords if skill in text))

    def get_feature_names_out(self, input_features=None):
        return self.mlb.classes_

class ColumnCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        return X_copy

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session

# Load company data
company_data = pd.read_csv("company.csv")

# Initialize the sentence transformer model for skill matching
model = SentenceTransformer('all-MiniLM-L6-v2')

# Pre-encode company job descriptions
job_descriptions = company_data['Job Description'].astype(str).tolist()
job_embeddings = model.encode(job_descriptions, convert_to_tensor=True)

# Load the trained pipeline
pipeline = joblib.load("placement_prediction_pipeline.pkl")

# Add Adzuna API credentials
ADZUNA_APP_ID = os.getenv('ADZUNA_APP_ID')
ADZUNA_APP_KEY = os.getenv('ADZUNA_APP_KEY')

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Step 1: Collect form inputs for placement prediction
            tenth = float(request.form["tenth"])
            twelfth = float(request.form["twelfth"])
            ug_cgpa = float(request.form["ug-cgpa"])
            pg_cgpa = float(request.form["pg-cgpa"])
            skills = ", ".join(request.form.getlist("skills"))

            # Step 2: Create DataFrame for ML pipeline
            data = {
                "10th percentage": tenth,
                "12th percentage": twelfth,
                "Graduation CGPA": ug_cgpa,
                "Post Graduation 1st year CGPA": pg_cgpa,
                "Skills": skills
            }
            df = pd.DataFrame([data])
            
            # Step 3: Predict using ML pipeline
            prediction = pipeline.predict(df)[0]
            is_placed = bool(prediction == 1)  # Convert to Python native boolean
            result = "Placed ✅" if is_placed else "Not Placed ❌"

            # Step 4: If placed, get company recommendations
            recommendations = []
            
            if is_placed:
                # Get recommendations from CSV data
                student_embedding = model.encode(skills, convert_to_tensor=True)
                cosine_scores = util.cos_sim(student_embedding, job_embeddings)[0]
                top_indices = cosine_scores.argsort(descending=True)[:5]  # Get top 5 matches

                seen_companies = set()
                for i in top_indices:
                    company = company_data.iloc[i.item()]
                    if company['Company'] not in seen_companies:
                        seen_companies.add(company['Company'])
                        try:
                            recommendations.append({
                                "Company": str(company['Company']),
                                "Job Title": str(company['Job Title']),
                                "City": str(company['Work Location']),
                                "Expected_Salary": float(company['Avg_Salary']) if pd.notna(company['Avg_Salary']) else 0.0,
                                "Match_Score": float(cosine_scores[i]) * 100
                            })
                        except (ValueError, KeyError):
                            continue
                
                # Fallback if no recommendations found
                if not recommendations:
                    # Find jobs with any of the skills mentioned
                    skills_list = [skill.strip().lower() for skill in skills.split(',')]
                    
                    for _, company in company_data.sample(min(10, len(company_data))).iterrows():
                        job_desc = str(company['Job Description']).lower()
                        for skill in skills_list:
                            if skill and len(skill) > 2 and skill in job_desc:
                                try:
                                    recommendations.append({
                                        "Company": str(company['Company']),
                                        "Job Title": str(company['Job Title']),
                                        "City": str(company['Work Location']),
                                        "Expected_Salary": float(company['Avg_Salary']) if pd.notna(company['Avg_Salary']) else 0.0,
                                        "Match_Score": 70.0
                                    })
                                    break
                                except (ValueError, KeyError):
                                    continue

                # If placed and we have recommendations, redirect to recommendations page
                if recommendations:
                    session['recommendations'] = recommendations
                    session['prediction_text'] = f"🎓 {result}"
                    return redirect(url_for('recommendations'))
                else:
                    # Placed but no recommendations found
                    session['prediction_text'] = f"🎓 {result}"
                    session['is_placed'] = is_placed
                    session['message'] = "Congratulations on being placed! However, we couldn't find company matches for your profile at this time."
                    return redirect(url_for('result'))
            else:
                # Not placed - show appropriate message
                session['prediction_text'] = f"🎓 {result}"
                session['is_placed'] = is_placed
                session['message'] = "Unfortunately, you are not placed based on the current data. Consider improving your skills and academic performance."
                return redirect(url_for('result'))

        except Exception as e:
            session['prediction_text'] = f"⚠️ Error: {str(e)}"
            return redirect(url_for('result'))

    # Initial page load - GET request
    return render_template("index.html", form_submitted=False, prediction_text="", is_placed=None, message="")

@app.route("/result")
def result():
    # Get values from session
    prediction_text = session.pop('prediction_text', '')
    is_placed = session.pop('is_placed', None)
    message = session.pop('message', '')
    
    return render_template("index.html", 
                         form_submitted=True,
                         prediction_text=prediction_text,
                         is_placed=is_placed,
                         message=message)

@app.route("/recommendations")
def recommendations():
    # Get values from session
    recommendations = session.pop('recommendations', [])
    prediction_text = session.pop('prediction_text', '')
    
    return render_template("recommendations.html",
                         recommendations=recommendations,
                         prediction_text=prediction_text)

@app.route("/job-listings")
def job_listings():
    try:
        # Get user's skills from session or use common IT skills if none found
        skills = session.get('skills', '').lower()
        
        # Define IT job categories and keywords
        it_keywords = [
            'software', 'developer', 'engineer', 'programming', 'web', 
            'full stack', 'frontend', 'backend', 'devops', 'cloud',
            'data scientist', 'machine learning', 'AI', 'python', 'java',
            'javascript', 'react', 'node', 'database', 'IT', 'technology','data analyst','data maneger'
        ]
        
        # Create search query combining skills and IT keywords
        search_terms = []
        
        # Add user's skills to search
        if skills:
            skill_list = [s.strip() for s in skills.split(',')]
            search_terms.extend(skill_list)
        
        # Add relevant IT job titles
        search_terms.extend(['developer', 'engineer', 'programmer'])
        
        # Combine terms for search
        search_query = ' OR '.join(search_terms)
        
        # Prepare parameters for Adzuna API
        params = {
            'app_id': ADZUNA_APP_ID,
            'app_key': ADZUNA_APP_KEY,
            'results_per_page': 50,  # Get more results to filter
            'what': search_query,
            'category': 'it-jobs',  # Specifically target IT jobs category
            'content-type': 'application/json'
        }
        
        # Make API request to Adzuna
        response = requests.get('https://api.adzuna.com/v1/api/jobs/gb/search/1', params=params)
        response.raise_for_status()
        
        # Parse the response
        job_data = response.json()
        all_jobs = []
        
        for job in job_data.get('results', []):
            # Calculate relevance score based on skill matches
            description = job.get('description', '').lower()
            title = job.get('title', '').lower()
            
            # Count how many of the user's skills match
            skill_matches = sum(1 for skill in skill_list if skill in description or skill in title) if skills else 0
            
            # Check if it's an IT job by looking for IT keywords
            is_it_job = any(keyword in description.lower() or keyword in title.lower() for keyword in it_keywords)
            
            if is_it_job:
                all_jobs.append({
                    'title': job.get('title', ''),
                    'company': job.get('company', {}).get('display_name', ''),
                    'location': job.get('location', {}).get('display_name', ''),
                    'description': job.get('description', ''),
                    'salary_min': job.get('salary_min', ''),
                    'salary_max': job.get('salary_max', ''),
                    'created': datetime.strptime(job.get('created', ''), '%Y-%m-%dT%H:%M:%SZ').strftime('%B %d, %Y'),
                    'redirect_url': job.get('redirect_url', ''),
                    'skill_matches': skill_matches
                })
        
        # Sort jobs by number of skill matches, descending
        all_jobs.sort(key=lambda x: x['skill_matches'], reverse=True)
        
        # Take top 10 most relevant jobs
        jobs = all_jobs[:10]
        
        return render_template('job_listings.html', jobs=jobs, skills=skills)
        
    except Exception as e:
        return render_template('job_listings.html', error=str(e), jobs=[], skills=skills)

if __name__ == "__main__":
    app.run(debug=True)