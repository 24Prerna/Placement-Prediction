# custom_transformers.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

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
        # Handle any data cleaning that was in the original transformer
        return X_copy
