import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import gradio as gr
import torch
import os

# Configuration
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Good balance of speed and performance

class SHLAssessmentRAG:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.assessments_df = None
        self.assessment_embeddings = None
        self.initialize_data()
    
    def initialize_data(self):
        """Load and process SHL assessment data"""
        print("Loading SHL assessment data...")
        self.assessments_df = self.load_assessment_data()
        
        # Create embeddings for each assessment
        print("Creating embeddings for assessments...")
        assessment_texts = [
            f"{row['Assessment Name']} {row['Description']} {row['Test Type']} {row['Competencies']} {row['Job Levels']}"
            for _, row in self.assessments_df.iterrows()
        ]
        self.assessment_embeddings = self.model.encode(assessment_texts)
        print(f"Loaded {len(self.assessments_df)} assessments")
    
    def load_assessment_data(self):
        """
        Load the assessment data from the provided sample
        """
        # Sample data to demonstrate the approach
        sample_data = [
            {
                "Assessment Name": "Verify Interactive - Java",
                "URL": "https://www.shl.com/solutions/products/verify-interactive-java/",
                "Remote Testing Support": "Yes",
                "Adaptive/IRT Support": "No",
                "Duration": "40 minutes",
                "Test Type": "Technical Skills",
                "Description": "Assesses proficiency in Java programming through interactive coding challenges",
                "Competencies": "Java, Object-Oriented Programming, Problem Solving, Coding",
                "Job Levels": "Junior, Mid-level, Senior Developer",
            },
            {
                "Assessment Name": "Verify Coding - JavaScript",
                "URL": "https://www.shl.com/solutions/products/verify-coding-javascript/",
                "Remote Testing Support": "Yes",
                "Adaptive/IRT Support": "No",
                "Duration": "30 minutes",
                "Test Type": "Technical Skills",
                "Description": "Evaluates JavaScript knowledge and practical coding abilities",
                "Competencies": "JavaScript, Front-end Development, Web Programming",
                "Job Levels": "Junior, Mid-level Developer",
            },
            {
                "Assessment Name": "Verify Coding - Python",
                "URL": "https://www.shl.com/solutions/products/verify-coding-python/",
                "Remote Testing Support": "Yes",
                "Adaptive/IRT Support": "No",
                "Duration": "30 minutes",
                "Test Type": "Technical Skills",
                "Description": "Tests Python programming skills through practical coding problems",
                "Competencies": "Python, Data Structures, Algorithms, Problem Solving",
                "Job Levels": "Junior, Mid-level, Senior Developer",
            },
            {
                "Assessment Name": "Verify Coding - SQL",
                "URL": "https://www.shl.com/solutions/products/verify-coding-sql/",
                "Remote Testing Support": "Yes",
                "Adaptive/IRT Support": "No",
                "Duration": "25 minutes",
                "Test Type": "Technical Skills",
                "Description": "Assesses SQL query writing skills and database knowledge",
                "Competencies": "SQL, Database Design, Query Optimization",
                "Job Levels": "Data Analyst, Database Developer",
            },
            {
                "Assessment Name": "Verify - General Aptitude",
                "URL": "https://www.shl.com/solutions/products/verify-general-aptitude/",
                "Remote Testing Support": "Yes",
                "Adaptive/IRT Support": "Yes",
                "Duration": "25 minutes",
                "Test Type": "Cognitive",
                "Description": "Measures critical thinking and problem-solving abilities",
                "Competencies": "Logical Reasoning, Problem Solving, Critical Thinking",
                "Job Levels": "All Levels",
            },
            {
                "Assessment Name": "Verify - Numerical Reasoning",
                "URL": "https://www.shl.com/solutions/products/verify-numerical-reasoning/",
                "Remote Testing Support": "Yes",
                "Adaptive/IRT Support": "Yes",
                "Duration": "20 minutes",
                "Test Type": "Cognitive",
                "Description": "Evaluates numerical analysis and problem-solving skills",
                "Competencies": "Numerical Analysis, Data Interpretation, Mathematical Reasoning",
                "Job Levels": "Analysts, Managers, Executives",
            },
            {
                "Assessment Name": "OPQ - Occupational Personality Questionnaire",
                "URL": "https://www.shl.com/solutions/products/opq/",
                "Remote Testing Support": "Yes",
                "Adaptive/IRT Support": "No",
                "Duration": "25 minutes",
                "Test Type": "Personality",
                "Description": "Comprehensive personality assessment for workplace behaviors",
                "Competencies": "Teamwork, Leadership, Communication, Work Style",
                "Job Levels": "All Levels",
            },
            {
                "Assessment Name": "Motivational Questionnaire",
                "URL": "https://www.shl.com/solutions/products/motivational-questionnaire/",
                "Remote Testing Support": "Yes",
                "Adaptive/IRT Support": "No",
                "Duration": "20 minutes",
                "Test Type": "Personality",
                "Description": "Measures motivational drivers and preferences in work settings",
                "Competencies": "Motivation, Work Preferences, Career Alignment",
                "Job Levels": "All Levels",
            },
            {
                "Assessment Name": "Full Stack Developer Assessment",
                "URL": "https://www.shl.com/solutions/products/full-stack-assessment/",
                "Remote Testing Support": "Yes",
                "Adaptive/IRT Support": "No",
                "Duration": "60 minutes",
                "Test Type": "Technical Skills",
                "Description": "Comprehensive assessment combining JavaScript, SQL, Python and collaborative skills",
                "Competencies": "JavaScript, SQL, Python, Front-end, Back-end, Problem Solving, Collaboration",
                "Job Levels": "Mid-level, Senior Developer",
            },
            {
                "Assessment Name": "Situational Judgment Test",
                "URL": "https://www.shl.com/solutions/products/situational-judgment/",
                "Remote Testing Support": "Yes",
                "Adaptive/IRT Support": "No",
                "Duration": "30 minutes",
                "Test Type": "Behavioral",
                "Description": "Evaluates decision-making in realistic workplace scenarios",
                "Competencies": "Decision Making, Judgment, Leadership, Teamwork",
                "Job Levels": "All Levels",
            },
            {
                "Assessment Name": "Cognitive and Personality Package",
                "URL": "https://www.shl.com/solutions/products/cognitive-personality-package/",
                "Remote Testing Support": "Yes",
                "Adaptive/IRT Support": "Yes",
                "Duration": "45 minutes",
                "Test Type": "Cognitive, Personality",
                "Description": "Combined assessment measuring both cognitive abilities and personality traits",
                "Competencies": "Problem Solving, Critical Thinking, Teamwork, Communication",
                "Job Levels": "All Levels, Analysts",
            },
            {
                "Assessment Name": "Remote Collaboration Assessment",
                "URL": "https://www.shl.com/solutions/products/remote-collaboration-assessment/",
                "Remote Testing Support": "Yes",
                "Adaptive/IRT Support": "No",
                "Duration": "30 minutes",
                "Test Type": "Behavioral",
                "Description": "Assesses ability to collaborate effectively in remote work environments",
                "Competencies": "Communication, Teamwork, Self-management, Virtual Collaboration",
                "Job Levels": "All Levels",
            },
            {
                "Assessment Name": "Technical Aptitude Battery",
                "URL": "https://www.shl.com/solutions/products/technical-aptitude/",
                "Remote Testing Support": "Yes",
                "Adaptive/IRT Support": "Yes",
                "Duration": "35 minutes",
                "Test Type": "Cognitive",
                "Description": "Evaluates aptitude for technical roles including spatial reasoning and technical understanding",
                "Competencies": "Technical Reasoning, Spatial Ability, Mechanical Comprehension",
                "Job Levels": "Technical Roles, Engineers",
            },
            {
                "Assessment Name": "Leadership Assessment",
                "URL": "https://www.shl.com/solutions/products/leadership-assessment/",
                "Remote Testing Support": "Yes",
                "Adaptive/IRT Support": "No",
                "Duration": "40 minutes",
                "Test Type": "Personality, Behavioral",
                "Description": "Comprehensive assessment of leadership potential and capabilities",
                "Competencies": "Strategic Thinking, Team Management, Decision Making, Vision",
                "Job Levels": "Managers, Directors, Executives",
            },
            {
                "Assessment Name": "Data Analyst Assessment",
                "URL": "https://www.shl.com/solutions/products/data-analyst-assessment/",
                "Remote Testing Support": "Yes",
                "Adaptive/IRT Support": "No",
                "Duration": "45 minutes",
                "Test Type": "Technical Skills, Cognitive",
                "Description": "Evaluates analytical thinking, SQL skills, and data interpretation abilities",
                "Competencies": "Data Analysis, SQL, Statistical Reasoning, Problem Solving",
                "Job Levels": "Junior, Mid-level Analysts",
            }
        ]
        
        return pd.DataFrame(sample_data)
    
    def get_recommendations(self, query, max_results=10, time_limit=None):
        """
        Generate assessment recommendations based on a query
        Args:
            query: Natural language query or job description
            max_results: Maximum number of recommendations to return
            time_limit: Optional time limit for assessments in minutes
        Returns:
            List of recommended assessments
        """
        # Process the query to extract time constraints if not explicitly provided
        if time_limit is None:
            time_match = re.search(r'(\d+)\s*minutes', query.lower())
            if time_match:
                time_limit = int(time_match.group(1))
        
        # Create query embedding
        query_embedding = self.model.encode([query])
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(query_embedding, self.assessment_embeddings)[0]
        
        # Create a copy of the DataFrame with similarity scores
        results_df = self.assessments_df.copy()
        results_df['similarity_score'] = similarity_scores
        
        # Filter by time limit if provided
        if time_limit:
            # Extract numeric duration values
            results_df['duration_min'] = results_df['Duration'].str.extract(r'(\d+)').astype(float)
            results_df = results_df[results_df['duration_min'] <= time_limit]
        
        # Sort by similarity score
        results_df = results_df.sort_values('similarity_score', ascending=False)
        
        # Select top recommendations
        top_recommendations = results_df.head(max_results)
        
        # Format results
        recommendations = []
        for _, row in top_recommendations.iterrows():
            recommendations.append({
                "Assessment Name": row["Assessment Name"],
                "URL": row["URL"],
                "Remote Testing Support": row["Remote Testing Support"],
                "Adaptive/IRT Support": row["Adaptive/IRT Support"],
                "Duration": row["Duration"],
                "Test Type": row["Test Type"],
                "Similarity Score": float(row["similarity_score"]),
            })
        
        return recommendations

# Initialize the model only once
rag_model = SHLAssessmentRAG()

def api_recommend(query, time_limit=None):
    """API function to get recommendations"""
    if time_limit and str(time_limit).strip():
        try:
            time_limit = int(time_limit)
        except ValueError:
            return {"error": "Please enter a valid number for time limit"}
    else:
        time_limit = None
        
    recommendations = rag_model.get_recommendations(query, time_limit=time_limit)
    return recommendations

# Gradio UI Implementation
def create_gradio_interface():
    def gradio_recommend(query, time_limit=None):
        """Gradio wrapper for recommend function"""
        result = api_recommend(query, time_limit)
        if "error" in result:
            return result["error"]
        return pd.DataFrame(result)
    
    with gr.Blocks(title="SHL Assessment Recommender") as interface:
        gr.Markdown("# SHL Assessment Recommendation System")
        gr.Markdown("Enter a job description or query to get assessment recommendations.")
        
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(label="Job Description or Query", lines=5, 
                                        placeholder="Enter job description or specific requirements...")
                time_limit = gr.Textbox(label="Time Limit (minutes, optional)", placeholder="e.g., 45")
                submit_btn = gr.Button("Get Recommendations")
        
        output = gr.Dataframe(label="Recommended Assessments")
        
        gr.Markdown("### API Usage")
        gr.Markdown("""
        You can also use this as an API:
        
        **GET** `/api/recommend?query=YOUR_QUERY&time_limit=OPTIONAL_TIME_LIMIT`
        
        Example: `/api/recommend?query=Java developer with OOP skills&time_limit=30`
        """)
        
        submit_btn.click(fn=gradio_recommend, inputs=[query_input, time_limit], outputs=output)
    
    return interface

# API endpoint for recommend
def predict(query, time_limit=None):
    return api_recommend(query, time_limit)

# Create the interface and expose the API
interface = create_gradio_interface()
app = gr.mount_gradio_app(app=None, blocks=interface, path="/")

# Add the API endpoint
@app.get("/api/recommend")
def recommend_api(query: str, time_limit: str = None):
    """API endpoint for recommendations"""
    return api_recommend(query, time_limit)

if __name__ == "__main__":
    # For local testing only
    interface.launch()
