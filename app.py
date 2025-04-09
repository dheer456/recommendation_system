from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import gradio as gr
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

class SHLAssessmentRAG:
    def __init__(self):
        # Use TF-IDF vectorizer instead of Sentence Transformers
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.assessments_df = None
        self.assessment_vectors = None
        self.initialize_data()
    
    def initialize_data(self):
        """Load and process SHL assessment data"""
        print("Loading SHL assessment data...")
        self.assessments_df = self.load_assessment_data()
        
        # Create text representations for each assessment
        assessment_texts = [
            f"{row['Assessment Name']} {row['Description']} {row['Test Type']} {row['Competencies']} {row['Job Levels']}"
            for _, row in self.assessments_df.iterrows()
        ]
        
        # Create TF-IDF vectors
        self.assessment_vectors = self.vectorizer.fit_transform(assessment_texts)
        print(f"Loaded {len(self.assessments_df)} assessments")
    
    def load_assessment_data(self):
        """
        Load assessment data from file
        In a production environment, you might fetch this from a database
        """
        # Sample data for demonstration
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
        
        # Create query vector using the same TF-IDF vectorizer
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(query_vector, self.assessment_vectors)[0]
        
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

# For Gradio UI
def create_gradio_interface():
    # Initialize the RAG model
    rag_model = SHLAssessmentRAG()
    
    def recommend(query, time_limit_str=None):
        # Process time limit if provided
        time_limit = None
        if time_limit_str and time_limit_str.strip():
            try:
                time_limit = int(time_limit_str)
            except ValueError:
                return "Please enter a valid number for time limit"
        
        # Get recommendations
        recommendations = rag_model.get_recommendations(query, max_results=10, time_limit=time_limit)
        
        # Format as a nicely formatted table for display
        if not recommendations:
            return "No matching assessments found."
        
        # Convert to Gradio-friendly format (DataFrame)
        df = pd.DataFrame(recommendations)
        return df
    
    # Create Gradio interface
    interface = gr.Interface(
        fn=recommend,
        inputs=[
            gr.Textbox(
                label="Job Description or Query",
                placeholder="Enter job description or requirements...",
                lines=5
            ),
            gr.Textbox(
                label="Time Limit (minutes, optional)",
                placeholder="e.g., 45"
            )
        ],
        outputs=gr.Dataframe(),
        title="SHL Assessment Recommendation System",
        description="Enter a job description or query to get relevant SHL assessment recommendations.",
        examples=[
            ["I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment that can be completed in 40 minutes."],
            ["Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript. Need an assessment package that can test all skills with max duration of 60 minutes."],
            ["Need to find people with strong leadership skills and decision-making abilities. The assessment should take less than 30 minutes."],
            ["Looking for a data analyst assessment that tests SQL knowledge and analytical thinking, under 45 minutes"]
        ]
    )
    return interface

# For Flask API
rag_model = SHLAssessmentRAG()

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "SHL Assessment Recommendation API",
        "usage": "Send GET requests to /api/recommend?query=your job description"
    })

@app.route('/api/recommend', methods=['GET'])
def api_recommend():
    """
    Endpoint to get assessment recommendations based on a query
    Query parameters:
    - query: Job description or requirements text
    - time_limit: Optional time limit in minutes
    - max_results: Optional maximum number of results (default 10)
    """
    query = request.args.get('query', '')
    
    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400
    
    # Parse optional parameters
    try:
        time_limit = request.args.get('time_limit')
        if time_limit:
            time_limit = int(time_limit)
            
        max_results = request.args.get('max_results', '10')
        max_results = int(max_results)
    except ValueError:
        return jsonify({"error": "Invalid parameter value"}), 400
    
    # Get recommendations
    recommendations = rag_model.get_recommendations(query, max_results=max_results, time_limit=time_limit)
    
    # Return JSON response
    return jsonify({
        "query": query,
        "time_limit": time_limit,
        "recommendations": recommendations
    })

# Main function to decide which interface to use
def main():
    # For Hugging Face Spaces, use Gradio
    interface = create_gradio_interface()
    interface.launch()

# Entry point
if __name__ == "__main__":
    # Choose which interface to use based on environment
    # Check if running on Hugging Face Spaces
    if os.environ.get('SPACE_ID'):
        main()  # Run Gradio interface
    else:
        # Get port from environment variable or use default
        port = int(os.environ.get("PORT", 5000))
        # Run the Flask application
        app.run(host="0.0.0.0", port=port)
