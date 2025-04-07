# SHL Assessment Recommender

This application uses semantic search to recommend appropriate SHL assessments based on job descriptions or specific requirements.

## Features

- **Semantic Search**: Uses sentence embeddings to find the most relevant assessments
- **Time-Based Filtering**: Option to filter assessments by duration
- **Web UI**: Interactive Gradio interface for easy use
- **API Access**: REST API for programmatic access

## API Usage

The application provides a REST API endpoint for recommendations:

```
GET /api/recommend?query=YOUR_QUERY&time_limit=OPTIONAL_TIME_LIMIT
```

### Parameters:
- `query`: Job description or requirements (required)
- `time_limit`: Maximum assessment duration in minutes (optional)

### Example:
```
GET /api/recommend?query=Java developer with OOP skills&time_limit=30
```

### Response:
```json
[
  {
    "Assessment Name": "Verify Coding - Java",
    "URL": "https://www.shl.com/solutions/products/verify-coding-java/",
    "Remote Testing Support": "Yes",
    "Adaptive/IRT Support": "No",
    "Duration": "30 minutes",
    "Test Type": "Technical Skills",
    "Similarity Score": 0.8934
  },
  ...
]
```

## Web Interface

The application also provides a user-friendly web interface where you can:
1. Enter a job description
2. Specify time constraints (optional)
3. View a table of recommended assessments

## Deployment

This application is configured to be deployed on Hugging Face Spaces.
