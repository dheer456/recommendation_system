# SHL Assessment Recommendation System

The **SHL Assessment Recommendation System** is a smart and lightweight web application that helps recruiters and HR professionals find the most suitable SHL assessments based on job roles, skills, and other hiring needs. Built using Python, Gradio, and TF-IDF, this tool uses semantic search to recommend assessments from a structured dataset.

---

## 🚀 Features

- 🔍 **Natural Language Input:** Just type what you need — the system understands queries like “Need a test for data analysis role” or “Looking for a 30-minute communication skill test.”
- 🤖 **TF-IDF Matching:** Uses TF-IDF vectorization and cosine similarity to match your input with the best-fit assessments.
- 💻 **Gradio Interface:** Simple, user-friendly web interface powered by Gradio.
- ⚡ **Fast & Lightweight:** No heavy models; runs efficiently on local systems or Hugging Face Spaces.
- 📊 **Real-World Dataset:** Works with a CSV of SHL assessment descriptions.

---

## 🧠 How It Works

1. Loads a dataset of SHL assessments from a CSV file.
2. Vectorizes each assessment description using **TF-IDF Vectorizer**.
3. Takes user query and vectorizes it the same way.
4. Computes **cosine similarity** between query vector and each assessment.
5. Ranks and returns top 5 matches with details like name, description, duration, and number of questions.

---


### 🔸 Parameters

| Name         | Type   | Description                                      | Required |
|--------------|--------|--------------------------------------------------|----------|
| `query`      | string | Job description, skills, or keywords             | ✅ Yes   |
| `time_limit` | int    | Maximum allowed assessment duration in minutes   | ❌ No    |

### ✅ Example


### 📥 Sample Response

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


