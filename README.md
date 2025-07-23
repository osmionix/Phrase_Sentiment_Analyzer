# Phrase Sentiment Analyzer 

A FastAPI-based NLP service that classifies Twitter posts as **Positive** or **Negative** using machine learning. Perfect for real-time sentiment analysis in social media applications.

---

## Features ✨

- **Text Classification**: Predicts sentiment (Positive/Negative) with ~85% accuracy.
- **NLP Pipeline**: 
  - Tokenization & stopword removal using NLTK
  - TF-IDF vectorization
  - LinearSVC classifier
- **REST API**: FastAPI endpoint for easy integration.
- **Production Ready**: Includes model training scripts and pre-trained models.

---

## Tech Stack 🛠️

| Component          | Technology |
|--------------------|------------|
| **Backend**        | FastAPI    |
| **ML Framework**   | Scikit-learn |
| **NLP**           | NLTK       |
| **Vectorization** | TF-IDF     |
| **Classifier**    | LinearSVC  |

---

## Installation ⚙️

### Steps
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows  
pip install -r requirements.txt  
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"  

Train the model - python scripts/train_model.py  
FastAPI Server - uvicorn app.main:app --reload  
