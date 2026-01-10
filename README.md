# Resume Screening Tool

## Overview
This project is a simple machine learning resume screening system that evaluates resume text to estimates how well a candidate matches a technical role.

## Goal
To analyze resume text and identify candidates with stronger technical backgrounds by learning patterns in technical skill related language.

## Frameworks & Libraries
- Python
- scikit-learn
- pandas

## Dataset
The model is trained on a small, labeled dataset of resume-style text:
- `1` → Good technical fit  
- `0` → Not a technical fit  

The dataset is intentionally lightweight and designed to demonstrate the screening pipeline rather than achieve production-level performance.


## How It Works

### 1. Training the Model
The model is trained using `train.py` on resume text from `data.csv`.

### 2. Converting Text to Numbers
Since we need numerical input, resume text is converted into numbers using CountVectorizer. This counts how often each word appears in a resume.

### 3. Learning Patterns
A **Logistic Regression** model learns which words are more associated with technical resumes.
- Technical words receive positive weights  
- Non-technical words receive negative weights  

Examples:
- `python` → positive weight  
- `machine learning` → positive weight  
- `retail` → negative weight  


## How to Run

1. Install dependencies:
   ```bash
   pip install pandas scikit-learn
2. Train the model:   
    ```bash
    python train.py
3. Predict resume fit:
    ```bash
    python predict.py

