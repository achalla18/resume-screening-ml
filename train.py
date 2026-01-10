import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# training data
data = pd.read_csv("data.csv")

X = data["resume_text"]
y = data["label"]

vectorizer = CountVectorizer(stop_words="english")
X_vectorized = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vectorized, y)

# Ex resume 
new_resume = [
    "SQL data analysis communication teamwork python matplotlib pipelines data engineering basics"
]

new_resume_vectorized = vectorizer.transform(new_resume)
prediction = model.predict(new_resume_vectorized)

if prediction[0] == 1:
    print("✅ Good Fit")
else:
    print("❌ Not a Good Fit")
