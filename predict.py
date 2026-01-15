import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

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

# Convert resume
new_resume_vectorized = vectorizer.transform(new_resume)


probability = model.predict_proba(new_resume_vectorized)[0][1]

print(f"Probability of being a good fit: {probability:.2%}")


#confidence label

def confidence_label(prob):
    if prob >= 0.85:
        return "Strong Match"
    elif prob >= 0.65:
        return "Moderate Match"
    else:
        return "Weak Match"

confidence = confidence_label(probability)

print(f"Confidence Level: {confidence}")

