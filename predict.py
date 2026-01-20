import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Train

data = pd.read_csv("data.csv")

X = data["resume_text"]
y = data["label"]

vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2))
X_vectorized = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vectorized, y)


def confidence_label(prob):
    if prob >= 0.85:
        return "Strong Match"
    elif prob >= 0.65:
        return "Moderate Match"
    else:
        return "Weak Match"

def explain_prediction(vectorizer, model, resume_vectorized, top_n=8):
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]

    present_indices = resume_vectorized.nonzero()[1]
    contributions = coefs[present_indices]

    top_indices = present_indices[np.argsort(contributions)[-top_n:]]
    return feature_names[top_indices]

# Input
print("\nPaste your resume below. Press ENTER twice when finished:\n")

lines = []
while True:
    line = input()
    if line.strip() == "":
        break
    lines.append(line)

resume_text = " ".join(lines)

# Predict
resume_vectorized = vectorizer.transform([resume_text])
probability = model.predict_proba(resume_vectorized)[0][1]
confidence = confidence_label(probability)

matched_terms = explain_prediction(
    vectorizer,
    model,
    resume_vectorized
)

# Output
print("\n===== Resume Fit Analysis =====")
print(f"Probability of Being a Good Fit: {probability:.2%}")
print(f"Confidence Level: {confidence}")

print("\nTop Matching Skills / Signals:")
for term in matched_terms:
    print(f"  • {term}")
