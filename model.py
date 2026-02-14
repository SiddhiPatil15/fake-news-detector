import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("news1.csv")

# Convert text to numeric form
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])

y = df["label"]

# Train model
model = MultinomialNB()
model.fit(X, y)

print("Model trained successfully!")

# Test prediction
test_news = ["Aliens spotted in Mumbai"]
test_vec = vectorizer.transform(test_news)
print("Prediction:", model.predict(test_vec)[0])