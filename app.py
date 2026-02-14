from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load dataset
df = pd.read_csv("news1.csv")

# Train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = MultinomialNB()
model.fit(X, y)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    news = data["news"]

    news_vec = vectorizer.transform([news])
    prediction = model.predict(news_vec)[0]

    if prediction == "real":
        result = "This news appears to be REAL."
    else:
        result = "This news appears to be FAKE."

    # Save history safely
    with open("history.txt", "a") as file:
        file.write(news + " -> " + prediction + "\n")

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run()