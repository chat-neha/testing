from flask import Flask, request, render_template, jsonify
import spacy
import numpy as np
from sklearn import svm

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Define categories
class Category:
    BOOKS = "BOOKS"
    BANK = "BANK"

# Training data
train_x = [
    "good characters and plot progression",
    "check out the book",
    "good story. would recommend",
    "novel recommendation",
    "need to make a deposit to the bank",
    "balance inquiry savings",
    "save money"
]
train_y = [Category.BOOKS, Category.BOOKS, Category.BOOKS, Category.BOOKS, Category.BANK, Category.BANK, Category.BANK]

# Convert text to vectors
train_x_vectors = [nlp(text).vector for text in train_x]

# Train SVM classifier
clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.form.get("text")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Convert text to vector
        text_vector = np.array([nlp(text).vector])

        # Predict category
        prediction = clf_svm.predict(text_vector)[0]

        return jsonify({"prediction": prediction})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
