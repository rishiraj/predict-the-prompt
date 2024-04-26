from flask import Flask, render_template, request
import random
import csv
import requests
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load the CSV file
csv_file = "predict_the_prompt.csv"
questions = []
answers = []
with open(csv_file, "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        questions.append(row["questions"])
        answers.append(row["answers"])

# Initialize the SentenceTransformer model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        selected_answers = random.sample(answers, 10)
        total_score = 0

        for i, answer in enumerate(selected_answers, 1):
            predicted_question = request.form[f"question{i}"]
            actual_question = questions[answers.index(answer)]

            predicted = model.encode(predicted_question, normalize_embeddings=True)
            actual = model.encode(actual_question, normalize_embeddings=True)

            similarity = predicted @ actual.T
            score = float(similarity)
            total_score += score

        # url = "https://api.example.com/submit"  # Replace with the actual API endpoint URL
        # data = {
        #     "name": name,
        #     "email": email,
        #     "total_score": total_score
        # }
        # response = requests.post(url, json=data)

        # if response.status_code == 200:
        #     result = "Score submitted successfully!"
        # else:
        #     result = "Failed to submit the score."

        return render_template("result.html", result="Score submitted successfully!", total_score=total_score)

    selected_answers = random.sample(answers, 10)
    return render_template("index.html", selected_answers=selected_answers)

if __name__ == "__main__":
    app.run(debug=True)