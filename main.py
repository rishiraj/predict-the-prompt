import random
import csv
import requests
from sentence_transformers import SentenceTransformer

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

# Get user information
name = input("Enter your name: ")
email = input("Enter your email: ")

# Select 10 random answers from the CSV file
selected_answers = random.sample(answers, 10)

total_score = 0

# Iterate over the selected answers
for i, answer in enumerate(selected_answers, 1):
    print(f"\nAnswer {i}: {answer}")
    predicted_question = input("Enter your predicted question: ")
    
    # Find the actual question for the current answer
    actual_question = questions[answers.index(answer)]
    
    # Encode the predicted and actual questions
    predicted = model.encode(predicted_question, normalize_embeddings=True)
    actual = model.encode(actual_question, normalize_embeddings=True)
    
    # Calculate the similarity score
    similarity = predicted @ actual.T
    score = float(similarity)
    
    print(f"Score: {score}")
    total_score += score

print(f"\nTotal Score: {total_score}")

# Make an API call to submit the user's name, email, and score
url = "https://api.example.com/submit"  # Replace with the actual API endpoint URL
data = {
    "name": name,
    "email": email,
    "total_score": total_score
}

response = requests.post(url, json=data)

if response.status_code == 200:
    print("Score submitted successfully!")
else:
    print("Failed to submit the score.")