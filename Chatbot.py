import json 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
import sys 

# dataset 
try:
    with open("data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print("data.json not found. Place data.json in the same folder as sample.py")
    sys.exit(1)
except json.JSONDecodeError as e:
    print("data.json is not valid JSON:", e)
    sys.exit(1)

questions = []
answers = []
for item in data:
    q = item.get("question", "")
    a = item.get("answer", "")
    if isinstance(q, str) and q.strip():
        questions.append(q.strip())
        answers.append(a if isinstance(a, str) else "")

if not questions:
    print("No valid questions found in data.json. Add Q&A entries and try again.")
    sys.exit(1) 

# TF IDF vectorizer
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)
print("Chatbot ready for you.. type your question. Type 'exit' or 'quit' to stop conversation.")

THRESHOLD = 0.40

try:
    while True:
        user_input = input("You: ").strip()

        if not user_input:
            print("Bot: Please type a question.")
            continue 

        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Bot: Goodbye! See you soon.")
            break 

        # vectorize and compute similarity
        
        user_vec = vectorizer.transform([user_input])
        sims = cosine_similarity(user_vec, question_vectors)[0]
        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])

        if best_score >= THRESHOLD:
            print("Bot:", answers[best_idx])
        else:
            print("Bot: Sorry, I don't understand the question.")
            print("Bot: Sorry, But i dont understand the Question.")
            
except KeyboardInterrupt:
    print("\nBot: Goodbye! see you soon")
    sys.exit(0)
