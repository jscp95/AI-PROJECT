import joblib

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print("===== AI SPAM EMAIL DETECTOR =====")
print("Type an email message and press Enter.")
print("Type 'exit' to quit.\n")

while True:
    email = input("Enter email text: ")

    if email.lower() == "exit":
        print("Program closed.")
        break

    email_tfidf = vectorizer.transform([email])
    prediction = model.predict(email_tfidf)[0]
    probability = model.predict_proba(email_tfidf)[0]

    spam_chance = probability[1] * 100

    if prediction == 1:
        print(f"Result: SPAM  ({spam_chance:.2f}% spam confidence)\n")
    else:
        print(f"Result: NOT SPAM  ({100 - spam_chance:.2f}% ham confidence)\n")
