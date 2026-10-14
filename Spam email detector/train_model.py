import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Starting spam email detector training...")

text_col = "text"
label_col = "label"

print("Loading CSV file...")
df = pd.read_csv("spam email.csv")

print("\nLabel counts:")
print(df[label_col].value_counts())

print("CSV loaded successfully.")
print("Columns found:", df.columns.tolist())
print("Total emails:", len(df))

df = df[[text_col, label_col]].dropna()
df[text_col] = df[text_col].astype(str)
df[label_col] = df[label_col].astype(int)

X = df[text_col]
y = df[label_col]

print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Converting text to numbers using TF-IDF...")
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training the AI model...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

print("Testing the model...")
y_pred = model.predict(X_test_tfidf)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("Saving model files...")
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nDone! spam_model.pkl and vectorizer.pkl have been created.")
