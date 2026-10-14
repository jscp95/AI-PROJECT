import streamlit as st
import joblib

# Load the trained AI model and text converter
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(
    page_title="AI Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

st.title("📧 AI Spam Email Detector")
st.write("Enter an email message below to check whether it is spam or not.")

email_text = st.text_area(
    "Email message",
    height=200,
    placeholder="Paste or type the email text here..."
)

if st.button("Check Email", use_container_width=True):
    if email_text.strip() == "":
        st.warning("Please enter an email message first.")
    else:
        email_tfidf = vectorizer.transform([email_text])

        prediction = model.predict(email_tfidf)[0]
        probabilities = model.predict_proba(email_tfidf)[0]

        spam_probability = probabilities[1] * 100
        ham_probability = probabilities[0] * 100

        st.subheader("Result")

        if prediction == 1:
            st.error(f"🚨 This email is likely SPAM.")
            st.progress(spam_probability / 100)
            st.write(f"Spam probability: **{spam_probability:.2f}%**")
        else:
            st.success(f"✅ This email is likely NOT SPAM.")
            st.progress(ham_probability / 100)
            st.write(f"Not-spam probability: **{ham_probability:.2f}%**")

st.divider()
st.caption("Built using Python, scikit-learn, TF-IDF, Naive Bayes, and Streamlit.")
