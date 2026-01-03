import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

model, vectorizer = joblib.load("sentiment_model.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]
    return " ".join(words)

print("🎬 Movie Review Sentiment Analyzer")
review = input("Enter your movie review: ")

cleaned = clean_text(review)
vector = vectorizer.transform([cleaned])
prediction = model.predict(vector)

if prediction[0] == 1:
    print("✅ Sentiment: POSITIVE")
else:
    print("❌ Sentiment: NEGATIVE")
