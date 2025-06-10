
# sentiment_recommender_final.py

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score, classification_report

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

df = pd.read_csv('final_sentiment_dataset.csv')
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 4 else 0)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

def recommend_similar(tweet_index, top_n=2):
    tweet_vec = X[tweet_index]
    cosine_sim = cosine_similarity(tweet_vec, X).flatten()
    indices = cosine_sim.argsort()[-top_n-1:-1][::-1]
    return df.iloc[indices][['text', 'sentiment']]

print("\nOriginal Tweet:")
print(df.iloc[0]['text'])
print("Recommended Tweets:")
print(recommend_similar(0))
