print("=" * 50)
print("SIMPLE SPAM DETECTOR TRAINING")
print("=" * 50)

print("\n1. Loading data...")
import pandas as pd
df = pd.read_csv('spam.csv')
print(f"✅ Loaded {len(df)} messages")

print("\n2. Preparing data...")
# Convert labels to numbers: ham=0, spam=1
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

print("\n3. Converting text to numbers...")
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label_num']

print("\n4. Training model...")
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X, y)

print("\n5. Testing...")
# Test with some examples
test_messages = [
    "WINNER!! You won prize money!",  # Should be spam
    "Hey, lets meet for coffee",       # Should be ham
    "URGENT: Your account is locked",  # Should be spam
    "What time is the meeting?"        # Should be ham
]

for msg in test_messages:
    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)[0]
    result = "SPAM" if prediction == 1 else "HAM"
    print(f"Message: '{msg[:30]}...' → {result}")

print("\n6. Saving model...")
import joblib
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("✅ Model saved!")

print("\n" + "=" * 50)
print("🎯 TRAINING COMPLETE!")
print("Next: Create web app with 'streamlit run app.py'")
print("=" * 50)