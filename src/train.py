"""
TRAIN SCRIPT - With accuracy saving
"""
import os
import sys
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

print("="*70)
print("🤖 SPAMSHIELD AI - MODEL TRAINING")
print("="*70)

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

print("\n[1/10] Loading NLP pipeline...")
try:
    from nlp_simple import SimpleNLPPipeline
    nlp = SimpleNLPPipeline()
    print("✅ NLP pipeline loaded")
except ImportError as e:
    print(f"❌ Error: {e}")
    print("Make sure nlp_simple.py is in the same folder as this script")
    exit()

print("\n[2/10] Finding dataset...")
dataset_locations = [
    'data/spam.csv',
    '../data/spam.csv',
    'spam.csv',
    '../spam.csv',
]

df = None
dataset_path = None
for loc in dataset_locations:
    if os.path.exists(loc):
        try:
            df = pd.read_csv(loc, encoding='latin-1')
            dataset_path = loc
            print(f"✅ Dataset found: {loc}")
            print(f"   Messages: {len(df)}")
            break
        except:
            continue

if df is None:
    print("❌ Dataset not found")
    exit()

print("\n[3/10] Preparing data...")
if 'v1' in df.columns and 'v2' in df.columns:
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    print("✅ Renamed columns")

df['clean_text'] = df['message'].apply(nlp.process_text)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

print(f"✅ Processed {len(df)} messages")
print(f"   - Ham: {sum(df['label_num']==0)}")
print(f"   - Spam: {sum(df['label_num']==1)}")

print("\n[4/10] Creating features...")
X = df['clean_text']
y = df['label_num']

vectorizer = TfidfVectorizer(max_features=2000)
X_features = vectorizer.fit_transform(X)
print(f"✅ Created {X_features.shape[1]} features")

print("\n[5/10] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Training: {X_train.shape[0]} messages")
print(f"✅ Testing:  {X_test.shape[0]} messages")

print("\n[6/10] Training model...")
model = MultinomialNB()
model.fit(X_train, y_train)
print("✅ Model trained")

print("\n[7/10] Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Detailed evaluation
print(f"\n📊 **ACCURACY: {accuracy:.2%}**")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Confusion Matrix
print("🎯 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"[[{cm[0,0]} {cm[0,1]}]")
print(f" [{cm[1,0]} {cm[1,1]}]]")

print("\n[8/10] Saving models...")
os.makedirs('models', exist_ok=True)

model_path = 'models/spam_model.pkl'
vectorizer_path = 'models/tfidf.pkl'
nlp_path = 'models/nlp_processor.pkl'

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)
joblib.dump(nlp, nlp_path)

print(f"✅ Model saved: {model_path}")
print(f"✅ Vectorizer saved: {vectorizer_path}")
print(f"✅ NLP processor saved: {nlp_path}")

print("\n[9/10] Saving accuracy statistics...")
results = {
    'accuracy': float(accuracy),
    'accuracy_percentage': f"{accuracy:.2%}",
    'total_messages': len(df),
    'training_messages': X_train.shape[0],
    'testing_messages': X_test.shape[0],
    'ham_count': int(sum(df['label_num']==0)),
    'spam_count': int(sum(df['label_num']==1)),
    'true_positives': int(cm[1, 1]),  # Correctly identified spam
    'true_negatives': int(cm[0, 0]),  # Correctly identified ham
    'false_positives': int(cm[0, 1]),  # Ham incorrectly marked as spam
    'false_negatives': int(cm[1, 0]),  # Spam incorrectly marked as ham
    'precision': float(cm[1, 1] / (cm[1, 1] + cm[0, 1])) if (cm[1, 1] + cm[0, 1]) > 0 else 0,
    'recall': float(cm[1, 1] / (cm[1, 1] + cm[1, 0])) if (cm[1, 1] + cm[1, 0]) > 0 else 0,
}

with open('models/model_stats.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Statistics saved: models/model_stats.json")

print("\n[10/10] Quick testing...")
test_messages = [
    ("WINNER!! You won $1000 prize! Click now!", "Should be SPAM"),
    ("Hey, are we meeting tomorrow at 3pm?", "Should be HAM"),
    ("URGENT: Your account needs verification", "Should be SPAM"),
    ("Don't forget to buy milk", "Should be HAM")
]

print("\n🧪 Test Results:")
correct = 0
total = len(test_messages)

for msg, expected in test_messages:
    processed = nlp.process_text(msg)
    features = vectorizer.transform([processed])
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    
    result = "SPAM" if pred == 1 else "HAM"
    confidence = prob[1]*100 if pred == 1 else prob[0]*100
    
    # Check if prediction matches expectation
    expected_result = "SPAM" if "SPAM" in expected else "HAM"
    is_correct = result == expected_result
    if is_correct:
        correct += 1
    
    status = "✅" if is_correct else "❌"
    print(f"{status} '{msg[:30]}...' → {result} ({confidence:.1f}%) [{expected_result}]")

print(f"\n📈 Test Accuracy: {correct}/{total} ({correct/total:.0%})")

print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print("="*70)
print(f"📊 **Final Model Accuracy: {accuracy:.2%}**")
print(f"📁 Models saved to: models/")
print(f"📊 Statistics saved to: models/model_stats.json")
print("\n🚀 To run the app:")
print("streamlit run app.py")
print("="*70)