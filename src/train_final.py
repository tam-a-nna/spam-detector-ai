print("="*70)
print("SPAM DETECTOR - SIMPLE VERSION")
print("="*70)

print("\nStep 1: Installing/checking requirements...")
try:
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib
    import os
    print("✅ All libraries imported")
except ImportError as e:
    print(f"❌ Missing library: {e}")
    print("Run: pip install pandas scikit-learn joblib")
    exit()

print("\nStep 2: Loading dataset...")
try:
    # Try multiple possible locations
    dataset_paths = [
        'data/spam.csv',      # data folder
        'spam.csv',           # current folder
        '../data/spam.csv',   # parent's data folder
        '../spam.csv',        # parent folder
    ]
    
    df = None
    for path in dataset_paths:
        if os.path.exists(path):
            df = pd.read_csv(path, encoding='latin-1')
            print(f"✅ Loaded from: {path}")
            break
    
    if df is None:
        print("❌ Could not find spam.csv")
        print("Make sure spam.csv is in one of these locations:")
        print("1. data/spam.csv (recommended)")
        print("2. spam.csv (current folder)")
        print("3. ../data/spam.csv")
        print("4. ../spam.csv")
        
        # Try to download/create dataset
        print("\n🔄 Attempting to create sample dataset...")
        sample_data = {
            'label': ['ham', 'ham', 'spam', 'ham', 'spam'] * 100,
            'message': [
                'Hey, meeting tomorrow?',
                'Dont forget to buy milk',
                'WINNER!! You won $1000 prize!',
                'Lets have dinner at 8pm',
                'URGENT: Your account needs verification'
            ] * 100
        }
        df = pd.DataFrame(sample_data)
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/spam.csv', index=False)
        print("✅ Created sample dataset with 500 messages")
    
    print(f"Total messages: {len(df)}")
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

print("\nStep 3: Importing NLP pipeline...")
try:
    from nlp_simple import SimpleNLPPipeline
    nlp = SimpleNLPPipeline()
    print("✅ NLP pipeline loaded")
except:
    print("❌ Could not load NLP pipeline")
    print("Make sure nlp_simple.py is in the same folder")
    exit()

print("\nStep 4: Processing messages...")
# Keep only needed columns
if 'v1' in df.columns and 'v2' in df.columns:
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    print("✅ Renamed columns")

# Apply NLP
df['clean_text'] = df['message'].apply(nlp.process_text)

# Convert labels
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
print(f"✅ Processed {len(df)} messages")
print(f"   Ham (0): {sum(df['label_num']==0)}")
print(f"   Spam (1): {sum(df['label_num']==1)}")

print("\nStep 5: Creating features (TF-IDF)...")
X = df['clean_text']
y = df['label_num']

# Create TF-IDF features
tfidf = TfidfVectorizer(max_features=2000)
X_tfidf = tfidf.fit_transform(X)
print(f"✅ Created {X_tfidf.shape[1]} features")

print("\nStep 6: Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)
print(f"✅ Training: {X_train.shape[0]} messages")
print(f"✅ Testing:  {X_test.shape[0]} messages")

print("\nStep 7: Training model...")
model = MultinomialNB()
model.fit(X_train, y_train)
print("✅ Model trained")

print("\nStep 8: Testing accuracy...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.2%}")

print("\nStep 9: Saving everything...")
# Create models folder if it doesn't exist
os.makedirs('models', exist_ok=True)

joblib.dump(model, 'models/spam_model.pkl')
joblib.dump(tfidf, 'models/tfidf.pkl')
joblib.dump(nlp, 'models/nlp_processor.pkl')

print("✅ models/spam_model.pkl saved")
print("✅ models/tfidf.pkl saved")
print("✅ models/nlp_processor.pkl saved")

print("\n" + "="*70)
print("🎉 PROJECT READY!")
print(f"Model Accuracy: {accuracy:.2%}")
print("\nTo run the app:")
print("streamlit run src/app.py")
print("="*70)

# Quick test
print("\nQuick test with sample messages:")
test_msgs = [
    "WINNER!! You won prize money!",
    "Hey, lets meet for coffee",
    "URGENT: Your account is locked"
]

for msg in test_msgs:
    processed = nlp.process_text(msg)
    features = tfidf.transform([processed])
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    
    result = "SPAM" if pred == 1 else "HAM"
    confidence = prob[1]*100 if pred == 1 else prob[0]*100
    
    print(f"'{msg[:30]}...' → {result} ({confidence:.1f}% confidence)")