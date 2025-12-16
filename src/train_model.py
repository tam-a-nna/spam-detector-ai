import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import os

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
except:
    pass

print("🤖 Training spam detection model...")

def clean_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join(text.split())

try:
    # Load data
    df = pd.read_csv('data/spam.csv')
    df['clean_text'] = df['message'].apply(clean_text)
    df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})
    
    # ADD EMAIL PHISHING EXAMPLES
    phishing_examples = [
        # Account suspension phishing
        ("your account has been suspended click here to verify details", 1),
        ("account locked due to suspicious activity verify now", 1),
        ("security alert unusual login attempt click to secure", 1),
        ("password reset required suspicious activity detected", 1),
        
        # Payment/invoice scams
        ("invoice payment required click to review and pay", 1),
        ("payment overdue immediate action required", 1),
        ("wire transfer needed for urgent payment", 1),
        ("refund available click to claim your money", 1),
        
        # More phishing patterns
        ("verify your identity to restore account access", 1),
        ("click this link to confirm your information", 1),
        ("urgent action required your account at risk", 1),
        ("immediate verification needed to avoid suspension", 1),
        
        # Legitimate messages (balance the dataset)
        ("meeting scheduled for tomorrow at conference room", 0),
        ("please find attached the project report", 0),
        ("team lunch this friday at restaurant", 0),
        ("project update attached are the weekly reports", 0),
    ]
    
    # Add to dataframe
    phishing_df = pd.DataFrame(phishing_examples, columns=['clean_text', 'label_num'])
    df = pd.concat([df, phishing_df], ignore_index=True)
    
    print(f"📊 Dataset: {len(df)} messages (including phishing examples)")
    print(f"   Spam: {df['label_num'].sum()}")
    print(f"   Ham: {len(df) - df['label_num'].sum()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], 
        df['label_num'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['label_num']
    )
    
    # Create features - IMPORTANT: Include bigrams for phrases like "click here"
    vectorizer = TfidfVectorizer(
        max_features=4000, 
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams
        min_df=2
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model - adjust class weight to catch more spam
    model = LogisticRegression(
        max_iter=1000, 
        random_state=42,
        class_weight='balanced',  # Give more weight to spam class
        C=0.5  # More regularization
    )
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_vec)
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'total_samples': len(df),
        'spam_samples': int(df['label_num'].sum()),
        'ham_samples': int(len(df) - df['label_num'].sum())
    }
    
    # Test on known phishing phrases
    print("\n🧪 Testing on phishing phrases:")
    test_phrases = [
        "your account has been suspended click here to verify",
        "account locked suspicious activity",
        "click here to verify details",
        "verify your account now",
        "immediate action required",
    ]
    
    for phrase in test_phrases:
        clean_phrase = clean_text(phrase)
        vec = vectorizer.transform([clean_phrase])
        prob = model.predict_proba(vec)[0][1]
        pred = model.predict(vec)[0]
        print(f"   '{phrase[:30]}...': {prob:.1%} spam ({'SPAM' if pred==1 else 'HAM'})")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/spam_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    
    with open('models/model_stats.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Results
    print("\n✅ Model trained successfully!")
    print(f"📈 Accuracy:  {metrics['accuracy']:.1%}")
    print(f"📈 Precision: {metrics['precision']:.1%}")
    print(f"📈 Recall:    {metrics['recall']:.1%}")
    print(f"📈 F1-Score:  {metrics['f1_score']:.1%}")
    print("\n💾 Saved to models/ folder")
    
except Exception as e:
    print(f"❌ Training failed: {e}")
    import traceback
    traceback.print_exc()