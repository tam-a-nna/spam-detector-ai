import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import json
import re

print("=" * 60)
print("SPAM SMS DETECTOR - MODEL TRAINING")
print("=" * 60)

def preprocess_text(text):
    """Clean text for training"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def train_model():
    # Check if dataset exists
    print("[1/6] Checking dataset...")
    if not os.path.exists('spam.csv'):
        print("❌ Error: spam.csv not found!")
        print("   Run: python create_dataset.py")
        return False
    
    try:
        # Load dataset
        print("[2/6] Loading dataset...")
        df = pd.read_csv('spam.csv')
        print(f"   ✅ Loaded {len(df)} messages")
        
        # Preprocess
        print("[3/6] Preprocessing text...")
        df['message_clean'] = df['message'].apply(preprocess_text)
        df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['message_clean'], 
            df['label_num'], 
            test_size=0.2, 
            random_state=42
        )
        
        # Vectorize
        print("[4/6] Creating features...")
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train model
        print("[5/6] Training model...")
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_vec, y_train)
        
        # Evaluate
        print("[6/6] Evaluating model...")
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
        
        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/spam_model.pkl')
        joblib.dump(vectorizer, 'models/vectorizer.pkl')
        
        with open('models/model_stats.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Print results
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print(f"📊 Model Performance:")
        print(f"   Accuracy:  {metrics['accuracy']:.1%}")
        print(f"   Precision: {metrics['precision']:.1%}")
        print(f"   Recall:    {metrics['recall']:.1%}")
        print(f"   F1-Score:  {metrics['f1_score']:.1%}")
        print(f"\n📁 Files saved in 'models/' folder:")
        print(f"   • spam_model.pkl (trained model)")
        print(f"   • vectorizer.pkl (feature extractor)")
        print(f"   • model_stats.json (performance metrics)")
        print("\n✅ Now run: streamlit run app.py")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    train_model()