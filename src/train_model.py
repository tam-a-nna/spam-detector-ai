print("=" * 60)
print("SPAM SMS DETECTOR - TRAINING STARTED")
print("=" * 60)

# Step 1: Import libraries
print("\n[1/8] Loading required libraries...")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
print("✅ Libraries loaded")

# Step 2: Load dataset
print("\n[2/8] Loading dataset...")
try:
    # Try reading the dataset
    df = pd.read_csv('spam.csv', encoding='latin-1')
    print(f"✅ Dataset loaded: {len(df)} rows")
    
    # Show first few rows
    print("\nFirst 3 rows of dataset:")
    print(df.head(3))
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("\nPlease make sure 'spam.csv' is in the same folder")
    exit()

# Step 3: Clean and prepare data
print("\n[3/8] Cleaning data...")
# Keep only needed columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Remove duplicates
initial_count = len(df)
df = df.drop_duplicates()
print(f"Removed {initial_count - len(df)} duplicate messages")

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
print(f"✅ Cleaned dataset: {len(df)} messages")
print(f"   - Ham (0): {len(df[df['label']==0])}")
print(f"   - Spam (1): {len(df[df['label']==1])}")

# Step 4: Prepare features
print("\n[4/8] Preparing features...")
X = df['message']  # Text messages
y = df['label']    # Labels (0=ham, 1=spam)

# Step 5: Split data
print("\n[5/8] Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {len(X_train)} messages")
print(f"Testing set: {len(X_test)} messages")

# Step 6: Text to numbers
print("\n[6/8] Converting text to numbers...")
vectorizer = CountVectorizer(stop_words='english', max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"Created {X_train_vec.shape[1]} features")

# Step 7: Train model
print("\n[7/8] Training model...")
model = MultinomialNB()
model.fit(X_train_vec, y_train)
print("✅ Model trained successfully")

# Step 8: Test and save
print("\n[8/8] Testing model...")
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("📊 RESULTS:")
print("=" * 60)
print(f"Model Accuracy: {accuracy:.2%}")
print(f"Correct predictions: {sum(y_pred == y_test)}/{len(y_test)}")

# Save model
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("\n💾 Model saved: spam_model.pkl")
print("💾 Vectorizer saved: vectorizer.pkl")

print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETE!")
print("=" * 60)
print("\nNext command to run: streamlit run app.py")