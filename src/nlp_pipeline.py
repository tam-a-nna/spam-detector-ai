# NLP PIPELINE FOR SPAM DETECTION
import re
import nltk

print("Setting up NLP pipeline...")

# Download required data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("✅ NLTK data downloaded")
except:
    print("⚠️ Could not download NLTK data")

class NLPPipeline:
    def __init__(self):
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = word_tokenize
    
    def clean_text(self, text):
        """Remove unwanted characters"""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove email
        text = re.sub(r'\S+@\S+', '', text)
        # Remove phone numbers
        text = re.sub(r'\b\d{10,}\b', '', text)
        # Remove special chars, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def to_lowercase(self, text):
        """Convert to lowercase"""
        return text.lower()
    
    def tokenize_text(self, text):
        """Split into words"""
        return self.tokenizer(text)
    
    def remove_stopwords(self, tokens):
        """Remove common words"""
        return [word for word in tokens if word not in self.stop_words and len(word) > 2]
    
    def lemmatize_words(self, tokens):
        """Convert words to base form"""
        return [self.lemmatizer.lemmatize(word) for word in tokens]
    
    def process_text(self, text):
        """Complete NLP processing"""
        # Step 1: Clean
        cleaned = self.clean_text(text)
        if not cleaned:
            return ""
        
        # Step 2: Lowercase
        lower = self.to_lowercase(cleaned)
        
        # Step 3: Tokenize
        tokens = self.tokenize_text(lower)
        
        # Step 4: Remove stopwords
        filtered = self.remove_stopwords(tokens)
        
        # Step 5: Lemmatize
        lemmatized = self.lemmatize_words(filtered)
        
        # Join back
        return ' '.join(lemmatized)

# Test
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing NLP Pipeline")
    print("="*60)
    
    # Create pipeline
    nlp = NLPPipeline()
    
    # Test messages
    tests = [
        "WINNER!! You won $1000! Call 1234567890 NOW!!!",
        "Hey, meeting tomorrow at 5pm?",
        "URGENT: Verify account http://bank.com"
    ]
    
    for i, msg in enumerate(tests, 1):
        print(f"\nTest {i}:")
        print(f"Input : {msg}")
        output = nlp.process_text(msg)
        print(f"Output: {output}")
    
    print("\n" + "="*60)
    print("✅ NLP Pipeline ready!")
    print("="*60)