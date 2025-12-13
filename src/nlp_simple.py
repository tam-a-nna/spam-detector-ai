# SIMPLE NLP PIPELINE - NO NLTK DOWNLOAD NEEDED
import re

class SimpleNLPPipeline:
    def __init__(self):
        # Common English stopwords (no NLTK needed)
        self.stop_words = {
            'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
            'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being',
            'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't",
            'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during',
            'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't",
            'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here',
            "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i',
            "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's",
            'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself',
            'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought',
            'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she',
            "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such',
            'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves',
            'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're",
            "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',
            'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were',
            "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which',
            'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would',
            "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours',
            'yourself', 'yourselves'
        }
    
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
        """Split into words - simple split"""
        return text.split()
    
    def remove_stopwords(self, tokens):
        """Remove common words"""
        return [word for word in tokens if word not in self.stop_words and len(word) > 2]
    
    def simple_stem(self, word):
        """Very simple stemming"""
        if len(word) < 4:
            return word
        
        # Remove common endings
        if word.endswith('ing'):
            return word[:-3]
        elif word.endswith('ed'):
            return word[:-2]
        elif word.endswith('s'):
            return word[:-1]
        elif word.endswith('ly'):
            return word[:-2]
        return word
    
    def stem_words(self, tokens):
        """Apply simple stemming"""
        return [self.simple_stem(word) for word in tokens]
    
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
        
        # Step 5: Simple stemming
        stemmed = self.stem_words(filtered)
        
        # Join back
        return ' '.join(stemmed)

# Test function
if __name__ == "__main__":
    print("="*60)
    print("Testing Simple NLP Pipeline")
    print("="*60)
    
    nlp = SimpleNLPPipeline()
    
    test_messages = [
        "WINNER!! You won $1000! Call 1234567890 NOW!!!",
        "Hey, meeting tomorrow at 5pm?",
        "URGENT: Verify account http://bank.com"
    ]
    
    for i, msg in enumerate(test_messages, 1):
        print(f"\nTest {i}:")
        print(f"Input : {msg}")
        output = nlp.process_text(msg)
        print(f"Output: {output}")
    
    print("\n" + "="*60)
    print("✅ Simple NLP Pipeline Ready!")
    print("="*60)