import pandas as pd
import urllib.request
import zipfile
import os

def download_dataset():
    print("📥 Downloading SMS spam dataset...")
    
    try:
        # Download file
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        urllib.request.urlretrieve(url, "temp.zip")
        
        # Extract
        with zipfile.ZipFile("temp.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # Load and save
        df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'], encoding='latin-1')
        
        # Create data folder
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/spam.csv', index=False)
        
        # Cleanup
        os.remove("temp.zip")
        if os.path.exists('SMSSpamCollection'):
            os.remove('SMSSpamCollection')
        
        print(f"✅ Dataset saved: data/spam.csv")
        print(f"   Messages: {len(df)}")
        print(f"   Spam: {len(df[df['label']=='spam'])}")
        print(f"   Ham: {len(df[df['label']=='ham'])}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    download_dataset()