import pandas as pd
import os
import urllib.request
import zipfile

def create_dataset():
    print("==================================================")
    print("DOWNLOADING SMS SPAM DATASET")
    print("==================================================")
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = "smsspamcollection.zip"
    
    try:
        # Download the dataset
        print("[1/4] Downloading dataset...")
        urllib.request.urlretrieve(url, zip_path)
        print("✅ Download completed")
        
        # Extract the zip file
        print("[2/4] Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("✅ Extraction completed")
        
        # Read the dataset
        print("[3/4] Reading dataset...")
        df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'], encoding='latin-1')
        
        # Save as CSV
        df.to_csv('spam.csv', index=False)
        print(f"✅ Dataset saved as 'spam.csv'")
        print(f"   Total messages: {len(df)}")
        print(f"   Spam messages: {len(df[df['label']=='spam'])}")
        print(f"   Ham messages: {len(df[df['label']=='ham'])}")
        
        # Cleanup
        print("[4/4] Cleaning up...")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists('SMSSpamCollection'):
            os.remove('SMSSpamCollection')
        print("✅ Cleanup completed")
        
        print("\n✅ Dataset ready! Now run: python train_model.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if os.path.exists('spam.csv'):
            print("✅ Using existing spam.csv file")
        else:
            print("❌ Please check your internet connection")

if __name__ == "__main__":
    create_dataset()