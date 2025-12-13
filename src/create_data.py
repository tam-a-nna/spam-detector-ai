import pandas as pd
import os

print("Creating clean spam dataset...")

# Sample SMS data - REALISTIC examples
data = [
    # HAM messages (not spam)
    ("ham", "Hey, are we meeting tomorrow at 5pm?"),
    ("ham", "Don't forget to buy milk on your way home"),
    ("ham", "Call me when you're free"),
    ("ham", "Lunch tomorrow? Let me know"),
    ("ham", "Meeting moved to 3pm in conference room"),
    ("ham", "Can you pick up the kids from school?"),
    ("ham", "Running late, be there in 10 mins"),
    ("ham", "Happy birthday! Have a great day"),
    ("ham", "See you at the party tonight"),
    ("ham", "What's the plan for the weekend?"),
    
    # SPAM messages
    ("spam", "WINNER!! You won $1000 cash prize! Call 0909090 to claim"),
    ("spam", "URGENT: Your bank account needs verification. Click: http://fake-bank.com"),
    ("spam", "FREE entry to win $5000! Text WIN to 88888"),
    ("spam", "You have won an iPhone! Claim now: http://freeiphone.com"),
    ("spam", "Congratulations! You won a lottery prize of $10,000"),
    ("spam", "Limited offer: 50% discount on all products. Shop now"),
    ("spam", "Your Netflix account is suspended. Verify: http://netflix-security.com"),
    ("spam", "You have 1 new voicemail. Call 09000 to listen"),
    ("spam", "URGENT: Your PayPal account is locked. Click to unlock"),
    ("spam", "Get free ringtones! Download: http://ringtones4u.com")
]

# Create DataFrame
df = pd.DataFrame(data, columns=['label', 'message'])

# Save to CSV
df.to_csv('spam.csv', index=False)

print("✅ Dataset created successfully!")
print(f"Total messages: {len(df)}")
print(f"Ham messages: {len(df[df['label']=='ham'])}")
print(f"Spam messages: {len(df[df['label']=='spam'])}")
print("\nFirst 5 messages:")
print(df.head())
print("\nFile saved as: spam.csv")