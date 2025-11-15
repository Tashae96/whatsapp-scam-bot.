# generate_dataset.py
import random
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)

# templates for each class
legit_templates = [
    "Hey, are we still meeting at 5pm?",
    "Can you send the report by tomorrow morning?",
    "Let's push the deployment to Friday.",
    "Happy birthday! Hope you have a great day :)",
    "Call me when you are free."
]

otp_scams = [
    "I just sent you a code by mistake, can you forward it?",
    "WhatsApp verification code: 123456. Please share the code so I can confirm.",
    "Your account code 987654 — please confirm by sending it here.",
    "This is WhatsApp support. Your OTP is 345678. Please share to verify.",
    "Please send me the 6-digit code I sent you — I need to log in."
]

phishing_scams = [
    "Your parcel is ready for delivery. Click http://fake-delivery.example to confirm.",
    "We detected unusual activity, verify here: http://verify.example",
    "You have won a prize! Click this link to claim your reward: http://scam.example",
    "Bank alert: Your account needs verification. Login at http://bank.example",
]

impersonation = [
    "Hi it's John on a new number — can you confirm my code? 564321",
    "Hey it's Sam, lost my phone. I changed numbers, send me the verification code.",
    "This is your colleague Mark, I'm on a temporary phone — what's the code?"
]

other_scams = [
    "Transfer me $200 now, my account is frozen.",
    "Urgent! Send money to this number or I'll be in trouble.",
    "Investment opportunity, send your account details."
]

def make_messages(n_legit=300, n_otp=150, n_phish=100, n_imp=50, n_other=50):
    records = []
    start = datetime(2025,1,1)
    for i in range(n_legit):
        text = random.choice(legit_templates)
        ts = start + timedelta(minutes=random.randint(0,60*24*90))
        records.append({"text": text, "label": "legit", "scam_type": "none", "timestamp": ts})
    for i in range(n_otp):
        t = random.choice(otp_scams)
        # randomly obfuscate digits
        t = t.replace("123456", str(random.randint(100000,999999)))
        ts = start + timedelta(minutes=random.randint(0,60*24*90))
        records.append({"text": t, "label": "scam", "scam_type": "otp", "timestamp": ts})
    for i in range(n_phish):
        t = random.choice(phishing_scams)
        ts = start + timedelta(minutes=random.randint(0,60*24*90))
        records.append({"text": t, "label": "scam", "scam_type": "phishing", "timestamp": ts})
    for i in range(n_imp):
        t = random.choice(impersonation)
        ts = start + timedelta(minutes=random.randint(0,60*24*90))
        records.append({"text": t, "label": "scam", "scam_type": "impersonation", "timestamp": ts})
    for i in range(n_other):
        t = random.choice(other_scams)
        ts = start + timedelta(minutes=random.randint(0,60*24*90))
        records.append({"text": t, "label": "scam", "scam_type": "other", "timestamp": ts})
    random.shuffle(records)
    return pd.DataFrame(records)

if __name__ == "__main__":
    df = make_messages()
    df.to_csv("messages.csv", index=False)
    print("messages.csv created with", len(df), "rows")
