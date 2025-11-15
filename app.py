# app.py
import os
import re
import logging
from flask import Flask, request, Response
from twilio.twiml.messaging_response import MessagingResponse
import joblib
from dotenv import load_dotenv
from datetime import datetime

# Load .env if present
load_dotenv()

# Config
MODEL_PATH = os.getenv("MODEL_PATH", "scam_classifier.joblib")
VECT_PATH = os.getenv("VECT_PATH", "tfidf_vectorizer.joblib")
LOG_PATH = os.getenv("LOG_PATH", "classify_log.csv")

# Twilio creds (only needed if proactively sending messages)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")

# Basic logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whatsapp-scam-bot")

app = Flask(__name__)

# ----------- HOMEPAGE ROUTE -----------------
@app.route("/")
def home():
    return "WhatsApp Scam Detection Bot is running!"

# Load ML artifacts
try:
    clf = joblib.load(MODEL_PATH)
    tf = joblib.load(VECT_PATH)
    logger.info("Loaded model and vectorizer.")
except Exception as e:
    logger.exception("Failed to load model or vectorizer. Ensure files exist.")
    raise


# ----------- CLEAN TEXT -----------------
def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"http\S+", " <URL> ", s)
    s = re.sub(r"\d{3,}", " <NUM> ", s)
    s = re.sub(r"[^\w\s<>]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ----------- LOGGING -----------------
def log_interaction(sender: str, text: str, pred: str, prob: float):
    try:
        import hashlib
        sender_hash = hashlib.sha256(sender.encode("utf-8")).hexdigest()[:16]
        ts = datetime.utcnow().isoformat()
        row = f'"{ts}","{sender_hash}","{pred}","{prob:.4f}","{text.replace("\"","\'")}"\n'
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            if f.tell() == 0:
                f.write("timestamp,sender_hash,prediction,probability,text\n")
            f.write(row)
    except Exception as e:
        logger.exception("Failed to log interaction")

# ----------- RATE LIMITING -----------------
RATE_LIMIT = {}
RATE_LIMIT_WINDOW = 60
MAX_PER_WINDOW = 6

def check_rate_limit(sender: str) -> bool:
    import time
    now = int(time.time())
    window_start = now - RATE_LIMIT_WINDOW
    recs = RATE_LIMIT.get(sender, [])
    recs = [t for t in recs if t >= window_start]
    if len(recs) >= MAX_PER_WINDOW:
        RATE_LIMIT[sender] = recs
        return False
    recs.append(now)
    RATE_LIMIT[sender] = recs
    return True

# ----------- TWILIO WEBHOOK -----------------
@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    sender = request.form.get("From", "")
    body = request.form.get("Body", "")
    logger.info("Incoming message from %s", sender)

    if not check_rate_limit(sender):
        resp = MessagingResponse()
        resp.message("⚠ Too many requests. Please try again later.")
        return Response(str(resp), mimetype="application/xml")

    cleaned = clean_text(body)
    if cleaned.strip() == "":
        resp = MessagingResponse()
        resp.message("Please send a text describing the suspicious message.")
        return Response(str(resp), mimetype="application/xml")

    try:
        vec = tf.transform([cleaned])
        prob = float(clf.predict_proba(vec)[0,1])
        label = "SCAM" if prob >= 0.5 else "LEGIT"
    except Exception as e:
        logger.exception("Model error")
        resp = MessagingResponse()
        resp.message("⚠ Error analyzing the message.")
        return Response(str(resp), mimetype="application/xml")

    advice = []
    if label == "SCAM":
        advice.append("⚠ This message looks suspicious.")
        advice.append(f"Risk score: {prob:.2f}")
        advice.append("Do NOT share verification codes.")
        advice.append("Do NOT click unknown links.")
        advice.append("Consider blocking the sender.")
    else:
        advice.append("✅ This message appears legitimate.")
        advice.append(f"Risk score: {prob:.2f}")

    reply_text = "\n".join(advice)

    try:
        log_interaction(sender, body, label, prob)
    except Exception:
        pass

    resp = MessagingResponse()
    resp.message(reply_text)
    return Response(str(resp), mimetype="application/xml")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
