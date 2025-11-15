# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.cluster import KMeans
import joblib
import re
import string

def clean_text(s):
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"\d{4,}", " <NUM> ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s

if __name__ == "__main__":
    df = pd.read_csv("messages.csv")
    df["text_clean"] = df["text"].astype(str).apply(clean_text)

    X = df["text_clean"]
    y = (df["label"] == "scam").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    # TF-IDF
    tf = TfidfVectorizer(ngram_range=(1,2), min_df=2)
    X_train_tfidf = tf.fit_transform(X_train)
    X_test_tfidf = tf.transform(X_test)

    # classifier
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    try:
        print("ROC AUC:", roc_auc_score(y_test, clf.predict_proba(X_test_tfidf)[:,1]))
    except:
        pass

    # clustering similar messages (use all data)
    X_all_tfidf = tf.transform(df["text_clean"])
    # choose number of clusters heuristically: small dataset -> 10 clusters
    k = 10
    km = KMeans(n_clusters=k, random_state=42)
    labels_clusters = km.fit_predict(X_all_tfidf)

    df["cluster"] = labels_clusters
    df.to_csv("messages_with_clusters.csv", index=False)

    # Save models
    joblib.dump(tf, "tfidf_vectorizer.joblib")
    joblib.dump(clf, "scam_classifier.joblib")
    joblib.dump(km, "kmeans_clusters.joblib")
    print("Saved tfidf_vectorizer.joblib, scam_classifier.joblib, kmeans_clusters.joblib")
