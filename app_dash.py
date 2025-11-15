# app_dash.py
import dash
from dash import html, dcc, Input, Output, State
from dash import dash_table
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# load models and data
tf = joblib.load("tfidf_vectorizer.joblib")
clf = joblib.load("scam_classifier.joblib")
km = joblib.load("kmeans_clusters.joblib")
df = pd.read_csv("messages_with_clusters.csv")
df_sample = df.sample(min(200, len(df)), random_state=42)  # smaller table for UI speed

app = dash.Dash(__name__)
app.layout = html.Div(style={"fontFamily":"Arial","maxWidth":"900px","margin":"auto"}, children=[
    html.H2("WhatsApp Metadata / Scam Message Detector - Demo"),
    html.P("Paste a message and click Classify. The model will predict scam vs legit and show similar messages."),

    dcc.Textarea(id="msg-input", placeholder="Paste message here...", style={"width":"100%","height":"120px"}),
    html.Br(),
    html.Button("Classify", id="classify-btn"),
    html.Div(id="result", style={"marginTop":"20px"}),
    html.Hr(),
    html.H4("Sample Dataset (preview)"),
    dash_table.DataTable(
        id="table-sample",
        columns=[{"name":c,"id":c} for c in ["text","label","scam_type","cluster"]],
        data=df_sample[["text","label","scam_type","cluster"]].to_dict("records"),
        page_size=8,
        style_table={"overflowX":"auto"}
    ),
    html.Hr(),
    html.H4("Cluster Explorer"),
    dcc.Dropdown(id="cluster-select", options=[{"label":str(i),"value":i} for i in sorted(df["cluster"].unique())], placeholder="Choose cluster"),
    html.Div(id="cluster-view")
])

def clean_text(s):
    import re, string
    s = str(s).lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"\d{4,}", " <NUM> ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s

@app.callback(
    Output("result","children"),
    Input("classify-btn","n_clicks"),
    State("msg-input","value"),
    prevent_initial_call=True
)
def classify(n_clicks, text):
    if not text or str(text).strip()=="":
        return html.Div("Please paste a message to classify.", style={"color":"orange"})
    cleaned = clean_text(text)
    vec = tf.transform([cleaned])
    prob = clf.predict_proba(vec)[0,1]
    label = "SCAM" if prob>=0.5 else "LEGIT"
    # find nearest cluster centroid
    Xall = tf.transform(df["text"].astype(str))
    # predict cluster of input via kmeans
    cl = km.predict(vec)[0]
    # compute cosine similarity between input and messages in same cluster
    same = df[df["cluster"]==cl].reset_index()
    if len(same)>0:
        sims = cosine_similarity(vec, tf.transform(same["text"].astype(str))).flatten()
        top_idxs = sims.argsort()[::-1][:5]
        similar = same.loc[top_idxs, ["text","label","scam_type"]].to_dict("records")
    else:
        similar = []
    return html.Div([
        html.P(f"Predicted label: {label} (prob={prob:.2f})", style={"fontWeight":"bold","fontSize":"18px"}),
        html.P(f"Assigned cluster: {cl}"),
        html.H5("Top similar messages from dataset:"),
        html.Ul([html.Li(f"{r['text']} — [{r['label']}, type={r['scam_type']}]") for r in similar]) if similar else html.P("No similar messages found.")
    ])

@app.callback(
    Output("cluster-view","children"),
    Input("cluster-select","value")
)
def show_cluster(cl):
    if cl is None:
        return ""
    subset = df[df["cluster"]==int(cl)].head(15)
    rows = [html.Li(f"{t} — [{lab}, type={st}]") for t,lab,st in zip(subset["text"], subset["label"], subset["scam_type"])]
    return html.Div([
        html.H5(f"Cluster {cl} sample ({len(subset)} shown)"),
        html.Ul(rows)
    ])

if __name__ == "__main__":
    app.run(debug=True)
