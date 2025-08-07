import os
import base64
import io
import re
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for plotting (needed on servers/mac)
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

def extract_url(text):
    """Extract the first URL found in the provided text."""
    match = re.search(r'https?://[^\s]+', text)
    return match.group(0) if match else None

def try_scrape_table(url):
    """Try to scrape the first HTML table from the URL."""
    try:
        dfs = pd.read_html(url)
        if dfs:
            return dfs[0]
    except Exception:
        pass
    return None

def get_numeric_cols(df):
    """Return list of numeric-type columns in the dataframe."""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def correlation_answer(df):
    """Calculate correlation between first two numeric columns if possible."""
    cols = get_numeric_cols(df)
    if len(cols) < 2:
        return "NaN"
    corr = df[cols[0]].corr(df[cols[1]])
    if pd.isna(corr):
        return "NaN"
    return round(float(corr), 6)

def regression_slope(df):
    """Calculate simple linear regression slope between first two numeric columns."""
    from sklearn.linear_model import LinearRegression
    cols = get_numeric_cols(df)
    if len(cols) < 2:
        return "NaN"
    df_clean = df[[cols[0], cols[1]]].dropna()
    if df_clean.empty:
        return "NaN"
    try:
        model = LinearRegression()
        X = df_clean[cols[0]].values.reshape(-1, 1)
        y = df_clean[cols[1]].values
        model.fit(X, y)
        return round(float(model.coef_[0]), 6)
    except Exception:
        return "NaN"

def plot_scatter_with_regression(df):
    """Create scatterplot with dotted red regression line, return base64 PNG URI."""
    cols = get_numeric_cols(df)
    if len(cols) < 2:
        return "No plot data"
    df_clean = df[[cols[0], cols[1]]].dropna()
    if df_clean.empty:
        return "No plot data"
    fig, ax = plt.subplots()
    sns.scatterplot(x=cols[0], y=cols[1], data=df_clean, ax=ax)
    sns.regplot(x=cols[0], y=cols[1], data=df_clean, scatter=False, color='red', ax=ax, line_kws={"linestyle": "dotted"})
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    plt.tight_layout()
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', dpi=100)
    plt.close(fig)
    img_bytes.seek(0)
    img_b64 = base64.b64encode(img_bytes.read()).decode()
    uri = f"data:image/png;base64,{img_b64}"
    # Ensure size is less than 100,000 characters
    if len(uri) > 100000:
        return "Plot too large"
    return uri

@app.route('/api/', methods=['POST'])
def handle_api():
    # Ensure 'questions.txt' is present
    if 'questions.txt' not in request.files:
        return jsonify({'error': 'questions.txt file is required'}), 400

    questions_text = request.files['questions.txt'].read().decode()

    # Extract a URL to scrape if present
    url = extract_url(questions_text)

    df = None
    if url:
        df = try_scrape_table(url)

    if df is None or (hasattr(df, "empty") and df.empty):
        return jsonify(["No data found to analyze"])

    # Split questions by newlines and periods, then strip
    questions = [q.strip() for q in re.split(r'[\n\.]+', questions_text) if q.strip()]

    answers = []
    for q in questions:
        q_lower = q.lower()
        if "correlation" in q_lower:
            answers.append(correlation_answer(df))
        elif "regression slope" in q_lower or ("regression" in q_lower and "slope" in q_lower):
            answers.append(regression_slope(df))
        elif "plot" in q_lower or "scatterplot" in q_lower:
            answers.append(plot_scatter_with_regression(df))
        elif "count" in q_lower:
            answers.append(str(len(df)))
        else:
            # For unknown or unsupported questions, respond gracefully
            answers.append("Cannot answer")

    return jsonify(answers)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
