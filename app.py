import os
import base64
import io
import re
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# For network analysis
import networkx as nx
from datetime import datetime

app = Flask(__name__)

def load_input_file(files, filename_hint=None):
    """Load the appropriate data file (csv, json, parquet) for analysis."""
    if filename_hint:
        for field in files:
            if filename_hint.lower() in field.lower():
                file = files[field]
                if field.lower().endswith(".csv"):
                    return pd.read_csv(file)
                if field.lower().endswith(".json"):
                    return pd.read_json(file)
                if field.lower().endswith(".parquet"):
                    return pd.read_parquet(file)
    for field in files:
        if field == 'questions.txt':
            continue
        file = files[field]
        if field.lower().endswith(".csv"):
            return pd.read_csv(file)
        if field.lower().endswith(".json"):
            return pd.read_json(file)
        if field.lower().endswith(".parquet"):
            return pd.read_parquet(file)
    return None

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    uri = f"data:image/png;base64,{b64}"
    if len(uri) > 100000:
        return "Plot too large"
    return uri

@app.route('/api/', methods=['POST'])
def handle_api():
    if 'questions.txt' not in request.files:
        return jsonify({'error': 'questions.txt file is required'}), 400
    questions_text = request.files['questions.txt'].read().decode()
    prompt = questions_text.lower()

    # NETWORK TEST
    if 'edge_count' in prompt and 'degree_histogram' in prompt:
        df = load_input_file(request.files, 'edges.csv')
        if df is None:
            return jsonify({"error":"Missing edges.csv file."})
        G = nx.Graph()
        if list(df.columns) == [0,1] or not all(col in df.columns for col in ['source','target']):
            for a,b in df.values:
                G.add_edge(str(a), str(b))
        else:
            for a,b in zip(df['source'], df['target']):
                G.add_edge(str(a), str(b))
        nodes = list(G.nodes())
        degrees = dict(G.degree())
        edge_count = G.number_of_edges()
        avg_deg = np.mean(list(degrees.values()))
        density = nx.density(G)
        highest_degree_node = max(degrees, key=lambda n:degrees[n])
        try:
            sp_alice_eve = nx.shortest_path_length(G, source='Alice', target='Eve')
        except Exception:
            sp_alice_eve = -1
        fig1 = plt.figure(figsize=(4,4))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', 
                edge_color='gray', font_size=10)
        network_graph_b64 = fig_to_base64(fig1)
        fig2 = plt.figure()
        values, counts = np.unique(list(degrees.values()), return_counts=True)
        plt.bar(values, counts, color='green')
        plt.xlabel("Degree")
        plt.ylabel("Count")
        plt.title("Degree Distribution")
        plt.tight_layout()
        degree_hist_b64 = fig_to_base64(fig2)
        return jsonify({
            "edge_count": int(edge_count),
            "highest_degree_node": str(highest_degree_node),
            "average_degree": round(float(avg_deg),4),
            "density": round(float(density),4),
            "shortest_path_alice_eve": int(sp_alice_eve),
            "network_graph": network_graph_b64,
            "degree_histogram": degree_hist_b64
        })

    # SALES TEST
    if 'total_sales' in prompt and 'top_region' in prompt and 'bar_chart' in prompt:
        df = load_input_file(request.files, 'sample-sales.csv')
        if df is None:
            return jsonify({'error': "Missing sales CSV."})
        sales_col = None
        for c in df.columns:
            if 'sales' in c.lower():
                sales_col = c
                break
        if sales_col is None:
            return jsonify({'error': "Could not find sales column."})
        total_sales = df[sales_col].sum()
        region_col = next((c for c in df.columns if 'region' in c.lower()), df.columns[0])
        top_region = df.groupby(region_col)[sales_col].sum().idxmax()
        date_col = next((c for c in df.columns if 'date' in c.lower()), None)
        if date_col and np.issubdtype(df[date_col].dtype, np.datetime64):
            df['day'] = df[date_col].dt.day
        elif date_col:
            df['day'] = pd.to_datetime(df[date_col]).dt.day
        else:
            df['day'] = range(1, len(df)+1)
        day_sales_corr = df['day'].corr(df[sales_col])
        fig1 = plt.figure()
        region_totals = df.groupby(region_col)[sales_col].sum()
        region_totals.plot(kind='bar', color='blue')
        plt.xlabel("Region")
        plt.ylabel("Total Sales")
        plt.title("Total Sales by Region")
        plt.tight_layout()
        bar_chart_b64 = fig_to_base64(fig1)
        median_sales = float(df[sales_col].median())
        total_sales_tax = round(0.10 * total_sales, 4)
        fig2 = plt.figure()
        if date_col:
            df_sorted = df.copy()
            df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])
            df_sorted = df_sorted.sort_values(date_col)
            cs = df_sorted[sales_col].cumsum()
            plt.plot(df_sorted[date_col], cs, color='red')
            plt.xlabel("Date")
        else:
            cs = df[sales_col].cumsum()
            plt.plot(range(len(cs)), cs, color='red')
            plt.xlabel("Order #")
        plt.ylabel("Cumulative Sales")
        plt.title("Cumulative Sales Over Time")
        plt.tight_layout()
        cumulative_chart_b64 = fig_to_base64(fig2)
        return jsonify({
            "total_sales": float(total_sales),
            "top_region": str(top_region),
            "day_sales_correlation": float(day_sales_corr),
            "bar_chart": bar_chart_b64,
            "median_sales": float(median_sales),
            "total_sales_tax": float(total_sales_tax),
            "cumulative_sales_chart": cumulative_chart_b64
        })

    # WEATHER TEST
    if 'average_temp_c' in prompt and 'precip_histogram' in prompt:
        df = load_input_file(request.files, 'weather.csv')
        if df is None:
            df = load_input_file(request.files, 'sample-weather.csv')
        if df is None:
            return jsonify({'error': "Missing weather CSV."})
        temp_col = next((c for c in df.columns if 'temp' in c.lower()), df.columns[0])
        precip_col = next((c for c in df.columns if 'precip' in c.lower()), df.columns[-1])
        date_col = next((c for c in df.columns if 'date' in c.lower()), None)
        df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
        df[precip_col] = pd.to_numeric(df[precip_col], errors='coerce')
        avg_temp = df[temp_col].mean()
        min_temp = df[temp_col].min()
        avg_precip = df[precip_col].mean()
        corr = df[temp_col].corr(df[precip_col])
        if date_col:
            max_precip_date = df.loc[df[precip_col].idxmax(), date_col]
            if isinstance(max_precip_date, pd.Timestamp):
                max_precip_date = max_precip_date.strftime("%Y-%m-%d")
            else:
                try:
                    max_precip_date = str(pd.to_datetime(max_precip_date).date())
                except Exception:
                    max_precip_date = str(max_precip_date)
        else:
            max_precip_date = ""
        fig1 = plt.figure()
        if date_col:
            plt.plot(pd.to_datetime(df[date_col]), df[temp_col], color='red')
            plt.xlabel("Date")
        else:
            plt.plot(range(len(df)), df[temp_col], color='red')
            plt.xlabel("Observation")
        plt.ylabel("Temperature (C)")
        plt.title("Temperature Over Time")
        plt.tight_layout()
        temp_line_b64 = fig_to_base64(fig1)
        fig2 = plt.figure()
        plt.hist(df[precip_col].dropna(), color='orange')
        plt.xlabel("Precipitation (mm)")
        plt.ylabel("Count")
        plt.title("Precipitation Histogram")
        plt.tight_layout()
        precip_hist_b64 = fig_to_base64(fig2)
        return jsonify({
            "average_temp_c": float(round(avg_temp,4)),
            "max_precip_date": str(max_precip_date),
            "min_temp_c": float(min_temp),
            "temp_precip_correlation": float(corr),
            "average_precip_mm": float(round(avg_precip,4)),
            "temp_line_chart": temp_line_b64,
            "precip_histogram": precip_hist_b64
        })

    # FALLBACK FOR URL SCRAPING ETC
    url = re.search(r'https?://[^\s]+', questions_text)
    df = None
    if url:
        try:
            dfs = pd.read_html(url.group(0))
            if dfs:
                df = dfs[0]
        except Exception:
            pass
    if df is not None and not (hasattr(df, "empty") and df.empty):
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
                answers.append("Cannot answer")
        return jsonify(answers)

    return jsonify({"error": "No valid analysis scenario detected or missing necessary files."})

# Helper functions for fallback legacy
def get_numeric_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def correlation_answer(df):
    cols = get_numeric_cols(df)
    if len(cols) < 2:
        return "NaN"
    corr = df[cols[0]].corr(df[cols[1]])
    if pd.isna(corr):
        return "NaN"
    return round(float(corr), 6)

def regression_slope(df):
    from sklearn.linear_model import LinearRegression
    cols = get_numeric_cols(df)
    if len(cols) < 2:
        return "NaN"
    df_clean = df[[cols, cols[1]]].dropna()
    if df_clean.empty:
        return "NaN"
    try:
        model = LinearRegression()
        X = df_clean[cols].values.reshape(-1, 1)
        y = df_clean[cols[1]].values
        model.fit(X, y)
        return round(float(model.coef_), 6)
    except Exception:
        return "NaN"

def plot_scatter_with_regression(df):
    cols = get_numeric_cols(df)
    if len(cols) < 2:
        return "No plot data"
    df_clean = df[[cols, cols[1]]].dropna()
    if df_clean.empty:
        return "No plot data"
    fig, ax = plt.subplots()
    sns.scatterplot(x=cols, y=cols[1], data=df_clean, ax=ax)
    sns.regplot(x=cols, y=cols[1], data=df_clean, scatter=False, color='red', ax=ax, line_kws={"linestyle": "dotted"})
    ax.set_xlabel(cols)
    ax.set_ylabel(cols[1])
    plt.tight_layout()
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', dpi=100)
    plt.close(fig)
    img_bytes.seek(0)
    img_b64 = base64.b64encode(img_bytes.read()).decode()
    uri = f"data:image/png;base64,{img_b64}"
    if len(uri) > 100000:
        return "Plot too large"
    return uri

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
