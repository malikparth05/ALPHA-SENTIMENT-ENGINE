import sqlite3
import json
from flask import Flask, render_template, jsonify
from datetime import datetime, timezone, timedelta

app = Flask(__name__)
DB_PATH = "sentiment_data.db"
STOCKS_FILE = "data/indian_stocks.json"

# Load stock names for display
def load_stock_names():
    try:
        with open(STOCKS_FILE, "r") as f:
            data = json.load(f)
            return {s["symbol"]: s["name"] for s in data["stocks"]}
    except Exception:
        return {}
        
STOCK_NAMES = load_stock_names()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/stats")
def api_stats():
    """Get high-level statistics for the top cards."""
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM sentiment_scores")
    total_headlines = c.fetchone()[0]
    
    # Get number of unique stocks scored today
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    c.execute("SELECT COUNT(DISTINCT ticker) FROM sentiment_averages WHERE scraped_at LIKE ?", (f"{today}%",))
    stocks_scored = c.fetchone()[0]
    
    conn.close()
    
    return jsonify({
        "total_headlines": total_headlines,
        "stocks_scored": stocks_scored,
        "ai_accuracy": "86.39%"
    })

@app.route("/api/overview")
def api_overview():
    """Get the latest sentiment score for all companies (for the bar chart)."""
    conn = get_db_connection()
    c = conn.cursor()
    
    # Get the latest average score for each ticker
    query = """
    SELECT ticker, average_score as score 
    FROM sentiment_averages 
    WHERE id IN (
        SELECT MAX(id) 
        FROM sentiment_averages 
        GROUP BY ticker
    )
    ORDER BY average_score DESC
    """
    
    c.execute(query)
    results = [dict(row) for row in c.fetchall()]
    conn.close()
    
    # Split into bullish and bearish
    bullish = [r for r in results if r['score'] > 0.1][:15] # Top 15 positive
    bearish = [r for r in results if r['score'] < -0.1][-15:] # Top 15 negative
    
    # Add company names
    for items in [bullish, bearish]:
        for item in items:
            item["name"] = STOCK_NAMES.get(item["ticker"], item["ticker"])
            
    return jsonify({
        "bullish": bullish,
        "bearish": bearish[::-1] # Reverse so most negative is first
    })

@app.route("/api/headlines")
def api_headlines():
    """Get the 50 most recently scored headlines (for the live feed)."""
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT ticker, headline, score, source, scraped_at 
        FROM sentiment_scores 
        ORDER BY id DESC LIMIT 50
    ''')
    
    results = [dict(row) for row in c.fetchall()]
    conn.close()
    
    # Format time and add full name
    for r in results:
        try:
            dt = datetime.fromisoformat(r['scraped_at'].replace('Z', '+00:00'))
            r['time_ago'] = dt.strftime("%H:%M")
        except:
            r['time_ago'] = ""
            
        r['name'] = STOCK_NAMES.get(r['ticker'].replace('SECTOR_', ''), r['ticker'])
        
    return jsonify(results)

@app.route("/api/search")
def api_search():
    """Search for a specific company's sentiment data."""
    from flask import request
    query = request.args.get('q', '').upper()
    if not query:
        return jsonify({"error": "No query provided"}), 400
        
    conn = get_db_connection()
    c = conn.cursor()
    
    # Simple search: match ticker exactly or name partially
    # Find matching tickers from STOCK_NAMES
    matches = []
    for ticker, name in STOCK_NAMES.items():
        if query in ticker or query in name.upper():
            matches.append(ticker)
    
    if not matches:
        return jsonify({"results": []})
        
    # Get latest score for the top match
    best_match = matches[0]
    best_name = STOCK_NAMES[best_match]
    
    c.execute('''
        SELECT average_score as score, scraped_at
        FROM sentiment_averages 
        WHERE ticker = ? OR ticker = ? 
        ORDER BY id DESC LIMIT 1
    ''', (best_match, f"SECTOR_{best_match}"))
    
    score_row = c.fetchone()
    
    if score_row:
        result = dict(score_row)
        result['ticker'] = best_match
        result['name'] = best_name
        return jsonify({"results": [result]})
@app.route("/api/company/<ticker>")
def api_company(ticker):
    """Get detailed data for a single company (trend + headlines)."""
    conn = get_db_connection()
    c = conn.cursor()
    
    # 1. Get recent trend (last 10 averages)
    # The scraper saves both direct ticker and SECTOR_ticker scores
    # We will grab history for both and combine
    c.execute('''
        SELECT average_score as score, scraped_at 
        FROM sentiment_averages 
        WHERE ticker = ? OR ticker = ?
        ORDER BY id DESC LIMIT 10
    ''', (ticker, f"SECTOR_{ticker}"))
    
    trend_rows = [dict(row) for row in c.fetchall()]
    
    # Reverse so oldest is first for the chart
    trend = []
    for r in trend_rows[::-1]:
        try:
            dt = datetime.fromisoformat(r['scraped_at'].replace('Z', '+00:00'))
            r['time_label'] = dt.strftime("%H:%M")
        except:
            r['time_label'] = ""
        trend.append(r)
        
    # 2. Get specific headlines for this company/sector
    # To understand *why* the score is what it is
    # We find the sector from STOCK_NAMES if possible
    # We don't have direct access to the sector mapping array here easily,
    # but the scraper saves sector news as SECTOR_Name.
    
    # Try finding the sector name for this ticker
    # We need to read the JSON for the full mapping since STOCK_NAMES is just symbol->name
    try:
        with open(STOCKS_FILE, "r") as f:
            full_data = json.load(f)
            stock_obj = next((s for s in full_data["stocks"] if s["symbol"] == ticker), None)
            sector = stock_obj.get("sector", "General") if stock_obj else "General"
    except:
        sector = "General"

    c.execute('''
        SELECT headline, score, source, scraped_at 
        FROM sentiment_scores 
        WHERE ticker = ? OR ticker = ?
        ORDER BY id DESC LIMIT 15
    ''', (ticker, f"SECTOR_{sector}"))
    
    headlines_rows = [dict(row) for row in c.fetchall()]
    headlines = []
    for r in headlines_rows:
        try:
            dt = datetime.fromisoformat(r['scraped_at'].replace('Z', '+00:00'))
            r['time_ago'] = dt.strftime("%b %d, %H:%M")
        except:
            r['time_ago'] = ""
        headlines.append(r)
        
    conn.close()
    
    # Safely get current score or 0
    current_score = trend[-1]['score'] if trend else 0.0
    
    return jsonify({
        "ticker": ticker,
        "name": STOCK_NAMES.get(ticker, ticker),
        "sector": sector,
        "current_score": current_score,
        "trend": trend,
        "headlines": headlines
    })

if __name__ == "__main__":
    import os
    from services.database import create_tables
    
    # Initialize SQLite schema if new deployment
    create_tables()
    
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", debug=False, port=port)
