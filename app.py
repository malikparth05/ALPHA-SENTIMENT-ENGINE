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

@app.route("/api/debug")
def debug_sync():
    import os, json, traceback
    from services.database import create_tables
    logs = []
    try:
        logs.append("Starting debug sync...")
        create_tables()
        logs.append("Tables initialized.")
        
        if os.path.exists("sentiment_data.json"):
            logs.append("Found sentiment_data.json")
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("DELETE FROM sentiment_averages")
            c.execute("DELETE FROM sentiment_scores")
            
            with open("sentiment_data.json", "r") as f:
                data = json.load(f)
                logs.append(f"Loaded {len(data)} averages from JSON.")
                for r in data:
                    c.execute("INSERT INTO sentiment_averages (ticker, average_score, num_headlines, confidence, price_change, score_type, scraped_at) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                              (r['ticker'], r['average_score'], r['num_headlines'], r.get('confidence', 'LOW'), r.get('price_change'), r.get('score_type', 'DIRECT'), r['scraped_at']))
            
            if os.path.exists("sentiment_headlines.json"):
                logs.append("Found sentiment_headlines.json")
                with open("sentiment_headlines.json", "r") as f:
                    data2 = json.load(f)
                    logs.append(f"Loaded {len(data2)} headlines from JSON.")
                    for r in data2:
                        c.execute("INSERT INTO sentiment_scores (ticker, headline, score, source, validated, scraped_at) VALUES (?, ?, ?, ?, ?, ?)", 
                                  (r['ticker'], r['headline'], r['score'], r['source'], r.get('validated', 1), r['scraped_at']))
            
            conn.commit()
            
            c.execute("SELECT COUNT(*) FROM sentiment_scores")
            total = c.fetchone()[0]
            logs.append(f"Sync complete. DB sentiment_scores count is now: {total}")
            conn.close()
        else:
            logs.append("sentiment_data.json DOES NOT EXIST in this container.")
    except Exception as e:
        logs.append(f"FATAL Exception: {str(e)}")
        logs.append(traceback.format_exc())
    
    return jsonify({"debug_logs": logs})

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/stats")
def api_stats():
    """Get high-level statistics for the top cards."""
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM sentiment_scores WHERE validated = 1")
    total_headlines = c.fetchone()[0]
    
    # Get number of unique stocks scored (excluding sector entries)
    c.execute("SELECT COUNT(DISTINCT ticker) FROM sentiment_averages WHERE ticker NOT LIKE 'SECTOR_%'")
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
    SELECT ticker, average_score as score, confidence, price_change, score_type 
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
    
    # Filter out SECTOR_ entries from display
    results = [r for r in results if not r['ticker'].startswith('SECTOR_')]
    
    # Prioritize DIRECT and HYBRID scores over SECTOR-only
    direct_hybrid = [r for r in results if r.get('score_type') in ('DIRECT', 'HYBRID')]
    sector_only = [r for r in results if r.get('score_type') == 'SECTOR']
    
    # Bullish: DIRECT/HYBRID first, then fill with SECTOR if needed
    dh_bullish = [r for r in direct_hybrid if r['score'] > 0.1]
    s_bullish = [r for r in sector_only if r['score'] > 0.1]
    bullish = (dh_bullish + s_bullish)[:15]
    
    # Bearish: same priority
    dh_bearish = sorted([r for r in direct_hybrid if r['score'] < -0.1], key=lambda x: x['score'])
    s_bearish = sorted([r for r in sector_only if r['score'] < -0.1], key=lambda x: x['score'])
    bearish = (dh_bearish + s_bearish)[:15]
    
    # Add company names
    for items in [bullish, bearish]:
        for item in items:
            item["name"] = STOCK_NAMES.get(item["ticker"], item["ticker"])
            
    return jsonify({
        "bullish": bullish,
        "bearish": bearish
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
    
    # Find matching tickers from STOCK_NAMES
    matches = []
    for ticker, name in STOCK_NAMES.items():
        if query in ticker or query in name.upper():
            matches.append(ticker)
    
    if not matches:
        conn.close()
        return jsonify({"results": []})
        
    # Get latest score for top 10 matches
    results = []
    for match_ticker in matches[:10]:
        c.execute('''
            SELECT average_score as score, confidence, price_change, score_type, scraped_at
            FROM sentiment_averages 
            WHERE ticker = ?
            ORDER BY id DESC LIMIT 1
        ''', (match_ticker,))
        
        score_row = c.fetchone()
        if score_row:
            result = dict(score_row)
            result['ticker'] = match_ticker
            result['name'] = STOCK_NAMES[match_ticker]
            results.append(result)
    
    conn.close()
    return jsonify({"results": results})

@app.route("/api/company/<ticker>")
def api_company(ticker):
    """Get detailed data for a single company (trend + headlines)."""
    conn = get_db_connection()
    c = conn.cursor()
    
    # 1. Get recent trend (last 10 averages) — direct ticker only
    c.execute('''
        SELECT average_score as score, confidence, price_change, score_type, scraped_at 
        FROM sentiment_averages 
        WHERE ticker = ?
        ORDER BY id DESC LIMIT 10
    ''', (ticker,))
    
    trend_rows = [dict(row) for row in c.fetchall()]
    
    trend = []
    for r in trend_rows[::-1]:
        try:
            dt = datetime.fromisoformat(r['scraped_at'].replace('Z', '+00:00'))
            r['time_label'] = dt.strftime("%H:%M")
        except:
            r['time_label'] = ""
        trend.append(r)
        
    # 2. Get headlines for this company (direct only, entity-validated)
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
        WHERE ticker = ?
        ORDER BY id DESC LIMIT 15
    ''', (ticker,))
    
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
    
    current_score = trend[-1]['score'] if trend else 0.0
    latest_confidence = trend[-1].get('confidence', 'LOW') if trend else 'LOW'
    latest_price = trend[-1].get('price_change') if trend else None
    
    return jsonify({
        "ticker": ticker,
        "name": STOCK_NAMES.get(ticker, ticker),
        "sector": sector,
        "current_score": current_score,
        "confidence": latest_confidence,
        "price_change": latest_price,
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
