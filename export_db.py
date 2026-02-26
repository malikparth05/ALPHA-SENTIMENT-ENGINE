import sqlite3
import json

def export_to_json():
    print("📦 Packing SQLite database to JSON payload...")
    conn = sqlite3.connect("sentiment_data.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # We only need the averages for the dashboard charts
    cursor.execute("SELECT * FROM sentiment_averages")
    rows = cursor.fetchall()
    
    data = [dict(row) for row in rows]
    
    with open("sentiment_data.json", "w") as f:
        json.dump(data, f)
        
    # Also fetch recent headlines for the news feed
    cursor.execute("SELECT * FROM sentiment_scores ORDER BY id DESC LIMIT 500")
    headline_rows = cursor.fetchall()
    headline_data = [dict(row) for row in headline_rows]
    
    with open("sentiment_headlines.json", "w") as f:
        json.dump(headline_data, f)

    print(f"✅ Successfully exported {len(data)} averages and {len(headline_data)} headlines.")
    
if __name__ == "__main__":
    export_to_json()
