#!/usr/bin/env python3
# ===========================================
# Alpha Sentiment Engine — Database
# ===========================================
# This file manages the SQLite database.
# SQLite is a simple database that lives as
# a single file on your hard drive. No setup
# needed, no Docker, no passwords.
#
# It stores every single sentiment score so
# we can track trends over time.
# ===========================================

import sqlite3
from datetime import datetime, timezone


# The database file (created automatically)
DB_FILE = "sentiment_data.db"


def get_connection() -> sqlite3.Connection:
    """
    Open a connection to the database.
    If the database file doesn't exist yet, SQLite creates it automatically.
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # So we can access columns by name
    return conn


def create_tables() -> None:
    """
    Create the database tables if they don't already exist.
    This is safe to call multiple times — it won't delete existing data.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # The main table: stores every scored headline
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            headline TEXT NOT NULL,
            score REAL NOT NULL,
            source TEXT,
            scraped_at TEXT NOT NULL
        )
    """)

    # A summary table: stores the average per stock per scrape cycle
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_averages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            average_score REAL NOT NULL,
            num_headlines INTEGER NOT NULL,
            scraped_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def save_score(ticker: str, headline: str, score: float, source: str) -> None:
    """
    Save one scored headline to the database.
    Called once per headline after the AI scores it.
    """
    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now(timezone.utc).isoformat()

    cursor.execute(
        "INSERT INTO sentiment_scores (ticker, headline, score, source, scraped_at) VALUES (?, ?, ?, ?, ?)",
        (ticker, headline, score, source, now),
    )

    conn.commit()
    conn.close()


def save_average(ticker: str, average_score: float, num_headlines: int) -> None:
    """
    Save the average sentiment for a stock after a scrape cycle.
    This is what the dashboard will use to draw trend charts.
    """
    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now(timezone.utc).isoformat()

    cursor.execute(
        "INSERT INTO sentiment_averages (ticker, average_score, num_headlines, scraped_at) VALUES (?, ?, ?, ?)",
        (ticker, average_score, num_headlines, now),
    )

    conn.commit()
    conn.close()


def get_recent_scores(ticker: str = None, limit: int = 50) -> list[dict]:
    """
    Get the most recent scored headlines from the database.
    If ticker is provided, filter by that stock.
    """
    conn = get_connection()
    cursor = conn.cursor()

    if ticker:
        cursor.execute(
            "SELECT * FROM sentiment_scores WHERE ticker = ? ORDER BY scraped_at DESC LIMIT ?",
            (ticker, limit),
        )
    else:
        cursor.execute(
            "SELECT * FROM sentiment_scores ORDER BY scraped_at DESC LIMIT ?",
            (limit,),
        )

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_recent_averages(ticker: str = None, limit: int = 100) -> list[dict]:
    """
    Get the most recent average scores (for trend charts).
    If ticker is provided, filter by that stock.
    """
    conn = get_connection()
    cursor = conn.cursor()

    if ticker:
        cursor.execute(
            "SELECT * FROM sentiment_averages WHERE ticker = ? ORDER BY scraped_at DESC LIMIT ?",
            (ticker, limit),
        )
    else:
        cursor.execute(
            "SELECT * FROM sentiment_averages ORDER BY scraped_at DESC LIMIT ?",
            (limit,),
        )

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_all_tickers() -> list[str]:
    """
    Get a list of all unique ticker symbols in the database.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT ticker FROM sentiment_averages ORDER BY ticker")
    rows = cursor.fetchall()
    conn.close()

    return [row["ticker"] for row in rows]


def get_stats() -> dict:
    """
    Get overall database statistics.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as count FROM sentiment_scores")
    total_scores = cursor.fetchone()["count"]

    cursor.execute("SELECT COUNT(*) as count FROM sentiment_averages")
    total_averages = cursor.fetchone()["count"]

    cursor.execute("SELECT COUNT(DISTINCT ticker) as count FROM sentiment_scores")
    unique_tickers = cursor.fetchone()["count"]

    conn.close()

    return {
        "total_scores": total_scores,
        "total_averages": total_averages,
        "unique_tickers": unique_tickers,
    }


# Create the tables when this module is first imported
create_tables()
