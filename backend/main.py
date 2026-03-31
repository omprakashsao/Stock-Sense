"""
Stock Data Intelligence Dashboard — Backend (FastAPI)
JarNox Internship Assignment

Upgrades:
  ⚡ Async endpoints + thread-pool DB access
  ⚡ In-memory TTL cache (no Redis needed)
  🧠 /predict/{symbol} — Polynomial Regression ML model
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os, sqlite3, time, asyncio
from contextlib import contextmanager
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="Stock Data Intelligence Dashboard",
    description="Mini financial data platform — JarNox Internship Assignment",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "stocks.db")
executor = ThreadPoolExecutor(max_workers=4)

# ─────────────────────────────────────────────
# ⚡ TTL Cache
# ─────────────────────────────────────────────
_cache: dict = {}

def cached(ttl_seconds: int = 60):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = (fn.__name__, args, tuple(sorted(kwargs.items())))
            entry = _cache.get(key)
            if entry and (time.monotonic() - entry["ts"]) < ttl_seconds:
                return entry["val"]
            val = fn(*args, **kwargs)
            _cache[key] = {"val": val, "ts": time.monotonic()}
            return val
        return wrapper
    return decorator

# ─────────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────────
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

async def run_in_thread(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: fn(*args, **kwargs))

def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS companies (
                symbol TEXT PRIMARY KEY, name TEXT NOT NULL,
                sector TEXT, exchange TEXT DEFAULT 'NSE'
            );
            CREATE TABLE IF NOT EXISTS stock_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL, date TEXT NOT NULL,
                open REAL, high REAL, low REAL, close REAL,
                volume INTEGER, daily_return REAL, ma7 REAL,
                UNIQUE(symbol, date),
                FOREIGN KEY (symbol) REFERENCES companies(symbol)
            );
        """)
        conn.commit()

# ─────────────────────────────────────────────
# Mock data generation
# ─────────────────────────────────────────────
COMPANIES = [
    {"symbol": "RELIANCE",   "name": "Reliance Industries Ltd",   "sector": "Energy"},
    {"symbol": "TCS",        "name": "Tata Consultancy Services", "sector": "IT"},
    {"symbol": "INFY",       "name": "Infosys Ltd",               "sector": "IT"},
    {"symbol": "HDFCBANK",   "name": "HDFC Bank Ltd",             "sector": "Banking"},
    {"symbol": "ICICIBANK",  "name": "ICICI Bank Ltd",            "sector": "Banking"},
    {"symbol": "WIPRO",      "name": "Wipro Ltd",                 "sector": "IT"},
    {"symbol": "TATAMOTORS", "name": "Tata Motors Ltd",           "sector": "Auto"},
    {"symbol": "BAJFINANCE", "name": "Bajaj Finance Ltd",         "sector": "Finance"},
    {"symbol": "HINDUNILVR", "name": "Hindustan Unilever Ltd",    "sector": "FMCG"},
    {"symbol": "ASIANPAINT", "name": "Asian Paints Ltd",          "sector": "Paint"},
]
BASE_PRICES = {
    "RELIANCE": 2800, "TCS": 3700, "INFY": 1600,
    "HDFCBANK": 1700, "ICICIBANK": 1100, "WIPRO": 550,
    "TATAMOTORS": 950, "BAJFINANCE": 7200,
    "HINDUNILVR": 2600, "ASIANPAINT": 3200,
}
VOLATILITY = {
    "RELIANCE": 0.012, "TCS": 0.011, "INFY": 0.013,
    "HDFCBANK": 0.010, "ICICIBANK": 0.014, "WIPRO": 0.016,
    "TATAMOTORS": 0.020, "BAJFINANCE": 0.018,
    "HINDUNILVR": 0.009, "ASIANPAINT": 0.011,
}

def generate_stock_data(symbol: str, days: int = 365) -> pd.DataFrame:
    np.random.seed(abs(hash(symbol)) % (2**31))
    base, sigma, mu = BASE_PRICES[symbol], VOLATILITY[symbol], 0.0003
    all_dates = pd.bdate_range(end=datetime.today(), periods=days)
    prices = [base]
    for _ in range(len(all_dates) - 1):
        prices.append(prices[-1] * (1 + np.random.normal(mu, sigma)))
    opens   = np.array(prices)
    highs   = opens * (1 + np.abs(np.random.normal(0, sigma * 0.6, len(opens))))
    lows    = opens * (1 - np.abs(np.random.normal(0, sigma * 0.6, len(opens))))
    closes  = opens * (1 + np.random.normal(0, sigma * 0.5, len(opens)))
    volumes = np.random.randint(500_000, 5_000_000, len(opens))
    df = pd.DataFrame({
        "symbol": symbol, "date": all_dates.strftime("%Y-%m-%d"),
        "open": np.round(opens, 2), "high": np.round(highs, 2),
        "low": np.round(lows, 2), "close": np.round(closes, 2),
        "volume": volumes,
    })
    df["daily_return"] = np.round((df["close"] - df["open"]) / df["open"] * 100, 4)
    df["ma7"]          = np.round(df["close"].rolling(7).mean(), 2)
    return df

def seed_database():
    with get_db() as conn:
        if conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0] > 0:
            return
        for c in COMPANIES:
            conn.execute("INSERT OR IGNORE INTO companies VALUES (?,?,?,?)",
                         (c["symbol"], c["name"], c["sector"], "NSE"))
        for sym in BASE_PRICES:
            df = generate_stock_data(sym, 365)
            rows = [
                (r.symbol, r.date, r.open, r.high, r.low, r.close,
                 int(r.volume), r.daily_return,
                 None if pd.isna(r.ma7) else r.ma7)
                for r in df.itertuples(index=False)
            ]
            conn.executemany(
                "INSERT OR IGNORE INTO stock_data "
                "(symbol,date,open,high,low,close,volume,daily_return,ma7) "
                "VALUES (?,?,?,?,?,?,?,?,?)", rows)
        conn.commit()
    print("Database seeded.")

# ─────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────
@app.on_event("startup")
def startup():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    init_db()
    seed_database()

# ─────────────────────────────────────────────
# Cached DB helpers
# ─────────────────────────────────────────────
@cached(ttl_seconds=300)
def _fetch_companies():
    with get_db() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM companies ORDER BY symbol")]

@cached(ttl_seconds=60)
def _fetch_stock_data(symbol: str, days: int):
    with get_db() as conn:
        if not conn.execute("SELECT 1 FROM companies WHERE symbol=?", (symbol,)).fetchone():
            return None
        rows = conn.execute(
            "SELECT date,open,high,low,close,volume,daily_return,ma7 "
            "FROM stock_data WHERE symbol=? ORDER BY date DESC LIMIT ?",
            (symbol, days)).fetchall()
    return [dict(r) for r in reversed(rows)]

@cached(ttl_seconds=120)
def _fetch_summary(symbol: str):
    with get_db() as conn:
        meta = conn.execute("SELECT name,sector FROM companies WHERE symbol=?", (symbol,)).fetchone()
        if not meta:
            return None
        rows = conn.execute(
            "SELECT date,open,close,high,low,daily_return FROM stock_data "
            "WHERE symbol=? ORDER BY date DESC LIMIT 252", (symbol,)).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return None
    return {
        "symbol": symbol, "name": meta["name"], "sector": meta["sector"],
        "52w_high":          round(float(df["high"].max()), 2),
        "52w_low":           round(float(df["low"].min()), 2),
        "avg_close":         round(float(df["close"].mean()), 2),
        "latest_close":      round(float(df.iloc[0]["close"]), 2),
        "latest_date":       df.iloc[0]["date"],
        "volatility_score":  round(float(df["daily_return"].std()), 4),
        "avg_daily_return":  round(float(df["daily_return"].mean()), 4),
        "total_return_pct":  round(
            (df.iloc[0]["close"] - df.iloc[-1]["close"]) / df.iloc[-1]["close"] * 100, 2),
    }

@cached(ttl_seconds=120)
def _fetch_compare(s1: str, s2: str, days: int):
    with get_db() as conn:
        def fetch(sym):
            return conn.execute(
                "SELECT date,close,daily_return FROM stock_data "
                "WHERE symbol=? ORDER BY date DESC LIMIT ?", (sym, days)).fetchall()
        rows1, rows2 = fetch(s1), fetch(s2)
    if not rows1 or not rows2:
        return None
    df1 = pd.DataFrame([dict(r) for r in rows1]).set_index("date")
    df2 = pd.DataFrame([dict(r) for r in rows2]).set_index("date")
    merged = df1.join(df2, lsuffix=f"_{s1}", rsuffix=f"_{s2}", how="inner")
    corr = round(float(merged[f"daily_return_{s1}"].corr(merged[f"daily_return_{s2}"])), 4)
    def perf(df, sym):
        dr, cl = df["daily_return"].dropna(), df["close"]
        return {
            "symbol": sym,
            "latest_close":     round(float(cl.iloc[0]), 2),
            "start_close":      round(float(cl.iloc[-1]), 2),
            "total_return_pct": round((cl.iloc[0]-cl.iloc[-1])/cl.iloc[-1]*100, 2),
            "avg_daily_return": round(float(dr.mean()), 4),
            "volatility":       round(float(dr.std()), 4),
        }
    return {
        "period_days": days, "correlation": corr,
        "interpretation": (
            "Strong positive correlation" if corr > 0.7
            else "Moderate correlation" if corr > 0.3
            else "Low / no correlation"),
        s1: perf(df1, s1), s2: perf(df2, s2),
    }

@cached(ttl_seconds=300)
def _fetch_gainers_losers(days: int):
    with get_db() as conn:
        rows = conn.execute(
            """SELECT s.symbol, c.name,
                      FIRST_VALUE(s.close) OVER (PARTITION BY s.symbol ORDER BY s.date DESC) AS latest,
                      LAST_VALUE(s.close)  OVER (PARTITION BY s.symbol ORDER BY s.date DESC
                           ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS oldest
               FROM stock_data s JOIN companies c USING (symbol)
               WHERE s.date >= date('now', ? || ' days')
               GROUP BY s.symbol, s.date""",
            (f"-{days}",)).fetchall()
    df = pd.DataFrame([dict(r) for r in rows]).drop_duplicates("symbol")
    df["change_pct"] = ((df["latest"] - df["oldest"]) / df["oldest"] * 100).round(2)
    df = df.sort_values("change_pct", ascending=False)
    return {
        "period_days": days,
        "gainers": df.head(3)[["symbol","name","latest","change_pct"]].to_dict("records"),
        "losers":  df.tail(3)[["symbol","name","latest","change_pct"]].iloc[::-1].to_dict("records"),
    }

# ─────────────────────────────────────────────
# 🧠 ML — Polynomial Regression Price Predictor
# ─────────────────────────────────────────────
@cached(ttl_seconds=300)
def _run_prediction(symbol: str, forecast_days: int = 14):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT date,close FROM stock_data WHERE symbol=? "
            "ORDER BY date DESC LIMIT 90", (symbol,)).fetchall()
    if not rows:
        return None
    df     = pd.DataFrame([dict(r) for r in reversed(rows)])
    closes = df["close"].values.astype(float)
    n      = len(closes)
    x      = np.arange(n)

    # Fit degree-2 polynomial
    coeffs  = np.polyfit(x, closes, deg=2)
    poly_fn = np.poly1d(coeffs)
    fitted  = poly_fn(x)

    # 95% confidence interval from residuals
    residuals = closes - fitted
    std_err   = float(np.std(residuals))
    ci_95     = 1.96 * std_err

    # Future business-day dates
    last_date    = datetime.strptime(df["date"].iloc[-1], "%Y-%m-%d")
    future_dates, cursor = [], last_date
    while len(future_dates) < forecast_days:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            future_dates.append(cursor.strftime("%Y-%m-%d"))

    x_future   = np.arange(n, n + forecast_days)
    forecast   = [round(float(poly_fn(xi)), 2) for xi in x_future]
    upper_band = [round(float(poly_fn(xi) + ci_95), 2) for xi in x_future]
    lower_band = [round(float(poly_fn(xi) - ci_95), 2) for xi in x_future]

    slope     = float(coeffs[1] + 2 * coeffs[0] * n)
    ss_res    = float(np.sum(residuals ** 2))
    ss_tot    = float(np.sum((closes - closes.mean()) ** 2))
    r2        = round(1 - ss_res / ss_tot, 4) if ss_tot else 0.0

    return {
        "symbol": symbol,
        "model":  "Polynomial Regression (degree=2)",
        "trained_on_days": n,
        "forecast_days":   forecast_days,
        "r2_score":        r2,
        "direction":       "bullish" if slope > 0 else "bearish",
        "slope":           round(slope, 4),
        "confidence_interval_95": round(ci_95, 2),
        "history": {
            "dates":  df["date"].tolist(),
            "actual": [round(float(v), 2) for v in closes],
            "fitted": [round(float(v), 2) for v in fitted],
        },
        "forecast": {
            "dates":      future_dates,
            "predicted":  forecast,
            "upper_band": upper_band,
            "lower_band": lower_band,
        },
    }

# ─────────────────────────────────────────────
# ── Async Endpoints ──
# ─────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "StockSense API v2.0"}

@app.get("/cache/stats", tags=["Health"])
async def cache_stats():
    now = time.monotonic()
    return {
        "cached_keys": len(_cache),
        "keys": [{"fn": k[0], "age_seconds": round(now - v["ts"], 1)} for k, v in _cache.items()]
    }

@app.get("/companies", tags=["Companies"])
async def get_companies():
    return await run_in_thread(_fetch_companies)

@app.get("/data/{symbol}", tags=["Stock Data"])
async def get_stock_data(symbol: str, days: int = Query(30, ge=7, le=365)):
    symbol = symbol.upper()
    data   = await run_in_thread(_fetch_stock_data, symbol, days)
    if data is None:
        raise HTTPException(404, f"Symbol '{symbol}' not found.")
    return {"symbol": symbol, "days": days, "count": len(data), "data": data}

@app.get("/summary/{symbol}", tags=["Stock Data"])
async def get_summary(symbol: str):
    result = await run_in_thread(_fetch_summary, symbol.upper())
    if not result:
        raise HTTPException(404, f"Symbol '{symbol}' not found.")
    return result

@app.get("/compare", tags=["Stock Data"])
async def compare_stocks(
    symbol1: str = Query(...), symbol2: str = Query(...),
    days: int = Query(30, ge=7, le=365),
):
    result = await run_in_thread(_fetch_compare, symbol1.upper(), symbol2.upper(), days)
    if not result:
        raise HTTPException(404, "Data not found.")
    return result

@app.get("/gainers-losers", tags=["Insights"])
async def top_gainers_losers(days: int = Query(7, ge=1, le=30)):
    return await run_in_thread(_fetch_gainers_losers, days)

@app.get("/predict/{symbol}", tags=["ML Prediction"])
async def predict(symbol: str, forecast_days: int = Query(14, ge=5, le=30)):
    """
    Polynomial Regression prediction — trains on last 90 days,
    forecasts next N trading days with 95% confidence band.
    """
    result = await run_in_thread(_run_prediction, symbol.upper(), forecast_days)
    if not result:
        raise HTTPException(404, f"No data for '{symbol}'.")
    return result

# ─────────────────────────────────────────────
# ── AI Insights ──
# ─────────────────────────────────────────────

@app.get("/insights/{symbol}", tags=["AI Insights"])
async def get_insights(symbol: str):
    """
    Generate simple AI-based insights using existing summary metrics
    """
    summary = await run_in_thread(_fetch_summary, symbol.upper())

    if not summary:
        raise HTTPException(404, f"Symbol '{symbol}' not found.")

    avg_return = summary["avg_daily_return"]
    volatility = summary["volatility_score"]
    total_return = summary["total_return_pct"]

    # 🧠 Decision Logic
    if total_return > 5 and volatility < 1.5:
        recommendation = "BUY"
        confidence = 80
    elif total_return > 0:
        recommendation = "HOLD"
        confidence = 60
    else:
        recommendation = "SELL"
        confidence = 50

    trend = "Bullish" if avg_return > 0 else "Bearish"

    # 🧠 Reasons (very important for creativity)
    reasons = []
    if avg_return > 0:
        reasons.append("Positive average daily return")
    else:
        reasons.append("Negative average daily return")

    if volatility < 1:
        reasons.append("Low volatility (stable stock)")
    elif volatility < 2:
        reasons.append("Moderate volatility")
    else:
        reasons.append("High volatility (risky)")

    if total_return > 0:
        reasons.append("Overall upward price movement")
    else:
        reasons.append("Overall downward trend")

    return {
        "symbol": symbol.upper(),
        "trend": trend,
        "recommendation": recommendation,
        "confidence": confidence,
        "reasons": reasons
    }