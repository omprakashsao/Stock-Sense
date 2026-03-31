# 📈 StockSense — Stock Data Intelligence Dashboard
 
> FastAPI · SQLite · Polynomial Regression ML · Docker · Chart.js

## 🚀 Introduction

StockSense is a stock data intelligence dashboard that simplifies complex market data into clear insights. It provides interactive charts, key metrics, AI-based recommendations, stock comparison, and price prediction. The system processes stock data, calculates indicators like returns and volatility, and delivers insights through APIs and a responsive UI to support better decision-making.


---

<p align="center">
  <img src="./asset/Screenshot (271).png" width="1000" alt="Dashboard Screenshot"/>
</p>

---

## ✅ Bonus Features Implemented

| Feature | Status | Detail |
|---|---|---|
| 🧠 ML Price Prediction | ✅ Done | Polynomial Regression, 95% confidence band, R² score |
| 🧩 Dockerization | ✅ Done | Dockerfile + docker-compose.yml, non-root, multi-stage |
| ⚡ Async API | ✅ Done | All endpoints are async, DB runs in thread pool |
| ⚡ Caching | ✅ Done | In-memory TTL cache (60–300s per endpoint) |
| ☁️ Deployment | 📋 Steps below | Render (free) + GitHub Pages |

---


## 🚀 Quick Start (Local)

    pip install -r requirements.txt
    cd backend 
    python -m uvicorn main:app --reload
    # Open frontend/index.html in browser

---

## 🧩 Docker

    docker compose up --build
    # API at http://localhost:8000

---

## ☁️ Deployment

### Render (Free)
1. Push to GitHub
2. render.com → New Web Service → connect repo
3. Build: `pip install -r requirements.txt`
4. Start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5. Update `const API` in index.html with your Render URL

### GitHub Pages (Frontend)
1. Settings → Pages → Source: main branch, /frontend folder
2. Live at https://yourusername.github.io/stock-dashboard

### Oracle Cloud (Always Free VM)
    sudo apt install -y docker.io docker-compose-plugin
    git clone <your-repo> && cd stock-dashboard
    sudo docker compose up -d
    # Open port 8000 in Oracle security rules

---

## 📡 API Endpoints

| Endpoint | Cache TTL |
|---|---|
| GET /companies | 5 min |
| GET /data/{symbol}?days=30 | 60 s |
| GET /summary/{symbol} | 2 min |
| GET /compare?symbol1=TCS&symbol2=INFY | 2 min |
| GET /gainers-losers?days=7 | 5 min |
| GET /predict/{symbol}?forecast_days=14 | 5 min |
| GET /cache/stats | live |

Swagger docs at /docs

---

## 🧠 ML Model

Polynomial Regression (degree 2) on last 90 days:
- Fits curve, computes residuals → 95% confidence interval
- Forecasts next N business days (weekends skipped)
- Returns R² score, bullish/bearish direction, upper/lower bands

---

## ⚡ Async + Cache Architecture

All endpoints are async. DB queries run in a ThreadPoolExecutor.
@cached(ttl_seconds=N) wraps all DB helpers. Zero Redis needed.

---

## 📬 Author

Name: Om Prakash Sao
GitHub: 
Live: 
Email: saoomprakash2002@gmail.com
