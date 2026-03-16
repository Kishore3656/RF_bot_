# api.py — FastAPI backend for RF_bot_ dashboard
# Run with: uvicorn api:app --reload --port 8000

import os
import sys
import yaml
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ── Add project root to path ───────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.fetcher     import MarketDataFetcher
from data.indicators  import IndicatorEngine
from data.normalizer  import DataNormalizer
from ai.inference     import InferenceEngine

# ── Load config ────────────────────────────────
with open("config.yaml") as f:
    config = yaml.safe_load(f)

app = FastAPI(title="RF_bot_ Dashboard API", version="1.0.0")

# Allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global bot state ───────────────────────────
class BotState:
    def __init__(self):
        self.running         = False
        self.mode            = config["broker"].get("mode", "paper")
        self.portfolio_value = config["broker"]["initial_capital"]
        self.initial_capital = config["broker"]["initial_capital"]
        self.position        = None
        self.entry_price     = None
        self.total_trades    = 0
        self.winning_trades  = 0
        self.trades          = []
        self.last_signal     = None
        self.last_price      = None
        self.last_tick       = None
        self.engine          = None
        self.thread          = None

state = BotState()

# ── Pydantic models ────────────────────────────
class ModeRequest(BaseModel):
    mode: str   # "paper" or "live"

# ── Helpers ────────────────────────────────────
def load_engine():
    """Load inference engine once."""
    if state.engine is None:
        state.engine = InferenceEngine(config)
        state.engine.load_model()
    return state.engine

def fetch_latest_data():
    """Fetch and normalize latest market data."""
    fetcher    = MarketDataFetcher(config)
    ind_engine = IndicatorEngine(config)
    normalizer = DataNormalizer(config)

    raw_df, raw_prices = fetcher.fetch()
    df_with_indicators = ind_engine.compute_all(raw_df)
    normalized_df      = normalizer.normalize(df_with_indicators)

    return normalized_df, raw_prices

def run_bot_loop():
    """Background thread — runs one tick every 60s."""
    logger.info("🤖 Bot loop started")
    import time

    while state.running:
        try:
            engine = load_engine()
            normalized_df, raw_prices = fetch_latest_data()

            observation   = normalized_df.iloc[-1].values.astype(np.float32)
            current_price = float(raw_prices.iloc[-1])
            state.last_price = current_price

            result = engine.predict(
                observation     = observation,
                portfolio_value = state.portfolio_value,
            )

            action     = result["action"]
            confidence = result["confidence"]
            state.last_signal = result
            state.last_tick   = datetime.now().isoformat()

            # Execute in paper mode
            if state.mode == "paper":
                _execute_paper(action, current_price)

            logger.info(
                f"[{state.last_tick}] "
                f"Price: ${current_price:,.2f} | "
                f"Signal: {result['action_name']} ({confidence}%)"
            )

        except Exception as e:
            logger.error(f"Bot loop error: {e}")

        time.sleep(60)

    logger.info("🛑 Bot loop stopped")

def _execute_paper(action: int, price: float):
    """Paper trading execution."""
    if action == 1 and state.position is None:
        state.position    = "LONG"
        state.entry_price = price
        state.total_trades += 1

    elif action == 2 and state.position is None:
        state.position    = "SHORT"
        state.entry_price = price
        state.total_trades += 1

    elif action == 3 and state.position is not None:
        if state.entry_price:
            if state.position == "LONG":
                pnl_pct = (price - state.entry_price) / state.entry_price * 100
            else:
                pnl_pct = (state.entry_price - price) / state.entry_price * 100

            pnl_dollar = state.portfolio_value * (pnl_pct / 100)
            state.portfolio_value += pnl_dollar

            won = pnl_pct > 0
            if won:
                state.winning_trades += 1

            state.trades.append({
                "id":        state.total_trades,
                "type":      state.position,
                "entry":     round(state.entry_price, 2),
                "exit":      round(price, 2),
                "pnl_pct":   round(pnl_pct, 2),
                "pnl_dollar": round(pnl_dollar, 2),
                "won":       won,
                "timestamp": datetime.now().isoformat(),
            })

        state.position    = None
        state.entry_price = None

# ── API Routes ──────────────────────────────────

@app.get("/status")
def get_status():
    """Full bot status — polled by dashboard every 5s."""
    win_rate = (
        round(state.winning_trades / state.total_trades * 100, 1)
        if state.total_trades > 0 else 0.0
    )
    pnl_pct = round(
        (state.portfolio_value - state.initial_capital)
        / state.initial_capital * 100, 2
    )
    return {
        "running":         state.running,
        "mode":            state.mode,
        "portfolio_value": round(state.portfolio_value, 2),
        "initial_capital": state.initial_capital,
        "pnl_pct":         pnl_pct,
        "position":        state.position,
        "entry_price":     state.entry_price,
        "total_trades":    state.total_trades,
        "winning_trades":  state.winning_trades,
        "win_rate":        win_rate,
        "last_price":      state.last_price,
        "last_tick":       state.last_tick,
        "last_signal":     state.last_signal,
        "symbol":          config["data"]["symbol"],
        "algorithm":       config["ai"]["algorithm"],
        "confidence_threshold": config["ai"].get("confidence_threshold", 55.0),
    }

@app.get("/trades")
def get_trades():
    """Return full trade history."""
    return {"trades": state.trades[-50:]}   # last 50 trades

@app.get("/prices")
def get_prices():
    """Return recent price data for chart."""
    try:
        normalized_df, raw_prices = fetch_latest_data()
        prices  = raw_prices.tail(60).tolist()
        dates   = [str(d)[:10] for d in raw_prices.tail(60).index.tolist()]
        return {"prices": prices, "dates": dates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start")
def start_bot():
    """Start the live trading loop."""
    if state.running:
        return {"message": "Bot already running"}

    state.running = True
    state.thread  = threading.Thread(target=run_bot_loop, daemon=True)
    state.thread.start()

    logger.info("✅ Bot started")
    return {"message": "Bot started", "mode": state.mode}

@app.post("/stop")
def stop_bot():
    """Stop the trading loop."""
    state.running = False
    logger.info("🛑 Bot stopped")
    return {"message": "Bot stopped"}

@app.post("/mode")
def set_mode(req: ModeRequest):
    """Switch between paper and live mode."""
    if req.mode not in ("paper", "live"):
        raise HTTPException(status_code=400, detail="mode must be 'paper' or 'live'")

    if state.running:
        raise HTTPException(status_code=400, detail="Stop bot before switching mode")

    state.mode = req.mode
    config["broker"]["mode"] = req.mode

    logger.info(f"Mode switched to: {req.mode}")
    return {"message": f"Mode set to {req.mode}"}

@app.post("/predict")
def predict_once():
    """Run one inference on latest data — for manual testing."""
    try:
        engine = load_engine()
        normalized_df, raw_prices = fetch_latest_data()

        observation   = normalized_df.iloc[-1].values.astype(np.float32)
        current_price = float(raw_prices.iloc[-1])

        result = engine.predict(
            observation     = observation,
            portfolio_value = state.portfolio_value,
        )

        return {
            "price":  current_price,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}