# api.py — FastAPI backend for RF_bot_ dashboard
# Run with: uvicorn api:app --reload --port 8000

import os
import sys
import yaml
import threading
import numpy as np
from datetime import datetime
from loguru import logger
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Add project root to path ───────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.fetcher     import MarketDataFetcher
from data.indicators  import IndicatorEngine
from data.normalizer  import DataNormalizer
from ai.inference     import InferenceEngine
from risk.stop_loss   import StopLossManager

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
        self.initial_capital = config["broker"]["initial_capital"]
        self.portfolio_value = self.initial_capital
        self.cash            = self.initial_capital   # tracks liquid cash
        self.position_units  = 0.0                   # units held (+long, -short)
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
        self.sl_manager      = StopLossManager(config)

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
    """Fetch and normalize latest market data, routing by config source."""
    fetcher    = MarketDataFetcher(config)
    ind_engine = IndicatorEngine(config)
    normalizer = DataNormalizer(config)

    data_cfg = config["data"]
    source   = data_cfg.get("source", "csv").lower()
    symbol   = data_cfg.get("symbol", "GOLD")

    if source == "csv":
        csv_path = data_cfg.get("csv_path")
        raw = fetcher.fetch_csv(csv_path, symbol=symbol)
    elif source == "yfinance":
        ticker = data_cfg.get("yfinance_ticker", symbol)
        raw = fetcher.fetch_yfinance(ticker, symbol=symbol)
    elif source == "binance":
        raw = fetcher.fetch_binance(
            symbol   = symbol,
            interval = data_cfg.get("timeframe", "1d"),
            limit    = data_cfg.get("lookback_days", 365),
        )
    else:
        raise ValueError(f"Unknown data source: '{source}'")

    enriched   = ind_engine.compute_all(raw)
    raw_prices = raw["close"].reindex(enriched.index)
    normalized = normalizer.normalize(enriched)
    raw_prices = raw_prices.reindex(normalized.index).dropna()
    normalized = normalized.loc[raw_prices.index]

    return normalized, raw_prices


def _build_portfolio_obs(current_price: float) -> np.ndarray:
    """
    Build the 6-element portfolio state matching TradingEnvironment._get_portfolio_state().
    Must be appended to the market observation before calling predict().
    """
    ic = state.initial_capital
    pv = state.portfolio_value

    cash_ratio     = min(state.cash / ic, 1.0)
    pos_value      = abs(state.position_units) * current_price
    position_ratio = min(pos_value / ic, 1.0)

    if state.position is not None and state.entry_price:
        if state.position == "LONG":
            upct = (current_price - state.entry_price) / state.entry_price
        else:
            upct = (state.entry_price - current_price) / state.entry_price
        unrealized_norm = float(np.clip(upct + 0.5, 0.0, 1.0))
    else:
        unrealized_norm = 0.5

    portfolio_ratio = min(pv / (ic * 2), 1.0)

    if state.position == "LONG":
        pos_dir = 1.0
    elif state.position == "SHORT":
        pos_dir = 0.0
    else:
        pos_dir = 0.5

    has_pos = 1.0 if state.position is not None else 0.0

    return np.array(
        [cash_ratio, position_ratio, unrealized_norm, portfolio_ratio, pos_dir, has_pos],
        dtype=np.float32,
    )

def run_bot_loop():
    """Background thread — runs one tick every 60s."""
    logger.info("🤖 Bot loop started")
    import time

    while state.running:
        try:
            engine = load_engine()
            normalized_df, raw_prices = fetch_latest_data()

            market_obs    = normalized_df.iloc[-1].values.astype(np.float32)
            current_price = float(raw_prices.iloc[-1])
            state.last_price = current_price

            portfolio_obs = _build_portfolio_obs(current_price)
            observation   = np.concatenate([market_obs, portfolio_obs]).astype(np.float32)

            result = engine.predict(
                observation     = observation,
                portfolio_value = state.portfolio_value,
            )

            action     = result["action"]
            confidence = result["confidence"]
            state.last_signal = result
            state.last_tick   = datetime.now().isoformat()

            # Check stop loss / take profit before executing
            if state.position is not None:
                sl_result = state.sl_manager.check(current_price)
                if sl_result["should_exit"]:
                    action = 3
                    logger.warning(
                        f"⚠️ Risk override: {sl_result['reason']} | "
                        f"P&L: {sl_result['pnl_pct']:+.2f}%"
                    )

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
    """Paper trading execution — mirrors TradingEnvironment logic."""
    tc = 0.001   # transaction cost fraction

    if action == 1 and state.position is None:
        # Open LONG: spend cash to buy units
        buy_value          = state.cash * config["risk"].get("max_position_pct", 0.95)
        cost               = buy_value * tc
        units              = (buy_value - cost) / price
        state.cash        -= buy_value
        state.position_units = units
        state.position     = "LONG"
        state.entry_price  = price
        state.total_trades += 1
        state.sl_manager.open_trade(price, "long")

    elif action == 2 and state.position is None:
        # Open SHORT: cash unchanged, negative units
        short_value        = state.cash * config["risk"].get("max_position_pct", 0.95)
        cost               = short_value * tc
        units              = (short_value - cost) / price
        state.position_units = -units
        state.position     = "SHORT"
        state.entry_price  = price
        state.total_trades += 1
        state.sl_manager.open_trade(price, "short")

    elif action == 3 and state.position is not None:
        if state.entry_price:
            if state.position == "LONG":
                sale_value   = state.position_units * price
                cost         = sale_value * tc
                realized_pnl = sale_value - (state.position_units * state.entry_price) - cost
                state.cash  += sale_value - cost
                pnl_pct      = (price - state.entry_price) / state.entry_price * 100
            else:
                units_owed   = abs(state.position_units)
                buyback      = units_owed * price
                cost         = buyback * tc
                realized_pnl = (units_owed * state.entry_price) - buyback - cost
                state.cash  += realized_pnl
                pnl_pct      = (state.entry_price - price) / state.entry_price * 100

            state.portfolio_value = state.cash   # flat — all in cash now
            won = pnl_pct > 0
            if won:
                state.winning_trades += 1

            state.trades.append({
                "id":         state.total_trades,
                "type":       state.position,
                "entry":      round(state.entry_price, 2),
                "exit":       round(price, 2),
                "pnl_pct":    round(pnl_pct, 2),
                "pnl_dollar": round(realized_pnl, 2),
                "won":        won,
                "timestamp":  datetime.now().isoformat(),
            })

        state.sl_manager.close_trade()
        state.position       = None
        state.entry_price    = None
        state.position_units = 0.0

    # Update portfolio value with mark-to-market
    if state.position == "LONG":
        state.portfolio_value = state.cash + (state.position_units * price)
    elif state.position == "SHORT":
        units_owed = abs(state.position_units)
        short_pnl  = (state.entry_price - price) * units_owed
        state.portfolio_value = state.cash + short_pnl

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
        _, raw_prices = fetch_latest_data()
        prices = raw_prices.tail(60).tolist()
        dates  = [str(d)[:10] for d in raw_prices.tail(60).index.tolist()]
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

        current_price = float(raw_prices.iloc[-1])
        market_obs    = normalized_df.iloc[-1].values.astype(np.float32)
        portfolio_obs = _build_portfolio_obs(current_price)
        observation   = np.concatenate([market_obs, portfolio_obs]).astype(np.float32)

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

@app.get("/")
def serve_dashboard():
    """Serve the trading dashboard."""
    return FileResponse("dashboard.html")