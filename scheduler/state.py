import json
import os
from loguru import logger

STATE_FILE = "scheduler/state.json"

def load_state() -> dict:
    """Load persisted bot state from disk."""
    if not os.path.exists(STATE_FILE):
        state = {
            "usdt":        1000.0,
            "btc":         0.0,
            "total_value": 1000.0,
            "in_position": False,
            "entry_price": None,
            "trade_count": 0,
            "wins":        0,
            "losses":      0,
            "last_run":    None,
        }
        save_state(state)
        logger.info(f"📁 New state created — starting balance: $1,000.00")
        return state

    with open(STATE_FILE, "r") as f:
        state = json.load(f)
    logger.info(f"📁 State loaded — USDT: ${state['usdt']:,.2f} | "
                f"BTC: {state['btc']:.6f} | "
                f"Total: ${state['total_value']:,.2f}")
    return state

def save_state(state: dict):
    """Persist bot state to disk."""
    os.makedirs("scheduler", exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    logger.debug(f"💾 State saved")

def update_state_after_cycle(state: dict, result: dict, broker) -> dict:
    """Update state dict after a trading cycle."""
    from datetime import datetime

    # Update balances from broker
    usdt  = broker.get_balance("USDT")
    btc   = broker.get_balance("BTC")
    total = broker.get_portfolio_value("BTCUSDT")
    pos   = broker.get_open_position("BTCUSDT")

    # Track trade outcomes
    action = result.get("action_name", "NO TRADE")
    if action == "CLOSE" and state.get("in_position"):
        entry = state.get("entry_price") or 0
        current_price = result.get("current_price", 0)
        if current_price > entry:
            state["wins"] += 1
        else:
            state["losses"] += 1
        state["trade_count"] += 1

    state.update({
        "usdt":        usdt,
        "btc":         btc,
        "total_value": total,
        "in_position": pos["in_position"],
        "entry_price": pos.get("entry_price"),
        "last_run":    datetime.now().isoformat(),
    })

    save_state(state)
    return state