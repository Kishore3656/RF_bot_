#!/usr/bin/env python
# push_to_hub.py — Upload trained RF_bot model to HuggingFace Hub
#
# Usage:
#   python push_to_hub.py --repo your-username/rf-gold-trading-bot
#   python push_to_hub.py --repo your-username/rf-gold-trading-bot --private
#
# Requires:  pip install huggingface-hub
# Auth:      huggingface-cli login   OR   set HF_TOKEN env var

import os
import sys
import argparse
import yaml


def push_to_hub(repo_id: str, token: str = None, private: bool = False):
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("❌  Install huggingface-hub first:  pip install huggingface-hub")
        sys.exit(1)

    # ── Check model exists ────────────────────────
    model_path = "models/PPO_trading_bot.zip"
    if not os.path.exists(model_path):
        print(f"❌  No trained model found at {model_path}")
        print("    Run  python main.py  first to train the bot.")
        sys.exit(1)

    # ── Load config for model card ────────────────
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    ai_cfg   = config["ai"]
    data_cfg = config["data"]
    risk_cfg = config["risk"]

    print(f"📤  Pushing to HuggingFace Hub: {repo_id}")
    print(f"    Model: {model_path}")
    print(f"    Private: {private}")

    api = HfApi(token=token)

    # ── Create repo (idempotent) ──────────────────
    create_repo(
        repo_id   = repo_id,
        token     = token,
        private   = private,
        repo_type = "model",
        exist_ok  = True,
    )
    print("✅  Repo ready")

    # ── Upload model weights ──────────────────────
    api.upload_file(
        path_or_fileobj = model_path,
        path_in_repo    = "PPO_trading_bot.zip",
        repo_id         = repo_id,
        repo_type       = "model",
    )
    print("✅  Model weights uploaded")

    # ── Upload config ─────────────────────────────
    api.upload_file(
        path_or_fileobj = "config.yaml",
        path_in_repo    = "config.yaml",
        repo_id         = repo_id,
        repo_type       = "model",
    )
    print("✅  config.yaml uploaded")

    # ── Build model card ──────────────────────────
    ema_list = ", ".join(str(p) for p in config.get("indicators", {}).get("ema_periods", [20,50,200]))
    rsi_list = ", ".join(str(p) for p in config.get("indicators", {}).get("rsi_periods", [7,14]))

    card = f"""---
license: mit
tags:
  - reinforcement-learning
  - trading
  - stable-baselines3
  - {ai_cfg["algorithm"]}
  - finance
  - gold
---

# RF Bot — {ai_cfg["algorithm"]} Trading Agent

A Proximal Policy Optimization (PPO) agent trained with reinforcement learning
to trade **{data_cfg["symbol"]}** on **{data_cfg["timeframe"]}** timeframes.

## Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | {ai_cfg["algorithm"]} |
| Training timesteps | {ai_cfg["training_timesteps"]:,} |
| Training runs (best kept) | {ai_cfg.get("training_runs", 1)} |
| Asset | {data_cfg["symbol"]} |
| Timeframe | {data_cfg["timeframe"]} |
| Confidence threshold | {ai_cfg.get("confidence_threshold", 55)}% |

## Observation Space

The model receives **n_market_features + 6** inputs:

**Market features (normalized 0–1):**
- OHLCV price data
- EMA ({ema_list})
- RSI ({rsi_list})
- MACD (line, signal, histogram)
- Bollinger Bands (upper, middle, lower, position, width)
- ATR, volume ratio, OBV, ROC, CCI, Stochastic

**Portfolio state (6 values):**
- Cash ratio
- Position size ratio
- Unrealized P&L (normalized)
- Portfolio value ratio
- Position direction (0=short, 0.5=flat, 1=long)
- Has open position (0/1)

## Action Space

| Action | Description |
|--------|-------------|
| 0 | HOLD — do nothing |
| 1 | LONG — enter long position |
| 2 | SHORT — enter short position |
| 3 | CLOSE — exit any open position |

## Risk Controls

| Parameter | Value |
|-----------|-------|
| Stop loss | {risk_cfg.get("stop_loss_pct", 0.012)*100:.1f}% |
| Take profit | {risk_cfg.get("take_profit_pct", 0.05)*100:.1f}% |
| Trailing stop | {risk_cfg.get("trailing_stop", True)} ({risk_cfg.get("trailing_pct", 0.05)*100:.0f}%) |
| Max drawdown halt | {risk_cfg.get("max_drawdown_pct", 0.15)*100:.0f}% |
| Max daily loss halt | {risk_cfg.get("max_daily_loss_pct", 0.03)*100:.0f}% |
| Position sizing | {risk_cfg.get("position_sizing", "kelly")} |

## Usage

```python
import numpy as np
from stable_baselines3 import PPO
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(repo_id="{repo_id}", filename="PPO_trading_bot.zip")
model = PPO.load(model_path)

# obs must be shape (n_features + 6,) — all values in [0, 1]
# Last 6 elements are portfolio state
obs = np.zeros(model.observation_space.shape, dtype=np.float32)
obs[-6:] = [1.0, 0.0, 0.5, 0.5, 0.5, 0.0]   # flat portfolio

action, _ = model.predict(obs, deterministic=True)
action_names = {{0: "HOLD", 1: "LONG", 2: "SHORT", 3: "CLOSE"}}
print(f"Action: {{action_names[action]}}")
```

## Training Reward Function

The reward combines:
- **Portfolio change** (weight 0.8) — reward making money
- **Drawdown penalty** (weight 0.6) — punish losses from peak
- **Transaction cost penalty** (weight 0.4) — discourage over-trading
- **Sharpe bonus** (weight 0.4) — reward consistent returns
- **Win/loss trade bonus** (weight 0.8) — directly reward profitable trades
- **Hold penalty** — gentle nudge to avoid excessive inaction

## Disclaimer

This model is for **research and educational purposes only**.
It is not financial advice. Past performance in simulation does not
guarantee future results. Never trade with money you cannot afford to lose.
"""

    api.upload_file(
        path_or_fileobj = card.encode("utf-8"),
        path_in_repo    = "README.md",
        repo_id         = repo_id,
        repo_type       = "model",
    )
    print("✅  Model card (README.md) uploaded")

    url = f"https://huggingface.co/{repo_id}"
    print(f"\n🎉  Done! View your model at:\n    {url}")
    return url


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Push RF_bot trained model to HuggingFace Hub"
    )
    parser.add_argument(
        "--repo", required=True,
        help="HuggingFace repo ID, e.g.  username/rf-gold-trading-bot",
    )
    parser.add_argument(
        "--token", default=None,
        help="HF access token (or set HF_TOKEN env var, or run: huggingface-cli login)",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create a private repo (default: public)",
    )
    args = parser.parse_args()

    token = args.token or os.getenv("HF_TOKEN")
    push_to_hub(args.repo, token=token, private=args.private)
