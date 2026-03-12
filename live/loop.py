# live/loop.py

import time
import numpy as np
import pandas as pd
from loguru import logger
from datetime import datetime


class LiveTradingLoop:
    """
    Runs the bot in real-time (or paper trading mode).

    Single responsibility: on each tick/day,
    get the latest market data → normalize it →
    ask inference engine for action → execute it.

    Paper mode: logs trades but never sends real orders.
    Live mode:  sends orders via broker (Binance etc.)
    """

    def __init__(self, config: dict, inference_engine, data_pipeline, broker=None):
        """
        config:           full config.yaml dict
        inference_engine: loaded InferenceEngine instance
        data_pipeline:    DataPipeline to fetch + normalize data
        broker:           broker client (None = paper trading)
        """
        self.config           = config
        self.engine           = inference_engine
        self.pipeline         = data_pipeline
        self.broker           = broker

        # ── Settings from config ───────────────────
        broker_cfg            = config["broker"]
        ai_cfg                = config["ai"]

        self.initial_capital  = broker_cfg["initial_capital"]
        self.paper_trading    = broker_cfg.get("paper_trading", True)
        self.symbol           = broker_cfg.get("symbol", "XAUUSD")
        self.confidence_threshold = ai_cfg.get("confidence_threshold", 55.0)
        self.poll_interval    = config.get("scheduler", {}).get("interval_seconds", 60)

        # ── Portfolio state ────────────────────────
        self.portfolio_value  = self.initial_capital
        self.position         = None      # "LONG", "SHORT", or None
        self.entry_price      = None
        self.total_trades     = 0
        self.winning_trades   = 0

        mode = "📄 PAPER" if self.paper_trading else "🔴 LIVE"
        logger.info(f"🔄 LiveTradingLoop initialized | Mode: {mode}")
        logger.info(f"   Symbol:     {self.symbol}")
        logger.info(f"   Capital:    ${self.initial_capital:,.2f}")
        logger.info(f"   Confidence: {self.confidence_threshold}% threshold")

    # ──────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ──────────────────────────────────────────────

    def run_once(self, normalized_df: pd.DataFrame, raw_prices: pd.Series) -> dict:
        """
        Run one tick of the trading loop.

        Called by the scheduler every day/interval.
        Returns a dict with the action taken and portfolio state.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"\n{'='*55}")
        logger.info(f"⏰ Tick: {now}")
        logger.info(f"{'='*55}")

        # ── Step 1: Get latest observation ────────
        # Last row = most recent market state
        observation = normalized_df.iloc[-1].values.astype(np.float32)
        current_price = float(raw_prices.iloc[-1])

        logger.info(f"   Price:      ${current_price:,.2f}")
        logger.info(f"   Portfolio:  ${self.portfolio_value:,.2f}")
        logger.info(f"   Position:   {self.position or 'None'}")

        # ── Step 2: Get inference ──────────────────
        # InferenceEngine already enforces confidence threshold
        # and overrides to HOLD if below threshold
        result = self.engine.predict(
            observation     = observation,
            portfolio_value = self.portfolio_value,
        )

        action      = result["action"]
        action_name = result["action_name"]
        confidence  = result["confidence"]
        overridden  = result.get("overridden", False)

        logger.info(f"   Signal:     {action_name} ({confidence}% confidence)")

        if overridden:
            logger.warning(
                f"   ⚠️  Signal overridden to HOLD — "
                f"confidence {confidence}% < {self.confidence_threshold}%"
            )

        # ── Step 3: Execute action ─────────────────
        trade_result = self._execute_action(
            action        = action,
            current_price = current_price,
        )

        # ── Step 4: Log tick summary ───────────────
        pnl_pct = (self.portfolio_value - self.initial_capital) / self.initial_capital * 100
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0

        logger.info(f"   PnL:        {pnl_pct:+.2f}%")
        logger.info(f"   Trades:     {self.total_trades} | Win rate: {win_rate:.1f}%")

        return {
            "timestamp":       now,
            "price":           current_price,
            "action":          action,
            "action_name":     action_name,
            "confidence":      confidence,
            "overridden":      overridden,
            "portfolio_value": self.portfolio_value,
            "pnl_pct":         round(pnl_pct, 2),
            "position":        self.position,
            "total_trades":    self.total_trades,
            "win_rate":        round(win_rate, 1),
            **trade_result,
        }

    def run_loop(self, normalized_df: pd.DataFrame, raw_prices: pd.Series):
        """
        Continuous loop — runs run_once() every poll_interval seconds.
        Used when the scheduler calls the bot continuously.

        Ctrl+C to stop.
        """
        logger.info(f"🚀 Starting live loop | Interval: {self.poll_interval}s")

        while True:
            try:
                self.run_once(normalized_df, raw_prices)
                logger.info(f"   💤 Sleeping {self.poll_interval}s until next tick...")
                time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                logger.info("🛑 Loop stopped by user.")
                break

            except Exception as e:
                logger.error(f"❌ Error in trading loop: {e}")
                logger.info(f"   Retrying in {self.poll_interval}s...")
                time.sleep(self.poll_interval)

    # ──────────────────────────────────────────────
    # ACTION EXECUTION
    # ──────────────────────────────────────────────

    def _execute_action(self, action: int, current_price: float) -> dict:
        """
        Route the action to the correct handler.

        0 = HOLD  → do nothing
        1 = LONG  → open long position
        2 = SHORT → open short position
        3 = CLOSE → close current position
        """
        if action == 0:
            return self._hold()

        elif action == 1:
            if self.position == "LONG":
                logger.info("   ↩️  Already LONG — skipping duplicate entry")
                return {"trade": "skipped", "reason": "already_long"}
            if self.position == "SHORT":
                self._close_position(current_price)   # close short first
            return self._open_position("LONG", current_price)

        elif action == 2:
            if self.position == "SHORT":
                logger.info("   ↩️  Already SHORT — skipping duplicate entry")
                return {"trade": "skipped", "reason": "already_short"}
            if self.position == "LONG":
                self._close_position(current_price)   # close long first
            return self._open_position("SHORT", current_price)

        elif action == 3:
            if self.position is None:
                logger.info("   ↩️  No open position to close")
                return {"trade": "skipped", "reason": "no_position"}
            return self._close_position(current_price)

        return {"trade": "unknown_action"}

    def _hold(self) -> dict:
        logger.info("   🟡 HOLD — no action taken")
        return {"trade": "hold"}

    def _open_position(self, direction: str, price: float) -> dict:
        """Open a new LONG or SHORT position."""
        self.position    = direction
        self.entry_price = price
        self.total_trades += 1

        emoji = "📈" if direction == "LONG" else "📉"

        if self.paper_trading:
            logger.info(f"   {emoji} [PAPER] OPEN {direction} @ ${price:,.2f}")
        else:
            # Live: send order via broker
            self._send_order(direction, price)
            logger.info(f"   {emoji} [LIVE] OPEN {direction} @ ${price:,.2f}")

        return {
            "trade":     "opened",
            "direction": direction,
            "price":     price,
        }

    def _close_position(self, price: float) -> dict:
        """Close the current open position and calculate PnL."""
        if self.entry_price is None:
            return {"trade": "error", "reason": "no_entry_price"}

        direction = self.position

        # Calculate PnL
        if direction == "LONG":
            pnl_pct = (price - self.entry_price) / self.entry_price * 100
        else:  # SHORT
            pnl_pct = (self.entry_price - price) / self.entry_price * 100

        pnl_dollar = self.portfolio_value * (pnl_pct / 100)
        self.portfolio_value += pnl_dollar

        won = pnl_pct > 0
        if won:
            self.winning_trades += 1

        result_emoji = "✅" if won else "❌"
        mode_tag     = "[PAPER]" if self.paper_trading else "[LIVE]"

        logger.info(
            f"   🔴 {mode_tag} CLOSE {direction} | "
            f"Entry: ${self.entry_price:,.2f} → Exit: ${price:,.2f} | "
            f"PnL: {pnl_pct:+.2f}% (${pnl_dollar:+,.2f}) {result_emoji}"
        )

        if not self.paper_trading:
            self._send_order("CLOSE", price)

        # Reset position state
        self.position    = None
        self.entry_price = None

        return {
            "trade":      "closed",
            "direction":  direction,
            "entry":      self.entry_price,
            "exit":       price,
            "pnl_pct":    round(pnl_pct, 2),
            "pnl_dollar": round(pnl_dollar, 2),
            "won":        won,
        }

    # ──────────────────────────────────────────────
    # BROKER INTEGRATION (Live only)
    # ──────────────────────────────────────────────

    def _send_order(self, direction: str, price: float):
        """
        Send a real order via broker client.
        Only called when paper_trading = False.
        Extend this for Binance / MT5 / etc.
        """
        if self.broker is None:
            logger.error("❌ No broker client configured for live trading!")
            return

        try:
            # Example: self.broker.place_order(direction, price)
            logger.info(f"   📡 Order sent: {direction} @ ${price:,.2f}")
        except Exception as e:
            logger.error(f"❌ Order failed: {e}")

    # ──────────────────────────────────────────────
    # STATUS
    # ──────────────────────────────────────────────

    def get_status(self) -> dict:
        """Return current bot state as a dict."""
        pnl_pct  = (self.portfolio_value - self.initial_capital) / self.initial_capital * 100
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0

        return {
            "portfolio_value":  round(self.portfolio_value, 2),
            "initial_capital":  self.initial_capital,
            "pnl_pct":          round(pnl_pct, 2),
            "position":         self.position,
            "entry_price":      self.entry_price,
            "total_trades":     self.total_trades,
            "winning_trades":   self.winning_trades,
            "win_rate":         round(win_rate, 1),
            "paper_trading":    self.paper_trading,
        }