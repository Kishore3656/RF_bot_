# risk/stop_loss.py

from loguru import logger


class StopLossManager:
    """
    Monitors open positions and decides when to
    force-exit based on price levels.

    Two types:
    1. Fixed stop loss  → exit if loss exceeds X%
    2. Trailing stop    → exit if price pulls back X%
                          from the BEST price seen

    First principles — trailing stop example:
    Entry: $100
    Price rises to $120 → trailing stop moves to $108 (10% below $120)
    Price falls to $108 → EXIT (locked in +8% gain)

    vs fixed stop:
    Entry: $100, stop at $95 (5% below entry)
    Price rises to $120 → stop stays at $95
    Price falls to $80  → EXIT at $95 (only lost 5%)
    """

    def __init__(self, config: dict):
        risk_cfg = config.get("risk", {})

        self.stop_loss_pct    = risk_cfg.get("stop_loss_pct",    0.05)
        self.take_profit_pct  = risk_cfg.get("take_profit_pct",  0.15)
        self.trailing_stop    = risk_cfg.get("trailing_stop",     False)
        self.trailing_pct     = risk_cfg.get("trailing_pct",      0.07)

        # State for current trade
        self.entry_price      = None
        self.direction        = None   # "long" or "short"
        self.best_price       = None   # for trailing stop

        logger.info(
            f"🛡️ StopLossManager ready | "
            f"SL: {self.stop_loss_pct*100:.0f}% | "
            f"TP: {self.take_profit_pct*100:.0f}% | "
            f"Trailing: {self.trailing_stop}"
        )

    def open_trade(self, entry_price: float, direction: str):
        """
        Call when entering a new position.
        Resets stop/target levels for this trade.
        """
        self.entry_price = entry_price
        self.direction   = direction
        self.best_price  = entry_price

        sl_price, tp_price = self._get_levels(entry_price, direction)

        logger.debug(
            f"   🎯 Trade opened | {direction.upper()} @ ${entry_price:.2f} | "
            f"SL: ${sl_price:.2f} | TP: ${tp_price:.2f}"
        )

    def check(self, current_price: float) -> dict:
        """
        Check if current price has hit stop loss or take profit.

        Returns dict:
        - should_exit: bool
        - reason:      "stop_loss" | "take_profit" | "trailing_stop" | None
        - pnl_pct:     current unrealized P&L %
        """
        if self.entry_price is None:
            return {"should_exit": False, "reason": None, "pnl_pct": 0.0}

        # Update trailing stop best price
        if self.trailing_stop:
            if self.direction == "long":
                self.best_price = max(self.best_price, current_price)
            else:
                self.best_price = min(self.best_price, current_price)

        # Calculate current P&L
        if self.direction == "long":
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:  # short
            pnl_pct = (self.entry_price - current_price) / self.entry_price

        # Check stop loss (fixed)
        if pnl_pct <= -self.stop_loss_pct:
            return {
                "should_exit": True,
                "reason":      "stop_loss",
                "pnl_pct":     round(pnl_pct * 100, 2),
            }

        # Check take profit
        if pnl_pct >= self.take_profit_pct:
            return {
                "should_exit": True,
                "reason":      "take_profit",
                "pnl_pct":     round(pnl_pct * 100, 2),
            }

        # Check trailing stop
        if self.trailing_stop:
            if self.direction == "long":
                trail_stop_price = self.best_price * (1 - self.trailing_pct)
                if current_price <= trail_stop_price:
                    return {
                        "should_exit": True,
                        "reason":      "trailing_stop",
                        "pnl_pct":     round(pnl_pct * 100, 2),
                    }
            else:  # short
                trail_stop_price = self.best_price * (1 + self.trailing_pct)
                if current_price >= trail_stop_price:
                    return {
                        "should_exit": True,
                        "reason":      "trailing_stop",
                        "pnl_pct":     round(pnl_pct * 100, 2),
                    }

        return {
            "should_exit": False,
            "reason":      None,
            "pnl_pct":     round(pnl_pct * 100, 2),
        }

    def close_trade(self):
        """Call when trade is closed. Resets state."""
        self.entry_price = None
        self.direction   = None
        self.best_price  = None

    def _get_levels(
        self, entry_price: float, direction: str
    ) -> tuple:
        """Calculate the actual stop and target price levels."""
        if direction == "long":
            sl = entry_price * (1 - self.stop_loss_pct)
            tp = entry_price * (1 + self.take_profit_pct)
        else:  # short
            sl = entry_price * (1 + self.stop_loss_pct)
            tp = entry_price * (1 - self.take_profit_pct)
        return sl, tp