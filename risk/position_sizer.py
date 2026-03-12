# risk/position_sizer.py

import numpy as np
from loguru import logger


class PositionSizer:
    """
    Determines HOW MUCH capital to risk on each trade.

    First principles: even a perfect signal is useless
    if you size positions incorrectly.
    - Too large → one loss wipes you out
    - Too small → gains are negligible
    - Just right → consistent compounding

    Supports 4 sizing methods:
    1. fixed_pct    → always use X% of capital
    2. kelly        → mathematically optimal sizing
    3. volatility   → size inversely to recent volatility
    4. fixed_dollar → always risk $X per trade
    """

    METHODS = ["fixed_pct", "kelly", "volatility", "fixed_dollar"]

    def __init__(self, config: dict):
        risk_cfg = config.get("risk", {})

        self.method       = risk_cfg.get("position_sizing", "fixed_pct")
        self.fixed_pct    = risk_cfg.get("max_position_pct", 0.95)
        self.kelly_fraction = risk_cfg.get("kelly_fraction", 0.25)
        self.fixed_dollar = risk_cfg.get("fixed_dollar_risk", 200)
        self.min_pct      = risk_cfg.get("min_position_pct", 0.01)
        self.max_pct      = risk_cfg.get("max_position_pct", 0.95)

        if self.method not in self.METHODS:
            logger.warning(
                f"⚠️ Unknown sizing method '{self.method}', "
                f"defaulting to fixed_pct"
            )
            self.method = "fixed_pct"

        logger.info(f"📐 PositionSizer ready | Method: {self.method}")

    def calculate(
        self,
        portfolio_value:  float,
        current_price:    float,
        win_rate:         float = 0.5,
        avg_win:          float = 1.0,
        avg_loss:         float = 1.0,
        recent_volatility: float = 0.02,
    ) -> dict:
        """
        Calculate position size for the next trade.

        Returns dict with:
        - fraction:    % of portfolio to use (0.0 to 1.0)
        - dollar_size: dollar amount to invest
        - units:       number of shares/units to buy
        - method:      which method was used
        - reasoning:   human-readable explanation
        """

        if self.method == "fixed_pct":
            fraction  = self.fixed_pct
            reasoning = f"Fixed {fraction*100:.0f}% of portfolio"

        elif self.method == "kelly":
            fraction, reasoning = self._kelly_size(
                win_rate, avg_win, avg_loss
            )

        elif self.method == "volatility":
            fraction, reasoning = self._volatility_size(
                recent_volatility
            )

        elif self.method == "fixed_dollar":
            fraction  = min(
                self.fixed_dollar / portfolio_value,
                self.max_pct
            )
            reasoning = f"Fixed ${self.fixed_dollar} risk"

        # Clamp to [min_pct, max_pct]
        fraction = np.clip(fraction, self.min_pct, self.max_pct)

        dollar_size = portfolio_value * fraction
        units       = dollar_size / current_price if current_price > 0 else 0

        return {
            "fraction":    round(fraction, 4),
            "dollar_size": round(dollar_size, 2),
            "units":       round(units, 6),
            "method":      self.method,
            "reasoning":   reasoning,
        }

    def _kelly_size(
        self,
        win_rate: float,
        avg_win:  float,
        avg_loss: float,
    ) -> tuple:
        """
        Kelly Criterion position sizing.

        f* = (win_rate / avg_loss) - (loss_rate / avg_win)

        First principles:
        This is the mathematically OPTIMAL fraction that
        maximizes long-term geometric growth rate.

        We use half-Kelly (× kelly_fraction) because:
        - Kelly assumes perfect knowledge of win_rate
        - Real win_rate estimates are noisy
        - Half-Kelly gives ~75% of Kelly growth with
          much lower variance and risk of ruin
        """
        loss_rate = 1.0 - win_rate

        if avg_win <= 0 or avg_loss <= 0:
            return self.fixed_pct, "Kelly failed (no trade history), using fixed"

        kelly_full = (win_rate / avg_loss) - (loss_rate / avg_win)

        if kelly_full <= 0:
            # Negative Kelly = don't trade (negative edge)
            return self.min_pct, f"Kelly={kelly_full:.3f} (negative edge!)"

        # Apply fraction (default 0.25 = quarter-Kelly for safety)
        kelly_sized = kelly_full * self.kelly_fraction

        reasoning = (
            f"Kelly={kelly_full:.3f} × {self.kelly_fraction} "
            f"= {kelly_sized:.3f}"
        )
        return kelly_sized, reasoning

    def _volatility_size(self, recent_volatility: float) -> tuple:
        """
        Volatility-adjusted position sizing.

        First principles:
        Risk the same DOLLAR AMOUNT regardless of volatility.

        Low volatility asset  → price moves little → buy more units
        High volatility asset → price moves a lot  → buy fewer units

        Target: never lose more than target_risk% in one move.

        Example:
        Target risk = 2% of portfolio = $200
        Asset volatility = 1% daily

        Position = $200 / 1% = $20,000 worth of stock
        If stock drops 1% (its normal move), you lose $200 = 2% ✓

        If volatility doubles to 2%:
        Position = $200 / 2% = $10,000 worth (half as much)
        If stock drops 2%, you still lose $200 = 2% ✓
        """
        target_risk = 0.02  # target 2% portfolio risk per trade

        if recent_volatility <= 0:
            return self.fixed_pct, "Volatility=0, using fixed"

        fraction  = target_risk / recent_volatility
        reasoning = (
            f"VolSize: {target_risk:.0%} risk / "
            f"{recent_volatility:.2%} vol = {fraction:.2%}"
        )
        return fraction, reasoning