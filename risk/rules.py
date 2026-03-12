# risk/rules.py

from loguru import logger


class PortfolioRules:
    """
    Portfolio-level risk controls.
    Watches the WHOLE account, not individual trades.

    Three rules:
    1. Daily loss limit   → stop trading if down X% today
    2. Max drawdown       → shut down if down X% from all-time peak
    3. Max open trades    → never hold more than N positions

    First principles:
    These rules exist because markets can go haywire.
    A flash crash, earnings surprise, or macro shock
    can trigger many losses in a row very quickly.
    Portfolio rules are the circuit breaker that
    prevents a bad hour from becoming a blown account.
    """

    def __init__(self, config: dict):
        risk_cfg = config.get("risk", {})

        self.max_daily_loss_pct  = risk_cfg.get("max_daily_loss_pct",  0.03)
        self.max_drawdown_pct    = risk_cfg.get("max_drawdown_pct",     0.15)
        self.max_open_trades     = risk_cfg.get("max_open_trades",      1)

        # State — reset daily
        self.peak_value          = None   # all-time high portfolio value
        self.day_start_value     = None   # value at start of today
        self.open_trades         = 0      # current open positions
        self.trading_halted      = False  # circuit breaker
        self.halt_reason         = None

        logger.info(
            f"🏦 PortfolioRules ready | "
            f"Daily limit: -{self.max_daily_loss_pct*100:.0f}% | "
            f"Max DD: -{self.max_drawdown_pct*100:.0f}% | "
            f"Max trades: {self.max_open_trades}"
        )

    def start_day(self, portfolio_value: float):
        """
        Call at the start of each trading day.
        Records today's starting value for daily loss calc.
        """
        if self.peak_value is None:
            self.peak_value = portfolio_value

        self.day_start_value = portfolio_value
        self.peak_value      = max(self.peak_value, portfolio_value)

        # Reset halt at start of new day ONLY for daily limit
        # (drawdown halt persists until manually reset)
        if self.trading_halted and self.halt_reason == "daily_loss":
            logger.info("🔓 Daily loss limit reset for new day")
            self.trading_halted = False
            self.halt_reason    = None

    def check(self, portfolio_value: float) -> dict:
        """
        Check if any portfolio-level rule is violated.

        Returns dict:
        - allowed:      bool  (can we trade right now?)
        - halt_reason:  str | None
        - daily_pnl:    float (today's P&L %)
        - drawdown:     float (drawdown from peak %)
        - open_trades:  int
        """
        if self.day_start_value is None:
            self.start_day(portfolio_value)

        # Update peak
        self.peak_value = max(self.peak_value, portfolio_value)

        # Calculate metrics
        daily_pnl  = (
            (portfolio_value - self.day_start_value)
            / self.day_start_value * 100
        )
        drawdown   = (
            (self.peak_value - portfolio_value)
            / self.peak_value * 100
        )

        # Check daily loss limit
        if daily_pnl <= -(self.max_daily_loss_pct * 100):
            self._halt("daily_loss", daily_pnl, drawdown)

        # Check max drawdown (permanent until manual reset)
        if drawdown >= (self.max_drawdown_pct * 100):
            self._halt("max_drawdown", daily_pnl, drawdown)

        return {
            "allowed":     not self.trading_halted,
            "halt_reason": self.halt_reason,
            "daily_pnl":   round(daily_pnl,  2),
            "drawdown":    round(drawdown,    2),
            "open_trades": self.open_trades,
        }

    def can_open_trade(self, portfolio_value: float) -> dict:
        status = self.check(portfolio_value)

        if not status["allowed"]:
            return {
                **status,
                "allowed": False,
                "reason":  f"Trading halted: {status['halt_reason']}",
            }

        if self.open_trades >= self.max_open_trades:
            return {
                **status,
                "allowed": False,
                "reason": (
                    f"Max open trades reached "
                f"({self.open_trades}/{self.max_open_trades})"
            ),
        }

        return {"allowed": True, "reason": None, **status}
    def register_trade_open(self):
        """Call when a new position is opened."""
        self.open_trades += 1
        logger.debug(
            f"   📈 Trade opened | Open positions: {self.open_trades}"
        )

    def register_trade_close(self):
        """Call when a position is closed."""
        self.open_trades = max(0, self.open_trades - 1)
        logger.debug(
            f"   📉 Trade closed | Open positions: {self.open_trades}"
        )

    def reset_drawdown_halt(self):
        """
        Manually reset a drawdown halt.
        In production: requires human approval.
        """
        if self.halt_reason == "max_drawdown":
            logger.warning("⚠️  Drawdown halt manually reset by operator")
            self.trading_halted = False
            self.halt_reason    = None

    def _halt(self, reason: str, daily_pnl: float, drawdown: float):
        """Engage the circuit breaker."""
        if not self.trading_halted:
            self.trading_halted = True
            self.halt_reason    = reason
            logger.warning(
                f"🚨 TRADING HALTED — {reason} | "
                f"Daily P&L: {daily_pnl:+.2f}% | "
                f"Drawdown: {drawdown:.2f}%"
            )