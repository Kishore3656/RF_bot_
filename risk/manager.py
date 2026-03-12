# risk/manager.py

import numpy as np
from loguru import logger
from .position_sizer import PositionSizer
from .stop_loss      import StopLossManager
from .rules          import PortfolioRules


class RiskManager:
    """
    The single entry point for all risk decisions.

    Combines:
    - PortfolioRules   → "are we allowed to trade at all?"
    - PositionSizer    → "how much capital to use?"
    - StopLossManager  → "where are our exit levels?"

    First principles:
    The AI is the brain — it decides WHAT to do.
    The RiskManager is the adult supervision — it decides
    HOW MUCH to do it and WHEN to cut losses.

    Flow:
    1. Check portfolio rules (can we trade?)
    2. Calculate position size (how much?)
    3. Set stop loss / take profit levels (exits?)
    4. Return approved order or rejection
    """

    def __init__(self, config: dict):
        self.config  = config
        self.sizer   = PositionSizer(config)
        self.sl      = StopLossManager(config)
        self.rules   = PortfolioRules(config)

        # Track if we have an open position
        self.in_position  = False
        self.position_dir = None   # "long" or "short"

        logger.info("🛡️ RiskManager ready")
        logger.info(
            f"   Sizing:   {config['risk']['position_sizing']}"
        )
        logger.info(
            f"   Stop:     {config['risk'].get('stop_loss_pct', 0.05)*100:.0f}%"
        )
        logger.info(
            f"   Target:   {config['risk'].get('take_profit_pct', 0.15)*100:.0f}%"
        )

    def start_day(self, portfolio_value: float):
        """Call at market open each day."""
        self.rules.start_day(portfolio_value)

    def evaluate(
        self,
        ai_action:       int,
        current_price:   float,
        portfolio_value: float,
        win_rate:        float = 0.5,
        avg_win:         float = 1.0,
        avg_loss:        float = 1.0,
        volatility:      float = 0.02,
    ) -> dict:
        """
        Main method — takes an AI action and returns
        an approved order or rejection.

        ai_action: 0=HOLD, 1=LONG, 2=SHORT, 3=CLOSE

        Returns dict with:
        - approved:     bool
        - action:       int (may differ from ai_action)
        - reason:       str
        - position_size: dict (from PositionSizer)
        - stop_price:   float
        - target_price: float
        - override:     bool (risk changed the AI action)
        """

        # ── Step 1: Check stop loss on existing position ──
        if self.in_position:
            sl_check = self.sl.check(current_price)
            if sl_check["should_exit"]:
                logger.warning(
                    f"🛑 Risk override: force CLOSE "
                    f"({sl_check['reason']} | "
                    f"P&L: {sl_check['pnl_pct']:+.2f}%)"
                )
                self._close_position()
                return {
                    "approved":      True,
                    "action":        3,   # CLOSE
                    "action_name":   "CLOSE (risk override)",
                    "reason":        sl_check["reason"],
                    "pnl_pct":       sl_check["pnl_pct"],
                    "position_size": None,
                    "stop_price":    None,
                    "target_price":  None,
                    "override":      True,
                }

        # ── Step 2: HOLD — nothing to do ──────────────────
        if ai_action == 0:
            return self._no_trade("HOLD signal")

        # ── Step 3: CLOSE — just close if we have position ─
        if ai_action == 3:
            if not self.in_position:
                return self._no_trade("CLOSE but no position open")
            self._close_position()
            return {
                "approved":      True,
                "action":        3,
                "action_name":   "CLOSE",
                "reason":        "AI requested close",
                "pnl_pct":       self.sl.check(current_price)["pnl_pct"]
                                 if self.sl.entry_price else 0.0,
                "position_size": None,
                "stop_price":    None,
                "target_price":  None,
                "override":      False,
            }

        # ── Step 4: LONG or SHORT — check if we can open ──
        if ai_action in (1, 2):

            # Already in a position?
            if self.in_position:
                return self._no_trade(
                    f"Already in {self.position_dir} position"
                )

            # Check portfolio rules
            port_check = self.rules.can_open_trade(portfolio_value)
            if not port_check["allowed"]:
                return self._no_trade(port_check["reason"])

            # Calculate position size
            direction = "long" if ai_action == 1 else "short"
            size = self.sizer.calculate(
                portfolio_value   = portfolio_value,
                current_price     = current_price,
                win_rate          = win_rate,
                avg_win           = avg_win,
                avg_loss          = avg_loss,
                recent_volatility = volatility,
            )

            # Set stop loss levels
            self.sl.open_trade(current_price, direction)
            sl_price, tp_price = self.sl._get_levels(
                current_price, direction
            )

            # Register with portfolio rules
            self.rules.register_trade_open()
            self.in_position  = True
            self.position_dir = direction

            logger.info(
                f"✅ Order approved | {direction.upper()} | "
                f"${size['dollar_size']:,.0f} | "
                f"SL: ${sl_price:.2f} | TP: ${tp_price:.2f}"
            )

            return {
                "approved":      True,
                "action":        ai_action,
                "action_name":   direction.upper(),
                "reason":        "approved",
                "pnl_pct":       0.0,
                "position_size": size,
                "stop_price":    round(sl_price,  2),
                "target_price":  round(tp_price,  2),
                "override":      False,
            }

        return self._no_trade(f"Unknown action: {ai_action}")

    def _close_position(self):
        """Internal: clean up position state."""
        self.in_position  = False
        self.position_dir = None
        self.sl.close_trade()
        self.rules.register_trade_close()

    def _no_trade(self, reason: str) -> dict:
        """Return a rejection."""
        return {
            "approved":      False,
            "action":        0,
            "action_name":   "NO TRADE",
            "reason":        reason,
            "pnl_pct":       0.0,
            "position_size": None,
            "stop_price":    None,
            "target_price":  None,
            "override":      False,
        }