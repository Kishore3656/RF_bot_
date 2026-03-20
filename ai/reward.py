# ai/reward.py

import numpy as np
from collections import deque
from loguru import logger


class RewardFunction:
    """
    Reward function for trading.

    First principle: the bot does EXACTLY what
    you reward. Design every component intentionally.

    Components:
    1. Portfolio change   → reward making money
    2. Drawdown penalty   → punish losing it
    3. Cost penalty       → punish over-trading
    4. Sharpe bonus       → reward consistency
    5. Win/loss bonus     → directly reward winning trades  ← NEW
    6. Hold penalty       → stop the bot sitting idle       ← NEW
    """

    def __init__(self, config: dict):
        self.portfolio_weight  = 0.8    # main signal
        self.drawdown_weight   = 0.6    # drawdown penalty
        self.cost_weight       = 0.40   # higher cost penalty: strongly discourages overtrading
        self.sharpe_weight     = 0.4    # consistency bonus
        self.win_weight        = 0.8    # win/loss trade bonus
        self.hold_penalty      = 0.001  # gentler nudge — don't rush into bad trades
        self.initial_capital   = config["broker"]["initial_capital"]

        # Rolling window of recent returns for Sharpe calc
        self.return_window     = 20
        self.recent_returns    = deque(maxlen=self.return_window)

        # Track peak for drawdown
        self.peak_value        = self.initial_capital

        # trade tracking
        self.entry_price       = None     # price when trade opened
        self.entry_value       = None     # portfolio value when trade opened
        self.in_trade          = False    # are we currently in a position?
        self.hold_steps        = 0        # how many steps we've been holding
        self.max_hold_steps    = 25       # longer grace period before hold penalty fires
        self.consecutive_holds = 0        # how many HOLDs in a row

        logger.info("🏆 RewardFunction ready")
        logger.info(f"   Portfolio weight:  {self.portfolio_weight}")
        logger.info(f"   Drawdown weight:   {self.drawdown_weight}")
        logger.info(f"   Cost weight:       {self.cost_weight}")
        logger.info(f"   Sharpe weight:     {self.sharpe_weight}")
        logger.info(f"   Win bonus weight:  {self.win_weight}")
        logger.info(f"   Hold penalty:      {self.hold_penalty}")

    def reset(self):
        """Clear history at the start of each episode."""
        self.recent_returns    = deque(maxlen=self.return_window)
        self.peak_value        = self.initial_capital
        self.entry_price       = None
        self.entry_value       = None
        self.in_trade          = False
        self.hold_steps        = 0
        self.consecutive_holds = 0

    def calculate(
        self,
        prev_value:   float,
        curr_value:   float,
        trade_cost:   float,
        trade_taken:  bool,
        action:       int   = 0,    # ← NEW: 0=Hold, 1=Long, 2=Short, 3=Close
        current_price: float = 0.0, # ← NEW: needed to track trade entry
    ) -> tuple:
        """
        Calculate the full reward for one step.

        prev_value:    portfolio value last step
        curr_value:    portfolio value this step
        trade_cost:    dollar cost of any trade this step
        trade_taken:   did we actually trade this step?
        action:        what action was taken (0-3)
        current_price: current market price
        """

        # ── Component 1: Portfolio Change ─────────
        pct_change       = (curr_value - prev_value) / self.initial_capital
        reward_portfolio = self.portfolio_weight * pct_change

        # ── Component 2: Drawdown Penalty ─────────
        if curr_value > self.peak_value:
            self.peak_value = curr_value

        drawdown = (
            (self.peak_value - curr_value) / self.peak_value
            if self.peak_value > 0 else 0.0
        )
        reward_drawdown = -self.drawdown_weight * (drawdown ** 2)

        # ── Component 3: Transaction Cost Penalty ─
        cost_fraction = trade_cost / self.initial_capital
        reward_cost   = -self.cost_weight * cost_fraction

        # ── Component 4: Sharpe Bonus ─────────────
        self.recent_returns.append(pct_change)
        reward_sharpe = 0.0

        if len(self.recent_returns) >= 5:
            returns_arr = np.array(self.recent_returns)
            mean_ret    = np.mean(returns_arr)
            std_ret     = np.std(returns_arr)

            if std_ret > 1e-8:
                sharpe = mean_ret / std_ret
                if sharpe > 0:
                    reward_sharpe = self.sharpe_weight * sharpe

        # ── Component 5: Win/Loss Trade Bonus ─────
        # NEW: directly reward when a trade closes profitably
        # This teaches the bot that WINNING matters, not
        # just holding a growing portfolio
        reward_trade = 0.0

        # Opening a trade — record entry state
        if action in (1, 2) and not self.in_trade and current_price > 0:
            self.in_trade   = True
            self.entry_price = current_price
            self.entry_value = curr_value
            self.hold_steps  = 0

        # Closing a trade — score the outcome
        elif action == 3 and self.in_trade and self.entry_value is not None:
            trade_pnl = (curr_value - self.entry_value) / self.initial_capital

            if trade_pnl > 0:
                # Won — bonus proportional to gain (reduced multiplier)
                reward_trade = self.win_weight * trade_pnl * 3
            else:
                # Lost — penalty proportional to loss
                # Asymmetric: wins rewarded 2x more than losses punished
                reward_trade = self.win_weight * trade_pnl * 1.5

            self.in_trade   = False
            self.entry_price = None
            self.entry_value = None
            self.hold_steps  = 0

        # Track hold steps while in a trade
        elif action == 0 and self.in_trade:
            self.hold_steps += 1

        # ── Component 6: Hold Penalty ─────────────
        # NEW: penalize excessive HOLDs when NOT in a trade
        # The bot was learning to just sit and do nothing
        # This nudges it to make decisions
        reward_hold = 0.0

        if action == 0 and not self.in_trade:
            self.consecutive_holds += 1
            # Only penalize after a streak of holds
            if self.consecutive_holds > self.max_hold_steps:
                reward_hold = -self.hold_penalty
        else:
            self.consecutive_holds = 0

        # ── Final Reward ───────────────────────────
        reward = (
            reward_portfolio
            + reward_drawdown
            + reward_cost
            + reward_sharpe
            + reward_trade
            + reward_hold
        )

        # Scale down for stable neural net learning
        reward = reward * 1e-3

        breakdown = {
            "reward_total":     round(reward,            8),
            "reward_portfolio": round(reward_portfolio,  6),
            "reward_drawdown":  round(reward_drawdown,   6),
            "reward_cost":      round(reward_cost,       6),
            "reward_sharpe":    round(reward_sharpe,     6),
            "reward_trade":     round(reward_trade,      6),
            "reward_hold":      round(reward_hold,       6),
            "drawdown_pct":     round(drawdown * 100,    2),
            "portfolio_change": round(pct_change * 100,  4),
            "in_trade":         self.in_trade,
            "consecutive_holds": self.consecutive_holds,
        }

        return reward, breakdown