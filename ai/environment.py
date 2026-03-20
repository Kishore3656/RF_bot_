# ai/environment.py

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from loguru import logger
from .reward import RewardFunction


class TradingEnvironment(gym.Env):
    """
    A Gymnasium-compatible trading environment.
    Uses RAW prices for P&L, NORMALIZED prices for observations.

    Actions:
        0 = HOLD    → do nothing
        1 = BUY     → enter LONG  (profit when price goes UP)
        2 = SELL    → enter SHORT (profit when price goes DOWN)
        3 = CLOSE   → exit ANY open position (long or short)

    Why 4 actions instead of 3?
    With only buy/sell/hold, the bot had no way to exit a trade
    without accidentally opening a new one in the opposite direction.
    CLOSE gives it a clean "exit whatever I'm in" action.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df:               pd.DataFrame,
        config:           dict,
        raw_prices:       pd.Series = None,
        initial_capital:  float = 10_000.0,
        transaction_cost: float = 0.001,
        reward_scaling:   float = 1e-4,
        max_position:     float = 1.0,
    ):
        super().__init__()

        # ── Normalized data (bot's eyes) ───────────
        self.df           = df.reset_index(drop=True)
        self.n_steps      = len(df)
        self.feature_cols = [
            c for c in df.columns
            if c not in ["symbol", "asset_type"]
        ]
        self.n_features   = len(self.feature_cols)
        # Precompute as numpy array to avoid per-step pandas overhead
        self._feature_array = self.df[self.feature_cols].values.astype(np.float32)

        # ── Raw prices (bot's wallet) ──────────────
        if raw_prices is not None:
            self.raw_prices = raw_prices.reset_index(drop=True).values
        else:
            norm_close = df["close"].values
            self.raw_prices = np.full(len(df), 100.0)
            for i in range(1, len(df)):
                change = norm_close[i] - norm_close[i - 1]
                self.raw_prices[i] = max(
                    self.raw_prices[i - 1] * (1 + change), 0.01
                )

        # ── Settings ───────────────────────────────
        self.initial_capital  = initial_capital
        self.transaction_cost = transaction_cost
        self.reward_scaling   = reward_scaling
        self.max_position     = max_position

        # ── Action space: 4 actions ────────────────
        # 0=Hold, 1=Long, 2=Short, 3=Close
        self.action_space = spaces.Discrete(4)

        # ── Observation space: features + portfolio ─
        # +1 extra portfolio value: position_direction
        # tells bot whether it's long, short, or flat
        obs_size = self.n_features + 6
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )

        # ── Internal state ─────────────────────────
        self.current_step             = 0
        self.cash                     = initial_capital
        self.position                 = 0.0   # units held
                                               # positive = long
                                               # negative = short
        self.entry_price              = 0.0
        self.portfolio_value          = initial_capital
        self.prev_portfolio_value     = initial_capital
        self.trades                   = []
        self.portfolio_history        = []
        self.last_reward_breakdown    = {}

        # ── Reward function ────────────────────────
        self.reward_fn = RewardFunction(config)

        logger.info(
            f"🎮 TradingEnvironment created | "
            f"{self.n_steps} steps | "
            f"{self.n_features} features | "
            f"Capital: ${initial_capital:,.0f} | "
            f"Price range: ${self.raw_prices.min():.2f}"
            f"–${self.raw_prices.max():.2f} | "
            f"Actions: Hold/Long/Short/Close"
        )

    # ─────────────────────────────────────────────
    # CORE GYMNASIUM METHODS
    # ─────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """Start a fresh episode."""
        super().reset(seed=seed)

        self.current_step             = 0
        self.cash                     = self.initial_capital
        self.position                 = 0.0
        self.entry_price              = 0.0
        self.portfolio_value          = self.initial_capital
        self.prev_portfolio_value     = self.initial_capital
        self.trades                   = []
        self.portfolio_history        = [self.initial_capital]
        self.last_reward_breakdown    = {}
        self.reward_fn.reset()

        return self._get_observation(), {}

    def step(self, action):
        """Agent takes one action. World reacts."""
        if self.current_step >= self.n_steps - 1:
            return self._get_observation(), 0.0, True, False, {}

        current_price = self._get_current_price()
        trade_info    = self._execute_action(action, current_price)

        self.current_step += 1
        next_price = self._get_current_price()

        self.portfolio_value = self._calculate_portfolio_value(next_price)
        self.portfolio_history.append(self.portfolio_value)

        reward     = self._calculate_reward(trade_info, action, current_price)
        terminated = self.current_step >= self.n_steps - 1
        truncated  = False

        info = {
            "step":            self.current_step,
            "price":           next_price,
            "cash":            self.cash,
            "position":        self.position,
            "portfolio_value": self.portfolio_value,
            "action":          action,
            **trade_info
        }

        return self._get_observation(), reward, terminated, truncated, info

    def render(self, mode="human"):
        """Print current state for debugging."""
        price       = self._get_current_price()
        pnl_pct     = (
            (self.portfolio_value - self.initial_capital)
            / self.initial_capital * 100
        )
        direction   = "LONG " if self.position > 0 else \
                      "SHORT" if self.position < 0 else "FLAT "
        print(
            f"Step {self.current_step:4d} | "
            f"Price: ${price:8.2f} | "
            f"Cash: ${self.cash:9.2f} | "
            f"{direction}: {abs(self.position):8.4f} | "
            f"Portfolio: ${self.portfolio_value:9.2f} | "
            f"P&L: {pnl_pct:+.2f}%"
        )

    # ─────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────

    def _get_current_price(self) -> float:
        """Real dollar price — used for all money calculations."""
        return float(self.raw_prices[self.current_step])

    def _get_observation(self) -> np.ndarray:
        """Normalized observation for the neural network."""
        market_state    = self._feature_array[self.current_step]

        portfolio_state = self._get_portfolio_state()

        return np.concatenate(
            [market_state, portfolio_state]
        ).astype(np.float32)

    def _get_portfolio_state(self) -> np.ndarray:
        """
        Bot's self-awareness: 6 normalized portfolio values.

        New vs before: added position_direction
        0.0 = short, 0.5 = flat, 1.0 = long
        Bot needs to know WHICH direction it's in,
        not just whether it has a position.
        """
        price = self._get_current_price()

        cash_ratio = min(self.cash / self.initial_capital, 1.0)

        position_value = abs(self.position) * price
        position_ratio = min(
            position_value / self.initial_capital, 1.0
        )

        if self.position != 0 and self.entry_price > 0:
            if self.position > 0:  # LONG
                unrealized_pct = (price - self.entry_price) / self.entry_price
            else:                  # SHORT
                unrealized_pct = (self.entry_price - price) / self.entry_price
            unrealized_norm = np.clip((unrealized_pct + 0.5), 0, 1)
        else:
            unrealized_norm = 0.5   # neutral

        portfolio_ratio = min(
            self.portfolio_value / (self.initial_capital * 2), 1.0
        )

        # Direction: 0.0=short, 0.5=flat, 1.0=long
        if self.position > 0:
            position_direction = 1.0
        elif self.position < 0:
            position_direction = 0.0
        else:
            position_direction = 0.5

        has_position = 1.0 if self.position != 0 else 0.0

        return np.array([
            cash_ratio,
            position_ratio,
            unrealized_norm,
            portfolio_ratio,
            position_direction,
            has_position,
        ], dtype=np.float32)

    def _execute_action(self, action: int, current_price: float) -> dict:
        """
        Execute one of 4 actions using real prices.

        0 = HOLD  → do nothing
        1 = LONG  → buy units, profit if price rises
        2 = SHORT → borrow and sell units, profit if price falls
        3 = CLOSE → exit whatever position is open
        """
        trade_info = {
            "action_taken": "hold",
            "trade_price":  current_price,
            "trade_size":   0.0,
            "trade_cost":   0.0,
            "realized_pnl": 0.0,
            "direction":    "flat",
        }

        # ── Action 0: HOLD ─────────────────────────
        if action == 0:
            pass

        # ── Action 1: GO LONG ──────────────────────
        elif action == 1:
            # Only enter if currently flat (no open position)
            if self.position == 0 and self.cash > 1.0:
                buy_value = self.cash * self.max_position
                cost      = buy_value * self.transaction_cost
                units     = (buy_value - cost) / current_price

                if units > 0:
                    self.position    = units        # positive = long
                    self.entry_price = current_price
                    self.cash       -= buy_value

                    trade_info.update({
                        "action_taken": "long",
                        "trade_size":   units,
                        "trade_cost":   cost,
                        "direction":    "long",
                    })

        # ── Action 2: GO SHORT ─────────────────────
        elif action == 2:
            # Only enter if currently flat (no open position)
            if self.position == 0 and self.cash > 1.0:
                # Short selling: borrow and sell units now,
                # buy them back later at (hopefully) lower price.
                # Cash does NOT change — proceeds held as collateral.
                short_value = self.cash * self.max_position
                cost        = short_value * self.transaction_cost
                units       = (short_value - cost) / current_price

                if units > 0:
                    self.position    = -units       # negative = short
                    self.entry_price = current_price
                    # Cash stays the same — no cash changes on short entry

                    trade_info.update({
                        "action_taken": "short",
                        "trade_size":   units,
                        "trade_cost":   cost,
                        "direction":    "short",
                    })
        # ── Action 3: CLOSE POSITION ───────────────
        elif action == 3:
            if self.position != 0:

                if self.position > 0:
                    # ── Close LONG: sell units ─────
                    sale_value   = self.position * current_price
                    cost         = sale_value * self.transaction_cost
                    realized_pnl = (
                        sale_value
                        - (self.position * self.entry_price)
                        - cost
                    )
                    self.cash += (sale_value - cost)
                    direction  = "close_long"

                else:
                    # ── Close SHORT: buy units back ─
                    # We shorted at entry_price, now buying back
                    # at current_price
                    units_to_buy  = abs(self.position)
                    buyback_value = units_to_buy * current_price
                    cost          = buyback_value * self.transaction_cost

                    # Profit = what we sold for - what we buy back for
                    realized_pnl = (
                        (units_to_buy * self.entry_price)
                        - buyback_value
                        - cost
                    )
                    # Add profit (or subtract loss) from cash
                    self.cash += realized_pnl
                    direction     = "close_short"

                trade_info.update({
                    "action_taken": "close",
                    "trade_size":   abs(self.position),
                    "trade_cost":   cost,
                    "realized_pnl": realized_pnl,
                    "direction":    direction,
                })

                self.trades.append({
                    "entry_price": self.entry_price,
                    "exit_price":  current_price,
                    "pnl":         realized_pnl,
                    "direction":   direction,
                    "step":        self.current_step,
                })

                # Clear position
                self.position    = 0.0
                self.entry_price = 0.0

        return trade_info

    def _calculate_portfolio_value(self, current_price: float) -> float:
        """
        Total account value = cash + unrealized P&L on any open position.

        Long:  value goes up when price rises
        Short: value goes up when price falls
        """
        if self.position > 0:
            # Long: cash is already deducted, position has market value
            position_value = self.position * current_price
            return self.cash + position_value

        elif self.position < 0:
            # Short P&L = profit if price fell since entry
            units_owed = abs(self.position)
            short_pnl  = (self.entry_price - current_price) * units_owed
            return self.cash + short_pnl

        else:
            return self.cash

    def _calculate_reward(self, trade_info: dict, action: int, current_price: float) -> float:
        """Use our carefully designed RewardFunction."""
        reward, breakdown = self.reward_fn.calculate(
            prev_value    = self.prev_portfolio_value,
            curr_value    = self.portfolio_value,
            trade_cost    = trade_info["trade_cost"],
            trade_taken   = trade_info["action_taken"] != "hold",
            action        = action,
            current_price = current_price,
        )

        self.prev_portfolio_value  = self.portfolio_value
        self.last_reward_breakdown = breakdown

        return float(reward)

    # ─────────────────────────────────────────────
    # UTILITY
    # ─────────────────────────────────────────────

    def get_performance_summary(self) -> dict:
        """Summary of how the episode went."""
        if len(self.portfolio_history) < 2:
            return {}

        final_value  = self.portfolio_value
        total_return = (
            (final_value - self.initial_capital)
            / self.initial_capital * 100
        )

        peak   = self.initial_capital
        max_dd = 0.0
        for val in self.portfolio_history:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd

        # Count ALL trades including any open position at end
        all_trades = self.trades.copy()

        if self.position != 0 and self.entry_price > 0:
            final_price = self._get_current_price()
            if self.position > 0:
                open_pnl = (final_price - self.entry_price) * self.position
                direction = "open_long"
            else:
                open_pnl  = (self.entry_price - final_price) * abs(self.position)
                direction = "open_short"

            all_trades.append({
                "entry_price": self.entry_price,
                "exit_price":  final_price,
                "pnl":         open_pnl,
                "direction":   direction,
                "open":        True,
            })

        if all_trades:
            wins     = sum(1 for t in all_trades if t["pnl"] > 0)
            win_rate = wins / len(all_trades) * 100
            long_trades  = [t for t in all_trades if "long"  in t.get("direction", "")]
            short_trades = [t for t in all_trades if "short" in t.get("direction", "")]
        else:
            win_rate     = 0.0
            long_trades  = []
            short_trades = []

        return {
            "initial_capital": self.initial_capital,
            "final_value":     round(final_value, 2),
            "total_return":    round(total_return, 2),
            "max_drawdown":    round(max_dd * 100, 2),
            "total_trades":    len(all_trades),
            "long_trades":     len(long_trades),
            "short_trades":    len(short_trades),
            "open_position":   self.position != 0,
            "win_rate":        round(win_rate, 2),
        }