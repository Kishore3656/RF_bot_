# data/synthetic.py

import numpy as np
import pandas as pd
from loguru import logger
from datetime import datetime, timedelta


class SyntheticDataGenerator:
    """
    Generates realistic synthetic market data for RL training.
    
    Why? Real data is limited. Synthetic data lets us:
    - Train on thousands of scenarios
    - Simulate rare events (crashes, rallies)
    - Test risk management against extremes
    """

    # Four market regimes with their characteristics
    REGIMES = {
        "bull": {
            "drift":      0.0008,   # +0.08% per day (strong uptrend)
            "volatility": 0.012,    # 1.2% daily volatility (calm)
            "description": "Strong uptrend, low volatility"
        },
        "bear": {
            "drift":     -0.0006,   # -0.06% per day (downtrend)
            "volatility": 0.018,    # 1.8% daily volatility (nervous)
            "description": "Downtrend, rising volatility"
        },
        "sideways": {
            "drift":      0.0001,   # Nearly flat
            "volatility": 0.008,    # 0.8% daily volatility (quiet)
            "description": "Choppy, going nowhere"
        },
        "volatile": {
            "drift":      0.0002,   # Slight positive bias
            "volatility": 0.035,    # 3.5% daily volatility (wild!)
            "description": "Extreme volatility, crash/rally risk"
        }
    }

    def __init__(self, config: dict):
        self.config = config
        self.random_seed = 42   # Makes results reproducible
        np.random.seed(self.random_seed)
        logger.info("🎲 SyntheticDataGenerator ready")

    # ─────────────────────────────────────────
    # PUBLIC METHODS
    # ─────────────────────────────────────────

    def generate(
        self,
        regime:        str   = "bull",
        n_days:        int   = 365,
        start_price:   float = 100.0,
        symbol:        str   = "SYNTH"
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for one regime.

        regime:      "bull", "bear", "sideways", "volatile"
        n_days:      how many days to generate
        start_price: starting price
        symbol:      name to give the synthetic asset
        """
        if regime not in self.REGIMES:
            raise ValueError(
                f"❌ Unknown regime '{regime}'. "
                f"Choose from: {list(self.REGIMES.keys())}"
            )

        params = self.REGIMES[regime]
        logger.info(
            f"🎲 Generating {n_days} days of "
            f"'{regime}' market data | "
            f"{params['description']}"
        )

        # Step 1: Generate the closing prices
        closes = self._generate_price_series(
            start_price = start_price,
            n_days      = n_days,
            drift       = params["drift"],
            volatility  = params["volatility"]
        )

        # Step 2: Generate realistic OHLV around those closes
        df = self._generate_ohlcv(closes, params["volatility"])

        # Step 3: Add metadata
        df["symbol"]     = f"{symbol}_{regime.upper()}"
        df["asset_type"] = "synthetic"

        logger.success(
            f"✅ Generated {len(df)} rows | "
            f"Price: ${closes[0]:.2f} → ${closes[-1]:.2f} | "
            f"Change: {((closes[-1]/closes[0])-1)*100:+.1f}%"
        )
        return df

    def generate_all_regimes(
        self,
        n_days:      int   = 365,
        start_price: float = 100.0
    ) -> dict:
        """
        Generate data for ALL four regimes at once.
        Returns dict: {"bull": df, "bear": df, ...}
        """
        logger.info("🎲 Generating all four market regimes...")
        all_data = {}

        for regime in self.REGIMES:
            df = self.generate(
                regime      = regime,
                n_days      = n_days,
                start_price = start_price,
                symbol      = "SYNTH"
            )
            all_data[regime] = df

        logger.success(
            f"✅ All four regimes generated | "
            f"{n_days} days each"
        )
        return all_data

    def generate_mixed_regime(
        self,
        n_days:      int   = 1000,
        start_price: float = 100.0
    ) -> pd.DataFrame:
        """
        Generate a single long series that switches between
        regimes randomly — most realistic training data.

        Like real markets: calm period → crash → recovery → bull run
        """
        logger.info(
            f"🎲 Generating {n_days} days of "
            f"mixed-regime data..."
        )

        all_closes  = [start_price]
        regime_log  = []

        regimes     = list(self.REGIMES.keys())
        current_day = 0

        while current_day < n_days:
            # Pick a random regime
            regime = np.random.choice(regimes)
            params = self.REGIMES[regime]

            # Stay in this regime for 30-120 days
            duration = np.random.randint(30, 121)
            duration = min(duration, n_days - current_day)

            # Generate prices for this regime chunk
            chunk = self._generate_price_series(
                start_price = all_closes[-1],
                n_days      = duration,
                drift       = params["drift"],
                volatility  = params["volatility"]
            )

            # Add to the full series (skip first price,
            # it's already in all_closes)
            all_closes.extend(chunk[1:])
            regime_log.append({
                "regime":   regime,
                "start":    current_day,
                "end":      current_day + duration,
                "duration": duration
            })

            current_day += duration

        # Trim to exact length
        all_closes = all_closes[:n_days + 1]

        # Build the full OHLCV DataFrame
        avg_volatility = np.mean(
            [r["volatility"] for r in self.REGIMES.values()]
        )
        df = self._generate_ohlcv(
            np.array(all_closes),
            avg_volatility
        )
        df["symbol"]     = "SYNTH_MIXED"
        df["asset_type"] = "synthetic"

        # Log the regime breakdown
        logger.success(
            f"✅ Mixed regime: {len(df)} days generated"
        )
        for entry in regime_log:
            logger.debug(
                f"   Day {entry['start']:4d}-{entry['end']:4d}: "
                f"{entry['regime'].upper()} "
                f"({entry['duration']} days)"
            )

        return df

    # ─────────────────────────────────────────
    # PRIVATE METHODS
    # ─────────────────────────────────────────

    def _generate_price_series(
        self,
        start_price: float,
        n_days:      int,
        drift:       float,
        volatility:  float
    ) -> np.ndarray:
        """
        Generate a realistic price series using
        Geometric Brownian Motion (GBM).

        Each day's price = yesterday's price
                         × exp(drift + random_shock)
        """
        prices = np.zeros(n_days + 1)
        prices[0] = start_price

        for i in range(1, n_days + 1):
            # Random daily shock (normally distributed)
            random_shock = np.random.normal(0, volatility)

            # GBM formula
            prices[i] = prices[i-1] * np.exp(drift + random_shock)

        return prices

    def _generate_ohlcv(
        self,
        closes:     np.ndarray,
        volatility: float
    ) -> pd.DataFrame:
        """
        Generate realistic Open, High, Low, Volume
        around the closing prices.

        Real candles aren't just a close price —
        they have intraday movement.
        """
        n = len(closes) - 1  # number of complete candles
        records = []

        # Generate dates starting from today going backward
        end_date   = datetime.now()
        start_date = end_date - timedelta(days=n)

        for i in range(n):
            close = closes[i + 1]
            prev_close = closes[i]

            # Open: near previous close with small gap
            open_noise = np.random.normal(0, volatility * 0.3)
            open_price = prev_close * np.exp(open_noise)

            # Intraday range based on volatility
            # High and low expand around the open/close range
            intraday_range = abs(close - open_price)
            extra_high = abs(np.random.normal(
                0, volatility * close * 0.5
            ))
            extra_low  = abs(np.random.normal(
                0, volatility * close * 0.5
            ))

            high_price = max(open_price, close) + extra_high
            low_price  = min(open_price, close) - extra_low

            # Volume: random around a base with some correlation
            # to price movement (bigger moves = more volume)
            price_move = abs(close - prev_close) / prev_close
            volume_base   = 1_000_000
            volume_factor = 1 + (price_move / volatility)
            volume = int(
                volume_base
                * volume_factor
                * np.random.lognormal(0, 0.5)
            )

            # Build the date for this row
            date = start_date + timedelta(days=i)

            records.append({
                "open":   round(open_price, 4),
                "high":   round(high_price, 4),
                "low":    round(low_price,  4),
                "close":  round(close,      4),
                "volume": volume
            })

        # Build DataFrame with datetime index
        dates = pd.date_range(
            start=start_date,
            periods=n,
            freq="D"
        )
        df = pd.DataFrame(records, index=dates)
        df.index.name = "datetime"

        return df