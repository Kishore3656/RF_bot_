# data/indicators.py

import pandas as pd
import numpy as np
from loguru import logger


class IndicatorEngine:
    """
    Takes raw OHLCV data and adds all 23 technical indicators.
    Single responsibility: enrich data, nothing else.
    """

    def __init__(self, config: dict):
        """
        Read indicator settings from config.yaml
        """
        ind = config["indicators"]

        # Trend settings
        self.ema_periods   = ind["ema_periods"]       # [20, 50, 200]
        self.rsi_periods   = ind["rsi_periods"]       # [7, 14]

        # MACD settings
        self.macd_fast     = ind["macd"]["fast"]      # 12
        self.macd_slow     = ind["macd"]["slow"]      # 26
        self.macd_signal   = ind["macd"]["signal"]    # 9

        # Bollinger settings
        self.bb_period     = ind["bollinger"]["period"]   # 20
        self.bb_std        = ind["bollinger"]["std_dev"]  # 2

        # ATR setting
        self.atr_period    = ind["atr_period"]        # 14

        logger.info("🧮 IndicatorEngine ready")

    # ─────────────────────────────────────────
    # PUBLIC METHOD
    # ─────────────────────────────────────────

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Master function — runs ALL indicators on a DataFrame.
        Input:  raw OHLCV DataFrame
        Output: same DataFrame + 23 new indicator columns
        """
        symbol = df["symbol"].iloc[0] if "symbol" in df.columns else "unknown"
        logger.info(f"🧮 Computing indicators for {symbol}...")

        # Work on a copy — never modify original data
        df = df.copy()

        # ── Add each indicator group ──────────
        df = self._add_moving_averages(df)
        df = self._add_macd(df)
        df = self._add_rsi(df)
        df = self._add_bollinger_bands(df)
        df = self._add_atr(df)
        df = self._add_volume_indicators(df)
        df = self._add_momentum_indicators(df)

        # ── Drop rows where indicators are NaN ──
        # (first N rows are always NaN until
        #  enough history exists to calculate)
        before = len(df)
        df.dropna(inplace=True)
        after  = len(df)

        dropped = before - after
        logger.success(
            f"✅ {symbol}: {after} rows with full indicators "
            f"({dropped} warmup rows removed)"
        )

        return df

    def compute_batch(self, data_dict: dict) -> dict:
        """
        Run compute_all on a whole dictionary of assets.
        Input:  {"AAPL": df, "BTC-USD": df, ...}
        Output: {"AAPL": df_with_indicators, ...}
        """
        enriched = {}
        for symbol, df in data_dict.items():
            result = self.compute_all(df)
            if result is not None and len(result) > 0:
                enriched[symbol] = result

        logger.success(
            f"✅ Indicators computed for "
            f"{len(enriched)}/{len(data_dict)} assets"
        )
        return enriched

    # ─────────────────────────────────────────
    # PRIVATE METHODS — one per indicator group
    # ─────────────────────────────────────────

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add EMA for each period in config.
        EMA = Exponential Moving Average
        Answers: "Which direction is price trending?"
        """
        for period in self.ema_periods:
            col_name = f"ema_{period}"
            df[col_name] = (
                df["close"]
                .ewm(span=period, adjust=False)
                .mean()
            )

        # Also add a simple golden/death cross signal
        # 1 = EMA20 above EMA50 (bullish)
        # 0 = EMA20 below EMA50 (bearish)
        if 20 in self.ema_periods and 50 in self.ema_periods:
            df["ema_cross"] = (
                df["ema_20"] > df["ema_50"]
            ).astype(int)

        logger.debug(f"   ✓ Moving averages: {self.ema_periods}")
        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add MACD line, signal line, and histogram.
        Answers: "Is momentum shifting up or down?"
        """
        # Calculate the two EMAs
        ema_fast = (
            df["close"]
            .ewm(span=self.macd_fast, adjust=False)
            .mean()
        )
        ema_slow = (
            df["close"]
            .ewm(span=self.macd_slow, adjust=False)
            .mean()
        )

        # MACD line = difference between fast and slow
        df["macd_line"] = ema_fast - ema_slow

        # Signal line = EMA of the MACD line
        df["macd_signal"] = (
            df["macd_line"]
            .ewm(span=self.macd_signal, adjust=False)
            .mean()
        )

        # Histogram = gap between MACD and signal
        # Positive = bullish momentum building
        # Negative = bearish momentum building
        df["macd_histogram"] = df["macd_line"] - df["macd_signal"]

        logger.debug(
            f"   ✓ MACD ({self.macd_fast}"
            f"/{self.macd_slow}/{self.macd_signal})"
        )
        return df

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add RSI for each period in config.
        RSI = Relative Strength Index (0-100)
        Answers: "Are buyers or sellers currently winning?"
        """
        for period in self.rsi_periods:
            col_name = f"rsi_{period}"
            df[col_name] = self._calculate_rsi(df["close"], period)

        logger.debug(f"   ✓ RSI periods: {self.rsi_periods}")
        return df

    def _calculate_rsi(
        self,
        prices: pd.Series,
        period: int
    ) -> pd.Series:
        """
        RSI calculation from scratch so you understand it.

        Step 1: Find price changes each day
        Step 2: Separate gains from losses
        Step 3: Average the gains and losses
        Step 4: Convert to 0-100 scale
        """
        # Step 1: Daily price change
        delta = prices.diff()

        # Step 2: Separate gains and losses
        gains  = delta.clip(lower=0)   # keep positives, zero out negatives
        losses = -delta.clip(upper=0)  # keep negatives (as positive), zero rest

        # Step 3: Rolling average of gains and losses
        avg_gain = gains.ewm(span=period, adjust=False).mean()
        avg_loss = losses.ewm(span=period, adjust=False).mean()

        # Step 4: RSI formula
        # RS = ratio of avg gains to avg losses
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)

        # Convert to 0-100 scale
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Bollinger Bands: upper, middle, lower, and width.
        Answers: "Is price stretched too far in either direction?"
        """
        # Middle band = simple moving average
        middle = df["close"].rolling(window=self.bb_period).mean()

        # Standard deviation over same period
        std = df["close"].rolling(window=self.bb_period).std()

        # Upper and lower bands
        df["bb_upper"]  = middle + (self.bb_std * std)
        df["bb_middle"] = middle
        df["bb_lower"]  = middle - (self.bb_std * std)

        # Band width = how wide the bands are
        # Squeeze (narrow) = big move coming
        # Wide = currently volatile
        df["bb_width"] = (
            (df["bb_upper"] - df["bb_lower"])
            / df["bb_middle"]
        )

        # Position within bands (0 = at lower band, 1 = at upper band)
        # Very useful for the RL agent!
        df["bb_position"] = (
            (df["close"] - df["bb_lower"])
            / (df["bb_upper"] - df["bb_lower"])
        )

        logger.debug(
            f"   ✓ Bollinger Bands "
            f"({self.bb_period}, {self.bb_std}σ)"
        )
        return df

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ATR: Average True Range.
        Answers: "How much does this asset typically move each day?"
        Critical for: stop loss placement and position sizing.
        """
        # True Range = biggest of these three:
        # 1. High - Low (today's range)
        # 2. High - Previous Close (gap up)
        # 3. Previous Close - Low (gap down)
        high_low   = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close  = abs(df["low"]  - df["close"].shift(1))

        # True range = the maximum of all three
        true_range = pd.concat(
            [high_low, high_close, low_close],
            axis=1
        ).max(axis=1)

        # ATR = smoothed average of true range
        df["atr_14"] = (
            true_range
            .ewm(span=self.atr_period, adjust=False)
            .mean()
        )

        # Also add ATR as % of price
        # Useful for comparing across assets
        # (Bitcoin ATR of $2000 vs Apple ATR of $3
        #  both become ~2% — now comparable!)
        df["atr_pct"] = df["atr_14"] / df["close"] * 100

        logger.debug(f"   ✓ ATR ({self.atr_period})")
        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based indicators.
        Answers: "Is the move backed by real conviction?"
        """
        # Volume SMA — average volume over 20 days
        df["volume_sma_20"] = (
            df["volume"]
            .rolling(window=20)
            .mean()
        )

        # Volume ratio — is today's volume above or below average?
        # > 1.0 = above average volume (conviction)
        # < 1.0 = below average volume (weak move)
        df["volume_ratio"] = (
            df["volume"]
            / df["volume_sma_20"].replace(0, np.finfo(float).eps)
        )

        # OBV — On Balance Volume
        # Running total: add volume on up days, subtract on down days
        # Rising OBV with falling price = smart money buying quietly
        obv = []
        obv_value = 0

        for i in range(len(df)):
            if i == 0:
                obv.append(0)
                continue

            if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                obv_value += df["volume"].iloc[i]   # up day: add
            elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                obv_value -= df["volume"].iloc[i]   # down day: subtract
            # if close unchanged: obv stays same

            obv.append(obv_value)

        df["obv"] = obv

        logger.debug("   ✓ Volume indicators (SMA, ratio, OBV)")
        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum-based indicators:
        - Stochastic %K/%D  → overbought/oversold with trend context
        - Williams %R       → momentum divergence signal
        - ROC (14)          → rate of change, trend acceleration
        - CCI (20)          → commodity channel index, cycle timing
        - candle_body_ratio → bullish/bearish candle strength

        These complement RSI/MACD and improve entry timing.
        """
        # ── Stochastic Oscillator %K and %D ──────
        # %K = (Close - LowestLow) / (HighestHigh - LowestLow) × 100
        # %D = 3-period SMA of %K (signal line)
        period = 14
        lowest_low   = df["low"].rolling(window=period).min()
        highest_high = df["high"].rolling(window=period).max()
        denom = (highest_high - lowest_low).replace(0, np.finfo(float).eps)
        df["stoch_k"] = (df["close"] - lowest_low) / denom * 100
        df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

        # ── Williams %R ───────────────────────────
        # Range: -100 to 0  (we normalize to 0-1 in normalizer)
        # Near 0 = overbought, near -100 = oversold
        df["williams_r"] = (
            (highest_high - df["close"]) / denom * -100
        )

        # ── Rate of Change (ROC) ──────────────────
        # % change over N periods — measures trend momentum
        roc_period = 14
        df["roc_14"] = (
            (df["close"] - df["close"].shift(roc_period))
            / df["close"].shift(roc_period).replace(0, np.finfo(float).eps)
            * 100
        )

        # ── Commodity Channel Index (CCI) ─────────
        # Measures distance from statistical mean
        # >100 = overbought, <-100 = oversold
        cci_period = 20
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        tp_sma = typical_price.rolling(window=cci_period).mean()
        tp_mad = typical_price.rolling(window=cci_period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        df["cci_20"] = (typical_price - tp_sma) / (0.015 * tp_mad.replace(0, np.finfo(float).eps))

        # ── Candlestick Body Ratio ────────────────
        # (close - open) / (high - low) → [-1, 1]
        # +1 = full bullish candle, -1 = full bearish candle
        candle_range = (df["high"] - df["low"]).replace(0, np.finfo(float).eps)
        df["candle_body"] = (df["close"] - df["open"]) / candle_range

        logger.debug("   ✓ Momentum indicators (Stoch, Williams %R, ROC, CCI, candle body)")
        return df