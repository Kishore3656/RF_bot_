# data/normalizer.py

import pandas as pd
import numpy as np
from loguru import logger


class DataNormalizer:
    """
    Scales all features to 0-1 range for neural network input.
    Single responsibility: normalization only, nothing else.
    """

    def __init__(self, config: dict):
        """
        Set up which columns get which scaling method.
        """
        self.config = config

        # Rolling window size for price-based features
        self.window = 50

        # ── Column groups by scaling method ──────

        # Fixed range columns (we KNOW their min/max)
        self.fixed_range_cols = {
            "rsi_7":       (0, 100),
            "rsi_14":      (0, 100),
            "bb_position": (0, 1),
            "bb_width":    (0, 1),
            "ema_cross":   (0, 1),
            "volume_ratio":(0, 5),   # rarely goes above 5x
            "atr_pct":     (0, 10),  # rarely above 10%
        }

        # Rolling window columns (scale relative to recent history)
        self.rolling_cols = [
            "open", "high", "low", "close",
            "ema_20", "ema_50", "ema_200",
            "bb_upper", "bb_middle", "bb_lower",
            "atr_14",
        ]

        # Special columns (need their own treatment)
        self.special_cols = [
            "macd_line",
            "macd_signal",
            "macd_histogram",
            "volume",
            "volume_sma_20",
            "obv",
        ]

        # Columns to DROP before feeding to neural network
        # (text columns, not numbers)
        self.drop_cols = ["symbol", "asset_type"]

        logger.info("📐 DataNormalizer ready")

    # ─────────────────────────────────────────
    # PUBLIC METHOD
    # ─────────────────────────────────────────

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Master function — normalizes all columns.
        Input:  enriched DataFrame (26 columns)
        Output: normalized DataFrame (all values 0 to 1)
        """
        symbol = (
            df["symbol"].iloc[0]
            if "symbol" in df.columns
            else "unknown"
        )
        logger.info(f"📐 Normalizing {symbol}...")

        # Work on a copy — never touch original
        df = df.copy()

        # Step 1: Drop text columns
        df = self._drop_text_columns(df)

        # Step 2: Scale fixed-range columns
        df = self._scale_fixed_range(df)

        # Step 3: Scale rolling window columns
        df = self._scale_rolling(df)

        # Step 4: Scale special columns
        df = self._scale_special(df)

        # Step 5: Clip any outliers that sneak through
        df = self._clip_outliers(df)

        # Step 6: Final check
        df = self._verify_normalization(df)

        logger.success(
            f"✅ {symbol}: Normalized "
            f"{len(df.columns)} columns, "
            f"{len(df)} rows"
        )
        return df

    def normalize_batch(self, data_dict: dict) -> dict:
        """
        Normalize a whole dictionary of assets.
        """
        normalized = {}
        for symbol, df in data_dict.items():
            result = self.normalize(df)
            if result is not None:
                normalized[symbol] = result

        logger.success(
            f"✅ Normalized {len(normalized)} assets"
        )
        return normalized

    # ─────────────────────────────────────────
    # PRIVATE METHODS
    # ─────────────────────────────────────────

    def _drop_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns the neural network can't use.
        Text like "AAPL" or "stock" means nothing to math.
        """
        cols_to_drop = [
            c for c in self.drop_cols
            if c in df.columns
        ]
        df = df.drop(columns=cols_to_drop)
        logger.debug(f"   ✓ Dropped text columns: {cols_to_drop}")
        return df

    def _scale_fixed_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Min-Max scale columns where we know the range.

        Formula: scaled = (value - min) / (max - min)
        """
        for col, (min_val, max_val) in self.fixed_range_cols.items():
            if col not in df.columns:
                continue

            df[col] = (df[col] - min_val) / (max_val - min_val)

            # Clip in case any values sneak outside known range
            df[col] = df[col].clip(0, 1)

        logger.debug(
            f"   ✓ Fixed-range scaled: "
            f"{list(self.fixed_range_cols.keys())}"
        )
        return df

    def _scale_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale price-based columns using a rolling window.

        Instead of using the ALL-TIME min/max (which causes
        problems — a price from 3 years ago distorts today's
        scaling), we use the min/max of the last 50 days only.

        This keeps scaling RELEVANT to current market context.
        """
        for col in self.rolling_cols:
            if col not in df.columns:
                continue

            # Calculate rolling min and max
            roll_min = df[col].rolling(window=self.window).min()
            roll_max = df[col].rolling(window=self.window).max()

            # Avoid division by zero when min == max (flat price)
            roll_range = roll_max - roll_min
            roll_range = roll_range.replace(0, np.finfo(float).eps)

            # Apply the scaling
            df[col] = (df[col] - roll_min) / roll_range

            # Clip to 0-1 range
            df[col] = df[col].clip(0, 1)

        logger.debug(
            f"   ✓ Rolling-window scaled: {self.rolling_cols}"
        )
        return df

    def _scale_special(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale columns that need special treatment.

        MACD, OBV, Volume can be negative or astronomically large.
        We use a rolling z-score then squish to 0-1.

        Z-score = how many standard deviations from the average?
        Then sigmoid converts that to 0-1 smoothly.
        """
        for col in self.special_cols:
            if col not in df.columns:
                continue

            # Rolling mean and standard deviation
            roll_mean = df[col].rolling(window=self.window).mean()
            roll_std  = df[col].rolling(window=self.window).std()

            # Avoid division by zero
            roll_std = roll_std.replace(0, np.finfo(float).eps)

            # Z-score: how far from average (in standard deviations)
            z_score = (df[col] - roll_mean) / roll_std

            # Sigmoid: converts any number to 0-1 smoothly
            # sigmoid(0)  = 0.5  (average)
            # sigmoid(3)  ≈ 0.95 (very high)
            # sigmoid(-3) ≈ 0.05 (very low)
            df[col] = 1 / (1 + np.exp(-z_score))

        logger.debug(
            f"   ✓ Special scaled: {self.special_cols}"
        )
        return df

    def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final safety net — clip ALL columns to 0-1.
        No matter what happened above, nothing escapes
        the 0 to 1 range.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].clip(0, 1)
        logger.debug("   ✓ Outliers clipped to [0, 1]")
        return df

    def _verify_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check for any remaining NaN values and report.
        Drop remaining NaN rows (from rolling window warmup).
        """
        before = len(df)
        df.dropna(inplace=True)
        after  = len(df)

        if before != after:
            logger.debug(
                f"   ✓ Dropped {before - after} "
                f"rolling warmup rows"
            )

        # Quick sanity check
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        out_of_range = []

        for col in numeric_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min < -0.01 or col_max > 1.01:
                out_of_range.append(
                    f"{col} ({col_min:.3f} to {col_max:.3f})"
                )

        if out_of_range:
            logger.warning(
                f"⚠️ Columns outside [0,1]: {out_of_range}"
            )
        else:
            logger.debug("   ✓ All columns verified in [0, 1] range")

        return df