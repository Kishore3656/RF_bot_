# data/fetcher.py

import pandas as pd
import yfinance as yf
from loguru import logger
from datetime import datetime, timedelta
import time


class MarketDataFetcher:
    """
    Fetches market data for stocks, crypto, and forex.
    Single responsibility: get clean OHLCV data, nothing else.
    """

    # Valid timeframes yfinance understands
    VALID_TIMEFRAMES = {
        "1m":  "1 minute",
        "5m":  "5 minutes",
        "15m": "15 minutes",
        "1h":  "1 hour",
        "1d":  "1 day",
        "1wk": "1 week"
    }

    def __init__(self, config: dict):
        """
        config: the full config.yaml loaded as a dictionary
        """
        self.config = config
        self.timeframe = config["data"]["timeframe"]
        self.lookback_days = config["data"]["lookback_days"]

        logger.info(f"📊 MarketDataFetcher ready")
        logger.info(f"   Timeframe:     {self.timeframe}")
        logger.info(f"   Lookback:      {self.lookback_days} days")

    # ─────────────────────────────────────────
    # PUBLIC METHODS (what the rest of the
    # system calls)
    # ─────────────────────────────────────────

    def fetch_stock(self, symbol: str) -> pd.DataFrame:
        """
        Fetch stock data for a single symbol.
        Example: fetch_stock("AAPL")
        """
        logger.info(f"📈 Fetching stock: {symbol}")
        return self._fetch_yfinance(symbol, asset_type="stock")

    def fetch_crypto(self, symbol: str) -> pd.DataFrame:
        """
        Fetch crypto data for a single symbol.
        Example: fetch_crypto("BTC-USD")
        """
        logger.info(f"🪙 Fetching crypto: {symbol}")
        return self._fetch_yfinance(symbol, asset_type="crypto")

    def fetch_forex(self, symbol: str) -> pd.DataFrame:
        """
        Fetch forex data for a single symbol.
        Example: fetch_forex("EURUSD=X")
        """
        logger.info(f"💱 Fetching forex: {symbol}")
        return self._fetch_yfinance(symbol, asset_type="forex")

    def fetch_all(self) -> dict:
        """
        Fetch ALL assets defined in config.yaml
        Returns a dictionary: { "AAPL": DataFrame, "BTC-USD": DataFrame, ... }
        """
        logger.info("🌍 Fetching all configured assets...")
        all_data = {}

        assets = self.config["data"]["assets"]

        # Fetch stocks
        for symbol in assets.get("stocks", []):
            df = self.fetch_stock(symbol)
            if df is not None:
                all_data[symbol] = df
            time.sleep(0.5)  # Be polite to the API

        # Fetch crypto
        for symbol in assets.get("crypto", []):
            df = self.fetch_crypto(symbol)
            if df is not None:
                all_data[symbol] = df
            time.sleep(0.5)

        # Fetch forex
        for symbol in assets.get("forex", []):
            df = self.fetch_forex(symbol)
            if df is not None:
                all_data[symbol] = df
            time.sleep(0.5)

        logger.info(f"✅ Fetched {len(all_data)} assets successfully")
        return all_data

    # ─────────────────────────────────────────
    # PRIVATE METHODS (internal helpers)
    # ─────────────────────────────────────────

    def _fetch_yfinance(self, symbol: str, asset_type: str) -> pd.DataFrame:
        """
        Core fetch logic using yfinance.
        Returns clean OHLCV DataFrame or None if fetch fails.
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)

            # Download from Yahoo Finance
            raw = yf.download(
                symbol,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval=self.timeframe,
                progress=False,   # Suppress download bar
                auto_adjust=True  # Adjust for splits/dividends
            )

            # Check we actually got data
            if raw is None or raw.empty:
                logger.warning(f"⚠️ No data returned for {symbol}")
                return None

            # Clean and standardize the DataFrame
            df = self._clean_dataframe(raw, symbol, asset_type)

            logger.success(
                f"✅ {symbol}: {len(df)} rows | "
                f"{df.index[0].date()} → {df.index[-1].date()}"
            )
            return df

        except Exception as e:
            logger.error(f"❌ Failed to fetch {symbol}: {e}")
            return None

    def _clean_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        asset_type: str
    ) -> pd.DataFrame:
        """
        Standardize column names and add metadata.
        Every asset comes out in exactly the same format.
        """
        # yfinance sometimes returns MultiIndex columns - flatten them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Standardize column names to lowercase
        df.columns = [col.lower() for col in df.columns]

        # Keep only what we need
        required_cols = ["open", "high", "low", "close", "volume"]
        df = df[required_cols].copy()

        # Remove any rows with missing values
        before = len(df)
        df.dropna(inplace=True)
        after = len(df)

        if before != after:
            logger.warning(
                f"⚠️ {symbol}: Dropped {before - after} rows with NaN values"
            )

        # Remove rows with zero or negative prices (data errors)
        df = df[df["close"] > 0]

        # Add metadata columns (useful for multi-asset systems)
        df["symbol"] = symbol
        df["asset_type"] = asset_type

        # Make sure index is datetime
        df.index = pd.to_datetime(df.index)
        df.index.name = "datetime"

        return df


    def fetch_yfinance(self, ticker: str, symbol: str = None) -> pd.DataFrame:
        """
        Fetch data from yfinance by ticker.
        Example: fetch_yfinance("GC=F", symbol="GOLD")
        Used when config source: "yfinance"
        """
        symbol = symbol or ticker
        logger.info(f"📈 Fetching yfinance: {ticker} | Symbol: {symbol}")
        df = self._fetch_yfinance(ticker, asset_type="commodity")
        if df is not None:
            df["symbol"] = symbol
        return df

    def fetch_csv(self, filepath: str, symbol: str = "GOLD") -> pd.DataFrame: 
        logger.info(f"📂 Loading CSV: {filepath}")

        df = pd.read_csv(filepath)

        # Rename columns to match our pipeline
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={
        "Date":   "date",
        "Open":   "open",
        "High":   "high",
        "Low":    "low",
        "Close":  "close",
        "Volume": "volume",
        })

        # Parse date
        df["date"] = pd.to_datetime(df["date"], dayfirst=True)
        df = df.sort_values("date").reset_index(drop=True)
        df.set_index("date", inplace=True)

        # Add required columns
        df["symbol"]     = symbol
        df["asset_type"] = "commodity"

        # Ensure numeric
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna()

        logger.success(f"✅ {symbol}: {len(df)} rows | "
                   f"{df.index[0].date()} → {df.index[-1].date()}")
        return df