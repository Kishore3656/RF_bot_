# broker/binance_broker.py

import os
from pathlib import Path
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException
from loguru import logger

# Force find .env from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


class BinanceBroker:
    """
    Handles all communication with Binance.

    Responsibilities:
    - Connect to Binance (testnet or live)
    - Fetch current prices
    - Get account balance
    - Place buy/sell orders
    - Check order status

    First principles:
    The broker is the ONLY class that talks to Binance.
    Everything else (AI, Risk) is completely isolated
    from the broker. This means:
    - Easy to swap Binance for another exchange later
    - Easy to test without real money (testnet)
    - Clear separation of concerns
    """

    def __init__(self, config: dict):
        self.config   = config
        # self.testnet  = os.getenv("BINANCE_TESTNET", "True") == "True"
        self.testnet = False   # hardcoded for now
        api_key       = os.getenv("BINANCE_API_KEY",    "")
        api_secret    = os.getenv("BINANCE_API_SECRET", "")

        if not api_key or not api_secret:
            raise ValueError("❌ Missing BINANCE_API_KEY or BINANCE_API_SECRET in .env")

        # Connect to testnet or live
        if self.testnet:
            self.client = Client(api_key, api_secret, testnet=True)
            logger.info("🔧 Binance connected — TESTNET (fake money)")
        else:
            self.client = Client(api_key, api_secret)
            logger.warning("🚨 Binance connected — LIVE (real money!)")

        logger.info(f"   Testnet: {self.testnet}")

    # ── Price ──────────────────────────────────

    def get_price(self, symbol: str) -> float:
        """
        Get current market price for a symbol.
        symbol: e.g. "BTCUSDT", "ETHUSDT"
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            price  = float(ticker["price"])
            logger.debug(f"   💲 {symbol}: ${price:,.2f}")
            return price
        except BinanceAPIException as e:
            logger.error(f"❌ Price fetch failed: {e}")
            return 0.0

    def get_klines(
        self,
        symbol:   str,
        interval: str = "1d",
        limit:    int = 365,
    ) -> list:
        """
        Get historical candlestick data.
        interval: "1m", "5m", "1h", "1d" etc.
        Returns list of OHLCV dicts.
        """
        interval_map = {
            "1m":  Client.KLINE_INTERVAL_1MINUTE,
            "5m":  Client.KLINE_INTERVAL_5MINUTE,
            "1h":  Client.KLINE_INTERVAL_1HOUR,
            "1d":  Client.KLINE_INTERVAL_1DAY,
        }
        binance_interval = interval_map.get(interval, Client.KLINE_INTERVAL_1DAY)

        try:
            raw = self.client.get_klines(
                symbol   = symbol,
                interval = binance_interval,
                limit    = limit,
            )
            candles = []
            for k in raw:
                candles.append({
                    "open":   float(k[1]),
                    "high":   float(k[2]),
                    "low":    float(k[3]),
                    "close":  float(k[4]),
                    "volume": float(k[5]),
                })
            logger.info(
                f"📊 {symbol}: fetched {len(candles)} "
                f"{interval} candles"
            )
            return candles
        except BinanceAPIException as e:
            logger.error(f"❌ Klines fetch failed: {e}")
            return []

    # ── Account ────────────────────────────────

    def get_balance(self, asset: str = "USDT") -> float:
        """
        Get free balance for an asset.
        asset: "USDT", "BTC", "ETH" etc.
        """
        try:
            info    = self.client.get_asset_balance(asset=asset)
            balance = float(info["free"])
            logger.info(f"   💰 {asset} balance: {balance:,.4f}")
            return balance
        except BinanceAPIException as e:
            logger.error(f"❌ Balance fetch failed: {e}")
            return 0.0

    def get_portfolio_value(self, symbol: str = "BTCUSDT") -> float:
        """
        Total portfolio value in USDT.
        USDT balance + BTC holdings converted to USDT.
        """
        try:
            usdt_balance = self.get_balance("USDT")
            base_asset   = symbol.replace("USDT", "")
            crypto_bal   = self.get_balance(base_asset)
            price        = self.get_price(symbol)
            total        = usdt_balance + (crypto_bal * price)
            logger.info(f"   📊 Portfolio value: ${total:,.2f} USDT")
            return total
        except Exception as e:
            logger.error(f"❌ Portfolio value failed: {e}")
            return 0.0

    # ── Orders ─────────────────────────────────

    def buy_market(
        self,
        symbol:      str,
        usdt_amount: float,
    ) -> dict:
        """
        Place a market BUY order.
        usdt_amount: how many USDT to spend (from PositionSizer)
        """
        try:
            # Get current price to calculate quantity
            price    = self.get_price(symbol)
            quantity = round(usdt_amount / price, 6)

            logger.info(
                f"🟢 BUY {symbol} | "
                f"${usdt_amount:,.2f} USDT | "
                f"{quantity} units @ ${price:,.2f}"
            )

            order = self.client.order_market_buy(
                symbol   = symbol,
                quantity = quantity,
            )
            logger.info(f"   ✅ Order ID: {order['orderId']}")
            return {
                "success":  True,
                "order_id": order["orderId"],
                "symbol":   symbol,
                "side":     "BUY",
                "quantity": quantity,
                "price":    price,
            }
        except BinanceAPIException as e:
            logger.error(f"❌ Buy order failed: {e}")
            return {"success": False, "error": str(e)}

    def sell_market(
        self,
        symbol:   str,
        quantity: float,
    ) -> dict:
        """
        Place a market SELL order.
        quantity: how many units of crypto to sell
        """
        try:
            price = self.get_price(symbol)
            logger.info(
                f"🔴 SELL {symbol} | "
                f"{quantity} units @ ${price:,.2f}"
            )

            order = self.client.order_market_sell(
                symbol   = symbol,
                quantity = quantity,
            )
            logger.info(f"   ✅ Order ID: {order['orderId']}")
            return {
                "success":  True,
                "order_id": order["orderId"],
                "symbol":   symbol,
                "side":     "SELL",
                "quantity": quantity,
                "price":    price,
            }
        except BinanceAPIException as e:
            logger.error(f"❌ Sell order failed: {e}")
            return {"success": False, "error": str(e)}

    def get_open_position(self, symbol: str) -> dict:
        """
        Check if we currently hold any of this asset.
        Returns quantity held (0 if none).
        """
        base_asset = symbol.replace("USDT", "")
        quantity   = self.get_balance(base_asset)
        return {
            "symbol":       symbol,
            "quantity":     quantity,
            "in_position":  quantity > 0.0001,
        }