import os
from pathlib import Path
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException
from loguru import logger

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

class PaperBroker:
    """
    Paper trading broker — simulates orders with real prices.

    Responsibilities:
    - Fetch REAL live prices from Binance (read-only)
    - Simulate BUY/SELL orders with virtual balance
    - Track virtual portfolio and P&L
    - Zero real orders ever placed

    Drop-in replacement for BinanceBroker.
    Same method signatures — AI/Risk never knows the difference.
    """

    STARTING_BALANCE = 1_000.0   # virtual USDT to start with

    def __init__(self, config: dict):
        self.config = config

        # Real Binance client for live prices only
        api_key    = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")

        if api_key and api_secret:
            self.client = Client(api_key, api_secret)
            logger.info("📡 PaperBroker: live prices via Binance")
        else:
            self.client = None
            logger.warning("⚠️  PaperBroker: no API keys — prices will be 0.0")

        # Virtual portfolio state
        self._usdt    = self.STARTING_BALANCE
        self._holdings: dict[str, float] = {}   # symbol → quantity
        self._order_id = 1000

        logger.info("📄 PaperBroker ready — DRY RUN MODE")
        logger.info(f"   Virtual balance: ${self._usdt:,.2f} USDT")

    # ── Prices (real) ──────────────────────────────

    def get_price(self, symbol: str) -> float:
        if not self.client:
            return 0.0
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
        if not self.client:
            return []
        interval_map = {
            "1m":  Client.KLINE_INTERVAL_1MINUTE,
            "5m":  Client.KLINE_INTERVAL_5MINUTE,
            "1h":  Client.KLINE_INTERVAL_1HOUR,
            "1d":  Client.KLINE_INTERVAL_1DAY,
        }
        binance_interval = interval_map.get(interval, Client.KLINE_INTERVAL_1DAY)
        try:
            raw = self.client.get_klines(
                symbol=symbol, interval=binance_interval, limit=limit
            )
            candles = [
                {
                    "open":   float(k[1]),
                    "high":   float(k[2]),
                    "low":    float(k[3]),
                    "close":  float(k[4]),
                    "volume": float(k[5]),
                }
                for k in raw
            ]
            logger.info(f"📊 {symbol}: fetched {len(candles)} {interval} candles")
            return candles
        except BinanceAPIException as e:
            logger.error(f"❌ Klines fetch failed: {e}")
            return []

    # ── Virtual account ────────────────────────────

    def get_balance(self, asset: str = "USDT") -> float:
        if asset == "USDT":
            logger.info(f"   💰 [PAPER] USDT balance: {self._usdt:,.4f}")
            return self._usdt
        qty = self._holdings.get(asset, 0.0)
        logger.info(f"   💰 [PAPER] {asset} balance: {qty:.6f}")
        return qty

    def get_portfolio_value(self, symbol: str = "BTCUSDT") -> float:
        base_asset  = symbol.replace("USDT", "")
        crypto_qty  = self._holdings.get(base_asset, 0.0)
        price       = self.get_price(symbol)
        total       = self._usdt + (crypto_qty * price)
        logger.info(f"   📊 [PAPER] Portfolio value: ${total:,.2f} USDT")
        return total

    def get_open_position(self, symbol: str) -> dict:
        base_asset = symbol.replace("USDT", "")
        quantity   = self._holdings.get(base_asset, 0.0)
        return {
            "symbol":      symbol,
            "quantity":    quantity,
            "in_position": quantity > 0.0001,
        }

    # ── Simulated orders ───────────────────────────

    def buy_market(self, symbol: str, usdt_amount: float) -> dict:
        price      = self.get_price(symbol)
        if price == 0.0:
            return {"success": False, "error": "No price available"}

        # Cap to available balance
        usdt_amount = min(usdt_amount, self._usdt)
        quantity    = round(usdt_amount / price, 6)

        # Update virtual portfolio
        self._usdt -= usdt_amount
        base_asset  = symbol.replace("USDT", "")
        self._holdings[base_asset] = self._holdings.get(base_asset, 0.0) + quantity

        order_id = self._order_id
        self._order_id += 1

        logger.info(
            f"🟢 [PAPER] BUY {symbol} | "
            f"${usdt_amount:,.2f} USDT | "
            f"{quantity:.6f} units @ ${price:,.2f}"
        )
        logger.info(f"   💼 [PAPER] USDT remaining: ${self._usdt:,.2f}")

        return {
            "success":  True,
            "order_id": order_id,
            "symbol":   symbol,
            "side":     "BUY",
            "quantity": quantity,
            "price":    price,
            "paper":    True,
        }

    def sell_market(self, symbol: str, quantity: float) -> dict:
        base_asset = symbol.replace("USDT", "")
        held       = self._holdings.get(base_asset, 0.0)

        if held < quantity:
            logger.warning(
                f"⚠️  [PAPER] Tried to sell {quantity:.6f} "
                f"but only hold {held:.6f} {base_asset}"
            )
            quantity = held   # sell what we have

        price    = self.get_price(symbol)
        proceeds = quantity * price

        # Update virtual portfolio
        self._holdings[base_asset] = held - quantity
        self._usdt += proceeds

        order_id = self._order_id
        self._order_id += 1

        logger.info(
            f"🔴 [PAPER] SELL {symbol} | "
            f"{quantity:.6f} units @ ${price:,.2f} | "
            f"Proceeds: ${proceeds:,.2f}"
        )
        logger.info(f"   💼 [PAPER] USDT after sell: ${self._usdt:,.2f}")

        return {
            "success":  True,
            "order_id": order_id,
            "symbol":   symbol,
            "side":     "SELL",
            "quantity": quantity,
            "price":    price,
            "paper":    True,
        }