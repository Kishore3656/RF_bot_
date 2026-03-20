# main.py

import yaml
from loguru import logger
from data import MarketDataFetcher, IndicatorEngine, DataNormalizer
from ai   import AgentTrainer, InferenceEngine
from broker import BinanceBroker, PaperBroker   

def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError("❌ config.yaml is empty!")
    return config


def main():
    config = load_config()
    logger.info("🚀 Trading Bot Starting...")

    # ── Data pipeline ──────────────────────────
    fetcher    = MarketDataFetcher(config)
    engine     = IndicatorEngine(config)
    normalizer = DataNormalizer(config)

    data_cfg = config["data"]
    source   = data_cfg.get("source", "csv").lower()   # "csv" | "binance" | "yfinance"
    symbol   = data_cfg.get("symbol", "BTCUSDT")

    logger.info(f"📡 Data source: {source.upper()} | Symbol: {symbol}")

    if source == "csv":
        csv_path = data_cfg.get("csv_path")
        if not csv_path:
            raise ValueError("❌ config.yaml: data.csv_path is required when source=csv")
        raw = fetcher.fetch_csv(csv_path, symbol=symbol)

    elif source == "yfinance":
        ticker = data_cfg.get("yfinance_ticker", symbol)
        raw    = fetcher.fetch_yfinance(ticker, symbol=symbol)

    elif source == "binance":
        raw = fetcher.fetch_binance(
            symbol   = symbol,
            interval = data_cfg.get("timeframe", "1d"),
            limit    = data_cfg.get("lookback_days", 365),
        )

    else:
        raise ValueError(
            f"❌ Unknown data source: '{source}'. "
            f"Use 'csv', 'yfinance', or 'binance' in config.yaml"
        )

    enriched   = engine.compute_all(raw)
    raw_prices = raw["close"].reindex(enriched.index)
    normalized = normalizer.normalize(enriched)
    raw_prices = raw_prices.reindex(normalized.index).dropna()
    normalized = normalized.loc[raw_prices.index]
    '''
    mode = config.get("broker", {}).get("mode", "paper")
    if mode == "live":
        broker = BinanceBroker(config)
    else:
        broker = PaperBroker(config)
    '''
    # ── PHASE A: Train and save ────────────────
    logger.info("\n" + "="*50)
    logger.info("PHASE A: Training")
    logger.info("="*50)

    trainer = AgentTrainer(config)
    results = trainer.train_and_evaluate(
        normalized_df = normalized,
        raw_prices    = raw_prices,
    )

    # ── PHASE B: Load and run inference ───────
    logger.info("\n" + "="*50)
    logger.info("PHASE B: Load saved model & run inference")
    logger.info("="*50)

    engine_inf = InferenceEngine(config)
    engine_inf.load_model()

    # Run backtest with the LOADED model
    # (proves save/load works — not just in-memory)
    summary = engine_inf.run_backtest(normalized, raw_prices)

    # ── PHASE C: Single prediction demo ───────
    logger.info("\n" + "="*50)
    logger.info("PHASE C: Single-step prediction (live trading demo)")
    logger.info("="*50)

    # Simulate "today's" observation = last row of data
    last_obs = normalized.iloc[-1].values.astype("float32")

    # Pad with portfolio state (5 zeros = no position, full cash)
    portfolio_state = [1.0, 0.0, 0.5, 0.5, 0.5, 0.0]
    full_obs = __import__("numpy").concatenate(
        [last_obs, portfolio_state]
    ).astype("float32")

    decision = engine_inf.predict(
        observation     = full_obs,
        portfolio_value = config["broker"]["initial_capital"],
    )

    logger.info(f"\n🤖 Bot decision for TODAY:")
    logger.info(f"   Action:     {decision['action_name']}")
    logger.info(f"   Confidence: {decision['confidence']}%")
    logger.info(f"   Certain:    {decision['certain']}")
    logger.info(f"   All probs:")
    for name, pct in decision["probs"].items():
        bar = "█" * int(pct / 5)
        logger.info(f"      {name}: {pct:5.1f}% {bar}")

    # Confidence threshold warning
    CONFIDENCE_THRESHOLD = 55.0
    if decision["confidence"] < CONFIDENCE_THRESHOLD:
        logger.warning(
            f"   ⚠️  Low confidence ({decision['confidence']}% < {CONFIDENCE_THRESHOLD}%) "
            f"— bot would SKIP this trade in live mode"
        )

    # ── PHASE D: Position Sizing Demo ─────────   ← INSIDE main()
    logger.info("\n" + "="*50)
    logger.info("PHASE D: Position Sizing Demo")
    logger.info("="*50)

    from risk import PositionSizer
    methods = ["fixed_pct", "kelly", "volatility", "fixed_dollar"]
    for method in methods:
        config["risk"]["position_sizing"] = method
        sizer = PositionSizer(config)
        result = sizer.calculate(
            portfolio_value   = 10_000,
            current_price     = 227.48,
            win_rate          = 0.55,
            avg_win           = 1.8,
            avg_loss          = 1.0,
            recent_volatility = 0.018,
        )
        logger.info(
            f"\n   [{method}]"
            f"\n   Fraction:    {result['fraction']*100:.1f}%"
            f"\n   Dollar size: ${result['dollar_size']:,.0f}"
            f"\n   Units:       {result['units']:.2f} shares"
            f"\n   Reasoning:   {result['reasoning']}"
        )

    

    # ── PHASE E: Stop Loss Demo ────────────────
    logger.info("\n" + "="*50)
    logger.info("PHASE E: Stop Loss & Take Profit Demo")
    logger.info("="*50)

    from risk import StopLossManager

    sl_manager = StopLossManager(config)

    # Simulate a LONG trade on Apple
    entry = 246.47   # test period start price
    sl_manager.open_trade(entry_price=entry, direction="long")

    # Simulate price path over 10 days
    prices = [246.47, 248.10, 251.33, 255.80, 261.20,
              263.45, 258.90, 252.30, 244.10, 240.50]

    logger.info(f"\n   Simulating LONG trade entered at ${entry:.2f}")
    logger.info(f"   Stop Loss:   -5%  → exit at ${entry*0.95:.2f}")
    logger.info(f"   Take Profit: +15% → exit at ${entry*1.15:.2f}")
    logger.info(f"\n   {'Day':>4} | {'Price':>8} | {'P&L':>8} | Status")
    logger.info(f"   {'-'*45}")

    for day, price in enumerate(prices, 1):
        result = sl_manager.check(price)
        status = f"🔴 EXIT — {result['reason']}" if result["should_exit"] \
                 else "✅ Hold"
        logger.info(
            f"   {day:>4} | ${price:>7.2f} | "
            f"{result['pnl_pct']:>+7.2f}% | {status}"
        )
        if result["should_exit"]:
            sl_manager.close_trade()
            break

    # Now test trailing stop
    logger.info(f"\n   --- Trailing Stop Demo (7% trail) ---")
    config["risk"]["trailing_stop"] = True
    config["risk"]["trailing_pct"]  = 0.07
    sl_trail = StopLossManager(config)
    sl_trail.open_trade(entry_price=entry, direction="long")

    prices_bull = [246.47, 252.00, 258.00, 265.00,
                   271.00, 277.86, 271.00, 259.00, 257.50]

    logger.info(f"\n   {'Day':>4} | {'Price':>8} | {'Best':>8} | {'P&L':>8} | Status")
    logger.info(f"   {'-'*55}")

    for day, price in enumerate(prices_bull, 1):
        result = sl_trail.check(price)
        status = f"🔴 EXIT — {result['reason']}" if result["should_exit"] \
                 else "✅ Hold"
        logger.info(
            f"   {day:>4} | ${price:>7.2f} | "
            f"${sl_trail.best_price:>7.2f} | "
            f"{result['pnl_pct']:>+7.2f}% | {status}"
        )
        if result["should_exit"]:
            sl_trail.close_trade()
            break
   # ── PHASE F: Portfolio Rules Demo ─────────
    logger.info("\n" + "="*50)
    logger.info("PHASE F: Portfolio-Level Controls Demo")
    logger.info("="*50)

    from risk import PortfolioRules

    rules = PortfolioRules(config)

    # Simulate a trading day starting at $10,000
    start_capital = 10_000.0
    rules.start_day(start_capital)

    logger.info(f"\n   --- Scenario 1: Normal trading day ---")
    normal_values = [10_000, 10_050, 10_120, 10_080, 10_200]
    for i, val in enumerate(normal_values, 1):
        status = rules.can_open_trade(val)
        logger.info(
            f"   Tick {i}: ${val:,.0f} | "
            f"Daily: {status['daily_pnl']:+.2f}% | "
            f"DD: {status['drawdown']:.2f}% | "
            f"{'✅ Can trade' if status['allowed'] else '🚫 ' + status['reason']}"
        )

    logger.info(f"\n   --- Scenario 2: Bad day — hits daily limit ---")
    rules2 = PortfolioRules(config)
    rules2.start_day(start_capital)
    bad_day = [10_000, 9_900, 9_780, 9_650, 9_500, 9_400]
    for i, val in enumerate(bad_day, 1):
        status = rules2.can_open_trade(val)
        logger.info(
            f"   Tick {i}: ${val:,.0f} | "
            f"Daily: {status['daily_pnl']:+.2f}% | "
            f"DD: {status['drawdown']:.2f}% | "
            f"{'✅ Can trade' if status['allowed'] else '🚫 HALT: ' + status['halt_reason']}"
        )

    logger.info(f"\n   --- Scenario 3: Max drawdown breach ---")
    rules3 = PortfolioRules(config)
    rules3.start_day(12_000)   # peak was $12,000
    rules3.peak_value = 12_000
    drawdown_vals = [11_500, 11_000, 10_500, 10_200, 10_000]
    for i, val in enumerate(drawdown_vals, 1):
        status = rules3.can_open_trade(val)
        logger.info(
            f"   Tick {i}: ${val:,.0f} | "
            f"Daily: {status['daily_pnl']:+.2f}% | "
            f"DD: {status['drawdown']:.2f}% | "
            f"{'✅ Can trade' if status['allowed'] else '🚫 HALT: ' + status['halt_reason']}"
        )

    logger.info(f"\n   --- Scenario 4: Max open trades ---")
    rules4 = PortfolioRules(config)
    rules4.start_day(start_capital)
    rules4.register_trade_open()   # simulate 1 open trade
    status = rules4.can_open_trade(start_capital)
    logger.info(
        f"   Trying to open trade 2 of {config['risk'].get('max_open_trades', 1)} max: "
        f"{'✅ Allowed' if status['allowed'] else '🚫 ' + status['reason']}"
    )
    # ── PHASE G: Full Risk Manager Demo ───────
    logger.info("\n" + "="*50)
    logger.info("PHASE G: Full Risk Manager Demo")
    logger.info("="*50)

    from risk import RiskManager
    import numpy as np

    config["risk"]["position_sizing"] = "kelly"
    rm = RiskManager(config)
    rm.start_day(10_000)

    # Simulate a sequence of AI decisions + prices
    events = [
        # (day, ai_action, price, description)
        (1,  0, 246.47, "AI says HOLD"),
        (2,  1, 248.10, "AI says LONG ← open position"),
        (3,  0, 251.33, "AI says HOLD (in position)"),
        (4,  0, 255.80, "AI says HOLD (in position)"),
        (5,  0, 234.00, "Price crashes — stop loss!"),
        (6,  1, 235.00, "AI says LONG again after stop"),
        (7,  3, 242.00, "AI says CLOSE"),
        (8,  2, 240.00, "AI says SHORT"),
        (9,  0, 238.00, "AI says HOLD (in short)"),
        (10, 3, 235.00, "AI says CLOSE short (profit!)"),
    ]

    logger.info(f"\n   {'Day':>4} | {'Price':>8} | {'AI':>6} | Result")
    logger.info(f"   {'-'*60}")

    portfolio    = 10_000.0
    seen_days    = set()          # ← guard: skip duplicate day entries
    for day, action, price, desc in events:
        if day in seen_days:
            continue
        seen_days.add(day)

        action_names = {0:"HOLD", 1:"LONG", 2:"SHORT", 3:"CLOSE"}
        result = rm.evaluate(
            ai_action       = action,
            current_price   = price,
            portfolio_value = portfolio,
            win_rate        = 0.55,
            avg_win         = 1.8,
            avg_loss        = 1.0,
            volatility      = 0.018,
        )
        status = (
            f"✅ {result['action_name']}"
            if result["approved"]
            else f"🚫 {result['reason']}"
        )
        override = " ⚠️ OVERRIDE" if result["override"] else ""
        logger.info(
            f"   {day:>4} | ${price:>7.2f} | "
            f"{action_names[action]:>6} | "
            f"{status}{override}"
        )
        if result.get("stop_price"):
            logger.info(
                f"        SL: ${result['stop_price']:.2f} | "
                f"TP: ${result['target_price']:.2f} | "
                f"Size: ${result['position_size']['dollar_size']:,.0f}"
            )
    '''

    # ── PHASE H: Binance Connection Test ──────
    logger.info("\n" + "="*50)
    logger.info("PHASE H: Binance Broker Test")
    logger.info("="*50)

    

    broker = BinanceBroker(config)

    # Test 1: Get BTC price
    logger.info("\n   --- Test 1: Current Prices ---")
    btc_price = broker.get_price("BTCUSDT")
    eth_price = broker.get_price("ETHUSDT")
    logger.info(f"   BTC: ${btc_price:,.2f}")
    logger.info(f"   ETH: ${eth_price:,.2f}")

    # Test 2: Account balance
    logger.info("\n   --- Test 2: Account Balance ---")
    usdt_bal = broker.get_balance("USDT")
    btc_bal  = broker.get_balance("BTC")
    total    = broker.get_portfolio_value("BTCUSDT")

    # Test 3: Historical data
    logger.info("\n   --- Test 3: Historical BTC Data ---")
    candles = broker.get_klines("BTCUSDT", interval="1d", limit=5)
    logger.info(f"   Last 5 daily candles:")
    for i, c in enumerate(candles[-5:], 1):
        logger.info(
            f"   Day {i}: O:{c['open']:,.0f} "
            f"H:{c['high']:,.0f} "
            f"L:{c['low']:,.0f} "
            f"C:{c['close']:,.0f}"
        )

    # Test 4: Position check
    logger.info("\n   --- Test 4: Current Position ---")
    pos = broker.get_open_position("BTCUSDT")
    logger.info(
        f"   BTC held: {pos['quantity']} | "
        f"In position: {pos['in_position']}"
    )      
    
   # ── PHASE I: Live Trading Loop (Paper) ──────────────
    logger.info("\n" + "=" * 50)
    logger.info("PHASE I: Live Trading Loop — Paper Mode")
    logger.info("=" * 50)

    from live.loop import LiveTradingLoop

    paper_broker = PaperBroker(config)                        
    loop = LiveTradingLoop(config, paper_broker)              
    loop.risk.start_day(paper_broker.get_portfolio_value("BTCUSDT"))  
    result = loop.run_once()

    logger.info(f"\n📋 Cycle result: {result.get('action_name')} | "
            f"Approved: {result.get('approved')} | "
            f"Reason: {result.get('reason', 'ok')}")

    logger.info("\n" + "=" * 50)
    logger.info("PHASE J: Daily Scheduler")
    logger.info("=" * 50)

    from scheduler.daily import DailyScheduler

    scheduler = DailyScheduler(config, paper_broker)

    # Test: run one cycle immediately
    result = scheduler.run_once_now()
    # Phase J — run forever
    scheduler.run_forever() 
    '''

if __name__ == "__main__":
    main()
    