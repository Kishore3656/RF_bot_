import time
from datetime import datetime, timedelta
from loguru import logger

from live.loop       import LiveTradingLoop
from scheduler.state import load_state, update_state_after_cycle


class DailyScheduler:

    def __init__(self, config: dict, broker):
        self.config   = config
        self.broker   = broker
        self.symbol   = config.get("broker", {}).get("symbol", "BTCUSDT")
        self.run_hour = config.get("scheduler", {}).get("run_hour", 9)   # 9 AM default
        self.run_min  = config.get("scheduler", {}).get("run_minute", 0)
        self.loop     = LiveTradingLoop(config, broker)

        logger.info(f"⏰ DailyScheduler ready")
        logger.info(f"   Symbol:    {self.symbol}")
        logger.info(f"   Runs at:   {self.run_hour:02d}:{self.run_min:02d} daily")
        logger.info(f"   Broker:    {type(broker).__name__}")

    def _seconds_until_next_run(self) -> float:
        """Calculate seconds until next scheduled run time."""
        now  = datetime.now()
        next = now.replace(hour=self.run_hour, minute=self.run_min,
                           second=0, microsecond=0)
        if next <= now:
            next += timedelta(days=1)
        secs = (next - now).total_seconds()
        return secs

    def run_once_now(self):
        """Run one cycle immediately (for testing)."""
        logger.info("🔄 Running cycle NOW (manual trigger)")
        state = load_state()
        self.loop.risk.start_day(self.broker.get_portfolio_value(self.symbol))
        result = self.loop.run_once()
        state  = update_state_after_cycle(state, result, self.broker)
        self._print_summary(state, result)
        return result

    def run_forever(self):
        """Wait for scheduled time then run every 24 hours."""
        logger.info("🚀 DailyScheduler started — running forever")
        state = load_state()

        while True:
            secs = self._seconds_until_next_run()
            hrs  = int(secs // 3600)
            mins = int((secs % 3600) // 60)
            logger.info(f"⏳ Next run in {hrs}h {mins}m "
                        f"(at {self.run_hour:02d}:{self.run_min:02d})")
            time.sleep(secs)

            try:
                logger.info("=" * 50)
                logger.info(f"⏰ SCHEDULED RUN — {datetime.now():%Y-%m-%d %H:%M}")
                logger.info("=" * 50)

                self.loop.risk.start_day(
                    self.broker.get_portfolio_value(self.symbol)
                )
                result = self.loop.run_once()
                state  = update_state_after_cycle(state, result, self.broker)
                self._print_summary(state, result)

            except Exception as e:
                logger.error(f"❌ Cycle error: {e}")
                logger.info("⚠️  Bot will retry at next scheduled time")

    def _print_summary(self, state: dict, result: dict):
        wins   = state.get("wins", 0)
        losses = state.get("losses", 0)
        total  = wins + losses
        wr     = (wins / total * 100) if total > 0 else 0.0

        logger.info(f"\n{'='*50}")
        logger.info(f"📊 BOT SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"   Action:      {result.get('action_name', 'N/A')}")
        logger.info(f"   Approved:    {result.get('approved')}")
        logger.info(f"   USDT:        ${state['usdt']:,.2f}")
        logger.info(f"   BTC:          {state['btc']:.6f}")
        logger.info(f"   Total value: ${state['total_value']:,.2f}")
        logger.info(f"   Trades:      {state['trade_count']}")
        logger.info(f"   Win rate:    {wr:.1f}%")
        logger.info(f"   Last run:    {state['last_run']}")
        logger.info(f"{'='*50}")