# ai/trainer.py

import os
import shutil
import pandas as pd
import numpy as np
from loguru import logger
from .agent import TradingAgent


class AgentTrainer:
    """
    Manages the complete training workflow.

    Single responsibility: orchestrate the
    train → evaluate → save pipeline cleanly.

    Now supports:
    - Multi-run training: trains N times, keeps best win rate
    - Resume training: loads existing model and continues
      training instead of starting from scratch
    """

    def __init__(self, config: dict):
        self.config = config
        logger.info("🎓 AgentTrainer ready")

    def train_and_evaluate(
        self,
        normalized_df: pd.DataFrame,
        raw_prices:    pd.Series,
        timesteps:     int = None,
        resume:        bool = None,   # None = read from config
    ) -> dict:
        """
        Full pipeline with multi-run + resume support.

        resume=True  → load saved model and continue training
        resume=False → always train from scratch
        resume=None  → reads ai.resume_training from config
        """

        # ── Step 1: Split data ─────────────────────
        split     = self.config["data"]["train_split"]
        split_idx = int(len(normalized_df) * split)

        train_df     = normalized_df.iloc[:split_idx].copy()
        test_df      = normalized_df.iloc[split_idx:].copy()
        train_prices = raw_prices.iloc[:split_idx].copy()
        test_prices  = raw_prices.iloc[split_idx:].copy()

        logger.info(
            f"\n📊 Data split:"
            f"\n   Train: {len(train_df)} rows "
            f"(${train_prices.min():.2f}–${train_prices.max():.2f})"
            f"\n   Test:  {len(test_df)} rows "
            f"(${test_prices.min():.2f}–${test_prices.max():.2f})"
        )

        steps        = timesteps or self.config["ai"]["training_timesteps"]
        n_runs       = self.config["ai"].get("training_runs", 1)
        save_path    = self.config["ai"].get("model_save_path", "models/")
        best_path    = os.path.join(save_path, "PPO_trading_bot.zip")
        temp_path    = os.path.join(save_path, "PPO_temp_run.zip")

        # Resolve resume flag
        if resume is None:
            resume = self.config["ai"].get("resume_training", False)

        # Check if a saved model actually exists to resume from
        can_resume = resume and os.path.exists(best_path)

        if resume and can_resume:
            logger.info(f"\n⏩ RESUME MODE — continuing from: {best_path}")
        elif resume and not can_resume:
            logger.warning("⚠️  Resume requested but no saved model found — training from scratch")
        else:
            logger.info(f"\n🔁 FRESH training: {n_runs} runs | keeping best win rate")

        logger.info("="*55)

        best_win_rate  = -1.0
        best_after     = None
        best_before    = None
        run_results    = []

        for run in range(1, n_runs + 1):
            logger.info(f"\n🏃 Run {run}/{n_runs}")
            logger.info("-"*40)

            # ── Build agent ────────────────────────
            agent = TradingAgent(self.config)
            agent.build(
                normalized_df = train_df,
                raw_prices    = train_prices,
            )

            # ── Resume: load existing weights ──────
            if can_resume:
                logger.info(f"   📂 Loading saved model to resume...")
                agent.load("PPO_trading_bot")
                agent.model.set_env(agent.env)   # reattach environment
                logger.success(f"   ✅ Resuming from saved checkpoint")

            # ── Baseline before training (run 1 only) ──
            if run == 1:
                logger.info("📊 Evaluating BEFORE training...")
                before = agent.run_episode(test_df, test_prices)
                logger.info(f"   Return:   {before['total_return']:+.2f}%")
                logger.info(f"   Drawdown: {before['max_drawdown']:.2f}%")
                logger.info(f"   Trades:   {before['total_trades']}")
                logger.info(f"   Win rate: {before['win_rate']:.1f}%")
                best_before = before

            # ── Train ──────────────────────────────
            agent.train(total_timesteps=steps)

            # ── Evaluate ───────────────────────────
            after    = agent.run_episode(test_df, test_prices)
            win_rate = after["win_rate"]

            logger.info(
                f"   ✅ Run {run} result | "
                f"Win rate: {win_rate:.1f}% | "
                f"Return: {after['total_return']:+.2f}% | "
                f"Drawdown: {after['max_drawdown']:.2f}% | "
                f"Trades: {after['total_trades']}"
            )

            run_results.append({
                "run":       run,
                "win_rate":  win_rate,
                "return":    after["total_return"],
                "drawdown":  after["max_drawdown"],
                "trades":    after["total_trades"],
            })

            # ── Save best model ────────────────────
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_after    = after

                agent.model.save(temp_path.replace(".zip", ""))
                if os.path.exists(temp_path):
                    shutil.copy(temp_path, best_path)
                else:
                    src = temp_path.replace(".zip", "") + ".zip"
                    if os.path.exists(src):
                        shutil.copy(src, best_path)

                logger.info(f"   💾 New best model saved! Win rate: {best_win_rate:.1f}%")

        # ── Print summaries ────────────────────────
        self._print_run_summary(run_results)
        report = self._build_report(best_before, best_after)
        self._print_report(report)

        logger.info(
            f"\n🏆 Best run win rate: {best_win_rate:.1f}% "
            f"| Model saved to: {best_path}"
        )

        return {
            "agent":         None,
            "before":        best_before,
            "after":         best_after,
            "report":        report,
            "run_results":   run_results,
            "best_win_rate": best_win_rate,
        }

    def _print_run_summary(self, run_results: list):
        logger.info("\n" + "="*65)
        logger.info("📊 MULTI-RUN SUMMARY")
        logger.info("="*65)
        logger.info(
            f"{'Run':>4} | {'Win Rate':>9} | {'Return':>10} | "
            f"{'Drawdown':>9} | {'Trades':>7}"
        )
        logger.info("-"*65)

        best_wr = max(r["win_rate"] for r in run_results)

        for r in run_results:
            flag = " 🏆" if r["win_rate"] == best_wr else ""
            logger.info(
                f"{r['run']:>4} | "
                f"{r['win_rate']:>8.1f}% | "
                f"{r['return']:>+9.2f}% | "
                f"{r['drawdown']:>8.2f}% | "
                f"{r['trades']:>7}"
                f"{flag}"
            )

        logger.info("="*65)

    def _build_report(self, before: dict, after: dict) -> dict:
        def improvement(b, a, higher_is_better=True):
            diff = a - b
            direction = "✅" if (diff > 0) == higher_is_better else "❌"
            return diff, direction

        return_diff,   return_flag   = improvement(before["total_return"],  after["total_return"])
        drawdown_diff, drawdown_flag = improvement(before["max_drawdown"],  after["max_drawdown"], higher_is_better=False)
        winrate_diff,  winrate_flag  = improvement(before["win_rate"],      after["win_rate"])

        return {
            "return_before":   before["total_return"],
            "return_after":    after["total_return"],
            "return_diff":     return_diff,
            "return_flag":     return_flag,
            "drawdown_before": before["max_drawdown"],
            "drawdown_after":  after["max_drawdown"],
            "drawdown_diff":   drawdown_diff,
            "drawdown_flag":   drawdown_flag,
            "winrate_before":  before["win_rate"],
            "winrate_after":   after["win_rate"],
            "winrate_diff":    winrate_diff,
            "winrate_flag":    winrate_flag,
            "trades_before":   before["total_trades"],
            "trades_after":    after["total_trades"],
        }

    def _print_report(self, report: dict):
        logger.info("\n" + "="*55)
        logger.info("📋 TRAINING REPORT — BEFORE vs BEST RUN")
        logger.info("="*55)
        logger.info(
            f"{'Metric':<20} {'Before':>10} "
            f"{'After':>10} {'Change':>10}"
        )
        logger.info("-"*55)
        logger.info(
            f"{'Total Return':<20} "
            f"{report['return_before']:>9.2f}% "
            f"{report['return_after']:>9.2f}% "
            f"{report['return_diff']:>+9.2f}% "
            f"{report['return_flag']}"
        )
        logger.info(
            f"{'Max Drawdown':<20} "
            f"{report['drawdown_before']:>9.2f}% "
            f"{report['drawdown_after']:>9.2f}% "
            f"{report['drawdown_diff']:>+9.2f}% "
            f"{report['drawdown_flag']}"
        )
        logger.info(
            f"{'Win Rate':<20} "
            f"{report['winrate_before']:>9.1f}% "
            f"{report['winrate_after']:>9.1f}% "
            f"{report['winrate_diff']:>+9.1f}% "
            f"{report['winrate_flag']}"
        )
        logger.info(
            f"{'Total Trades':<20} "
            f"{report['trades_before']:>10} "
            f"{report['trades_after']:>10}"
        )
        logger.info("="*55)