# ai/inference.py

import numpy as np
import pandas as pd
from loguru import logger
from .agent import TradingAgent


class InferenceEngine:
    """
    Loads a trained model and runs it on new data.

    Single responsibility: given today's market data,
    return what action the bot would take.

    This is what Phase 5 (live trading) calls every day.
    It never trains — only predicts.
    """

    def __init__(self, config: dict):
        self.config          = config
        self.agent           = None
        self.initial_capital = config["broker"]["initial_capital"]
        logger.info("🔮 InferenceEngine ready")

    def load_model(self, filename: str = None) -> "InferenceEngine":
        """
        Load a previously trained model from disk.

        First principles: we recreate the agent with
        the same config, then load weights into it.
        The environment is needed so SB3 knows the
        observation/action space dimensions.
        """
        from .environment import TradingEnvironment
        from stable_baselines3.common.vec_env import DummyVecEnv

        # Create a dummy environment just for loading
        # (SB3 needs it to know input/output dimensions)
        # We pass an empty-ish df — it's never actually used
        logger.info("📂 Loading trained model...")

        self.agent = TradingAgent(self.config)

        # Build with None df placeholder — load() replaces weights
        # We need a minimal valid environment for SB3
        self.agent.load(filename)

        logger.success("✅ Model loaded and ready for inference")
        return self

    def predict(
        self,
        observation:     np.ndarray,
        portfolio_value: float = None,
    ) -> dict:
        """
        Given a normalized observation vector,
        return the action the bot would take.
        Skips low-confidence signals (returns HOLD)
        based on config ai.confidence_threshold.
        """
        if self.agent is None:
            raise RuntimeError(
                "❌ Call load_model() first"
            )

        portfolio_value    = portfolio_value or self.initial_capital
        min_confidence     = self.config.get("ai", {}).get("confidence_threshold", 55.0)

        result = self.agent.predict_with_info(
            observation     = observation,
            portfolio_value = portfolio_value,
            initial_capital = self.initial_capital,
        )

        # Override to HOLD if confidence is too low
        if result["confidence"] < min_confidence and result["action"] != 0:
            logger.warning(
                f"   ⚠️  Low confidence ({result['confidence']}% < {min_confidence}%) "
                f"— overriding {result['action_name']} → HOLD"
            )
            result["action"]      = 0
            result["action_name"] = "HOLD  🟡"
            result["certain"]     = False
            result["overridden"]  = True
        else:
            result["overridden"] = False

        return result

    def run_backtest(
        self,
        normalized_df: pd.DataFrame,
        raw_prices:    pd.Series,
    ) -> dict:
        """
        Run the loaded model on historical data.
        Like running run_episode() but from the
        inference side — no training, just evaluation.

        Returns performance summary.
        """
        if self.agent is None:
            raise RuntimeError(
                "❌ Call load_model() first"
            )

        summary = self.agent.run_episode(
            normalized_df = normalized_df,
            raw_prices    = raw_prices,
            render        = False,
        )

        logger.info("\n📊 Backtest Results (loaded model):")
        logger.info(
            f"   Return:    {summary['total_return']:+.2f}%"
        )
        logger.info(
            f"   Drawdown:  {summary['max_drawdown']:.2f}%"
        )
        logger.info(
            f"   Trades:    {summary['total_trades']}"
        )
        logger.info(
            f"   Win rate:  {summary['win_rate']:.1f}%"
        )

        return summary