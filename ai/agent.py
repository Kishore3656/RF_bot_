# ai/agent.py

import os
import numpy as np
import pandas as pd
from loguru import logger
from stable_baselines3 import PPO, A2C, DDPG, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
    BaseCallback
)
from .environment import TradingEnvironment


class TradingAgent:
    """
    Wraps Stable Baselines3 algorithms with our
    TradingEnvironment into one clean interface.

    Single responsibility: manage the RL agent lifecycle.
    - Create the agent with the right settings
    - Train it on market data
    - Save and load model weights
    - Run inference (make predictions)
    """

    # Supported algorithms and their classes
    ALGORITHMS = {
        "PPO":  PPO,
        "A2C":  A2C,
        "DDPG": DDPG,
        "TD3":  TD3,
    }

    def __init__(self, config: dict):
        """
        Set up the agent from config.yaml settings.
        Does NOT create the model yet — that happens
        in build() so we can inspect the environment first.
        """
        self.config     = config
        ai_cfg          = config["ai"]

        self.algorithm  = ai_cfg["algorithm"]        # "PPO"
        self.timesteps  = ai_cfg["training_timesteps"] # 100000
        self.lr         = ai_cfg["learning_rate"]    # 0.0003
        self.save_path  = ai_cfg["model_save_path"]  # "models/"

        self.model      = None   # created in build()
        self.env        = None   # set in build()

        # Create models directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)

        logger.info(f"🤖 TradingAgent initialized")
        logger.info(f"   Algorithm:  {self.algorithm}")
        logger.info(f"   Timesteps:  {self.timesteps:,}")
        logger.info(f"   Learn rate: {self.lr}")
        logger.info(f"   Save path:  {self.save_path}")

    def build(
        self,
        normalized_df: pd.DataFrame,
        raw_prices:    pd.Series,
    ):
        """
        Create the environment and RL model.

        Why separate from __init__?
        Because we need the data BEFORE we can
        build the environment and model.
        Data comes from the pipeline, not config.

        First principles: separate CONFIGURATION
        (what algorithm, what settings) from
        CONSTRUCTION (actually building the thing).
        """
        logger.info(f"🔨 Building {self.algorithm} agent...")

        # ── Step 1: Wrap environment for Stable Baselines3
        # DummyVecEnv is a "vectorized" wrapper
        # SB3 was designed to train on multiple environments
        # in parallel. DummyVecEnv runs just one — but
        # satisfies SB3's requirement for the wrapper.
        max_pos = self.config.get("ai", {}).get("max_position_pct", 0.30)
        self.env = DummyVecEnv([
            lambda: TradingEnvironment(
                df              = normalized_df,
                config          = self.config,
                raw_prices      = raw_prices,
                initial_capital = self.config["broker"]["initial_capital"],
                max_position    = max_pos,
            )
        ])

        # ── Step 2: Define the neural network architecture
        # [128, 128]: two hidden layers with 128 neurons each.
        # Larger than [64, 64] — better capacity for 35 input features.
        policy_kwargs = dict(
            net_arch = [128, 128]
        )

        # ── Step 3: Read seed from config ──────────
        seed = self.config.get("ai", {}).get("random_seed", None)

        # ── Step 4: Create the model ───────────────
        AlgorithmClass = self.ALGORITHMS[self.algorithm]

        # n_steps=2048: collect 2048 steps before each update.
        # Larger buffer → lower-variance gradient estimates → more stable learning.
        # batch_size=64, n_epochs=10: standard PPO best practices.
        # seed=seed: fixed seed makes weight init and batch shuffling reproducible.
        self.model = AlgorithmClass(
            policy          = "MlpPolicy",
            env             = self.env,
            learning_rate   = self.lr,
            policy_kwargs   = policy_kwargs,
            verbose         = 0,
            device          = "cpu",   # MlpPolicy runs faster on CPU than GPU
            tensorboard_log = "logs/tensorboard/",
            seed            = seed,
            **({
                "ent_coef":   0.01,   # exploration bonus (PPO/A2C only)
                "n_steps":    2048,   # larger rollout buffer = more stable updates
                "batch_size": 64,
                "n_epochs":   10,
            } if self.algorithm in ("PPO", "A2C") else {}),
        )

        # Count trainable parameters
        total_params = sum(
            p.numel()
            for p in self.model.policy.parameters()
        )

        logger.success(
            f"✅ {self.algorithm} agent built | "
            f"Neural net: 29→256→256→3 | "
            f"Parameters: {total_params:,}"
        )

        return self

    def train(self, total_timesteps: int = None):
        """
        Train the agent on the environment.

        total_timesteps: how many steps to train for.
        More steps = smarter agent (up to a point).
        Defaults to config value if not specified.

        First principles: training = running the
        environment loop thousands of times and
        updating the neural network after each batch.
        """
        if self.model is None:
            raise RuntimeError(
                "❌ Call build() before train()"
            )

        steps = total_timesteps or self.timesteps

        logger.info(
            f"🏋️ Training {self.algorithm} for "
            f"{steps:,} timesteps..."
        )
        logger.info(
            "   This will take a few minutes. "
            "Progress logged every 1,000 steps."
        )

        # Custom callback to log progress
        callback = TrainingLoggerCallback(log_every=1000)

        self.model.learn(
            total_timesteps = steps,
            callback        = callback,
            progress_bar    = False,
        )

        logger.success(
            f"✅ Training complete! "
            f"{steps:,} timesteps finished."
        )

        return self

    def save(self, filename: str = None):
        """
        Save model weights to disk.

        Why save? Neural network weights = the bot's
        entire learned knowledge. Saving preserves
        everything it learned so we don't have to
        retrain from scratch every time.
        """
        if self.model is None:
            raise RuntimeError("❌ No model to save")

        filename = filename or f"{self.algorithm}_trading_bot"
        filepath = os.path.join(self.save_path, filename)

        self.model.save(filepath)

        logger.success(f"💾 Model saved: {filepath}.zip")
        return filepath

    def load(self, filename: str = None):
        """
        Load previously saved model weights.

        Like opening a saved game — all the learning
        is restored instantly without retraining.
        """
        filename = filename or f"{self.algorithm}_trading_bot"
        filepath = os.path.join(self.save_path, filename)

        if not os.path.exists(filepath + ".zip"):
            raise FileNotFoundError(
                f"❌ No saved model at {filepath}.zip"
            )

        AlgorithmClass = self.ALGORITHMS[self.algorithm]
        self.model     = AlgorithmClass.load(
            filepath, env=self.env
        )

        logger.success(f"📂 Model loaded: {filepath}.zip")
        logger.info(f"   (Timesteps shown above reflect config default — actual trained steps may differ)")
        return self

    def predict(self, observation: np.ndarray) -> int:
        """
        Given a state, return the best action.
        This is INFERENCE — using the trained model
        without any learning happening.

        Used during live trading (Phase 5).
        """
        if self.model is None:
            raise RuntimeError(
                "❌ No model. Call build() + train() first."
            )

        action, _ = self.model.predict(
            observation,
            deterministic = True   # always pick best action
                                   # not random exploration
        )
        return int(action)


        # Add these methods to TradingAgent class
    # (after the existing predict() method)

    ACTION_NAMES = {
        0: "HOLD  🟡",
        1: "LONG  📈",
        2: "SHORT 📉",
        3: "CLOSE 🔴",
    }

    def get_action_name(self, action: int) -> str:
        """Convert action number to readable string."""
        return self.ACTION_NAMES.get(action, f"UNKNOWN({action})")

    def predict_with_info(
        self,
        observation: np.ndarray,
        portfolio_value: float,
        initial_capital: float,
    ) -> dict:
        """
        Full inference with human-readable output.
        Used in live trading (Phase 5).

        Returns dict with action, name, confidence,
        and whether the agent is certain or uncertain.
        """
        if self.model is None:
            raise RuntimeError("❌ No model loaded.")

        # Get action probabilities from the policy network
        # This tells us HOW confident the agent is
        import torch
        obs_tensor = torch.FloatTensor(observation.copy()).unsqueeze(0)

        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.squeeze().numpy()

        action     = int(probs.argmax())
        confidence = float(probs.max())
        pnl_pct    = (portfolio_value - initial_capital) / initial_capital * 100

        return {
            "action":      action,
            "action_name": self.get_action_name(action),
            "confidence":  round(confidence * 100, 1),
            "probs": {
                self.get_action_name(i): round(float(p) * 100, 1)
                for i, p in enumerate(probs)
            },
            "portfolio_pnl": round(pnl_pct, 2),
            "certain":  confidence > 0.7,   # >70% = agent is sure
        }

    def run_episode(
        self,
        normalized_df: pd.DataFrame,
        raw_prices:    pd.Series,
        render:        bool = False,
        use_risk:      bool = True,
    ) -> dict:
        """
        Run one complete episode with the TRAINED agent.
        Used for evaluation after training.

        use_risk=True applies StopLossManager to enforce
        SL/TP on every step (mirrors live paper trading).

        Returns performance summary.
        """
        if self.model is None:
            raise RuntimeError(
                "❌ No model. Train or load first."
            )

        from risk.stop_loss import StopLossManager

        max_pos = self.config.get("ai", {}).get("max_position_pct", 0.30)

        # Create fresh environment for evaluation
        eval_env = TradingEnvironment(
            df              = normalized_df,
            config          = self.config,
            raw_prices      = raw_prices,
            initial_capital = self.config["broker"]["initial_capital"],
            max_position    = max_pos,
        )

        obs, _       = eval_env.reset()
        total_reward = 0
        steps        = 0
        sl           = StopLossManager(self.config) if use_risk else None

        while True:
            action = self.predict(obs)

            # ── Risk enforcement: SL/TP override ──────
            if use_risk and sl:
                current_price = float(eval_env.raw_prices[eval_env.current_step])

                if eval_env.position != 0:
                    sl_result = sl.check(current_price)
                    if sl_result["should_exit"]:
                        action = 3   # force close

                # Track open/close for SL manager state
                if action in (1, 2) and eval_env.position == 0:
                    direction = "long" if action == 1 else "short"
                    sl.open_trade(current_price, direction)
                elif action == 3 and eval_env.position != 0:
                    sl.close_trade()
            # ──────────────────────────────────────────

            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            steps        += 1

            if render:
                eval_env.render()

            if terminated or truncated:
                break

        summary = eval_env.get_performance_summary()
        summary["total_reward"] = round(total_reward, 6)
        summary["steps"]        = steps

        return summary


class TrainingLoggerCallback(BaseCallback):
    """
    Custom callback that logs training progress
    every N timesteps.

    First principles: during training the agent
    runs thousands of steps silently. Without this
    callback we'd see nothing. With it, we get
    regular updates on how learning is progressing.
    """

    def __init__(self, log_every: int = 1000):
        super().__init__()
        self.log_every   = log_every
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        """
        Called after every single environment step.
        Return True to continue training.
        Return False to stop training early.
        """
        # Accumulate reward for current episode
        reward = self.locals.get("rewards", [0])[0]
        self.current_episode_reward += reward

        # Check if episode ended
        done = self.locals.get("dones", [False])[0]
        if done:
            self.episode_rewards.append(
                self.current_episode_reward
            )
            self.current_episode_reward = 0

        # Log every N steps
        if self.num_timesteps % self.log_every == 0:
            if self.episode_rewards:
                recent = self.episode_rewards[-10:]
                avg_reward = np.mean(recent)
                logger.info(
                    f"   Step {self.num_timesteps:6,} | "
                    f"Episodes: {len(self.episode_rewards):4d} | "
                    f"Avg reward (last 10): {avg_reward:+.6f}"
                )

        return True   # continue training