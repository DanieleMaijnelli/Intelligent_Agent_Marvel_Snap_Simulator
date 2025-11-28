from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from gameManager import GameState
from environment.SnapAgentEnvironmentMarvelSnapSimulator import MarvelSnapSingleSnapAgentEnv

SEED = 43


def make_env():
    def _init():
        environment = MarvelSnapSingleSnapAgentEnv(GameState(verbose=False))
        return Monitor(environment)

    return _init


def train_dqn(total_timesteps=200_000):
    vectorized_environment = make_vec_env(make_env(), n_envs=4, seed=SEED)

    evaluation_environment = Monitor(MarvelSnapSingleSnapAgentEnv(GameState(verbose=False)))

    eval_callback = EvalCallback(
        evaluation_environment,
        best_model_save_path="./logs_dqn_snapper/",
        log_path="./logs_dqn_snapper/",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model = DQN(
        "MlpPolicy",
        vectorized_environment,
        verbose=0,
        tensorboard_log="./tensorboard_marvelsnap_dqn_snapper/",
        learning_rate=1e-4,
        buffer_size=200_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10_000,
        learning_starts=30_000,
        exploration_fraction=0.4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        seed=SEED,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(f"marvelsnap_snap_agent_dqn_{total_timesteps}")
    return model


if __name__ == "__main__":
    model = train_dqn(total_timesteps=3_500_000)
