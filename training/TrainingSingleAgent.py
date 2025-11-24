from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from gameManager import GameState
from environment.SingleAgentEnvironmentMarvelSnapSimulator import MarvelSnapSingleAgentEnv

SEED = 43


def make_env():
    def _init():
        environment = MarvelSnapSingleAgentEnv(GameState(verbose=False))
        return Monitor(environment)

    return _init


def train_dqn(total_timesteps=200_000):
    vectorized_environment = make_vec_env(make_env(), n_envs=4, seed=SEED)

    evaluation_environment = Monitor(MarvelSnapSingleAgentEnv(GameState(verbose=False)))

    eval_callback = EvalCallback(
        evaluation_environment,
        best_model_save_path="./logs_dqn/",
        log_path="./logs_dqn/",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model = DQN(
        "MlpPolicy",
        vectorized_environment,
        verbose=1,
        tensorboard_log="./tensorboard_marvelsnap_dqn/",
        learning_rate=1e-4,
        buffer_size=200_000,
        batch_size=64,
        gamma=0.9,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10_000,
        learning_starts=20_000,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        seed=SEED,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save("marvelsnap_single_agent_dqn_last")
    return model


def evaluate_model(model, n_episodes=5):
    environment = MarvelSnapSingleAgentEnv(GameState(verbose=True))
    for episode in range(n_episodes):
        observation, info = environment.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = environment.step(action)
            episode_reward += reward
            done = terminated or truncated
        print(f"Episode {episode + 1}: total reward = {episode_reward}")


'''if __name__ == "__main__":
    model = train_dqn(total_timesteps=2_000_000)
    evaluate_model(model, n_episodes=3)'''

# Se il modello esiste già usa questa versione
if __name__ == "__main__":
    model = DQN.load("marvelsnap_single_agent_dqn_last")
    evaluate_model(model, n_episodes=5)
