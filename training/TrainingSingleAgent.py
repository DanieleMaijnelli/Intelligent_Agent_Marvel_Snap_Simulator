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
    model.save(f"marvelsnap_single_agent_dqn_{total_timesteps}")
    return model


def evaluate_model(model, n_episodes=100):
    environment = MarvelSnapSingleAgentEnv(GameState(verbose=True))

    wins = 0
    losses = 0
    ties = 0

    for episode in range(n_episodes):
        observation, info = environment.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = environment.step(action)
            episode_reward += reward
            done = terminated or truncated
        winner = environment.game_state.passStatus["winner"]

        if winner == "Ally":
            wins += 1
        elif winner == "Enemy":
            losses += 1
        elif winner == "Tie":
            ties += 1

        print(f"Episode {episode + 1}: total reward = {episode_reward}, winner = {winner}")

    print("\n=== Evaluation summary ===")
    print(f"Episodes: {n_episodes}")
    print(f"Wins:     {wins}")
    print(f"Losses:   {losses}")
    print(f"Ties:     {ties}")

    if n_episodes > 0:
        win_rate_all = wins / n_episodes
        print(f"Win rate (including ties): {win_rate_all:.2%}")

    if (wins + losses) > 0:
        win_rate_no_ties = wins / (wins + losses)
        print(f"Win rate (excluding ties): {win_rate_no_ties:.2%}")
    else:
        print("No decisive games (only ties).")


if __name__ == "__main__":
    model = train_dqn(total_timesteps=4_100_000)
    evaluate_model(model, n_episodes=10000)

# Se il modello esiste già usa questa versione
'''if __name__ == "__main__":
    model = DQN.load("marvelsnap_single_agent_dqn_5000000")
    evaluate_model(model, n_episodes=10000)'''
