import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from environment.EnvironmentMarvelSnapSimulator import TestEnvironmentMarvelSnapSimulator
from ray import tune

tune.register_env(
    "marvel_snap_env",
    lambda cfg: ParallelPettingZooEnv(TestEnvironmentMarvelSnapSimulator())
)

probe_env = TestEnvironmentMarvelSnapSimulator()
obs_space_p1 = probe_env.observation_space("player_1")
act_space_p1 = probe_env.action_space("player_1")
obs_space_p2 = probe_env.observation_space("player_2")
act_space_p2 = probe_env.action_space("player_2")

ray.init(ignore_reinit_error=True)

config = (
    PPOConfig()
    .environment(env="marvel_snap_env")
    .framework("torch")
    .env_runners(num_env_runners=0)
    .training(gamma=0.99, lr=3e-4, entropy_coeff=0.01)
    .resources(num_gpus=0)
    .multi_agent(
        policies={
            "player_1": (None, obs_space_p1, act_space_p1, {}),
            "player_2": (None, obs_space_p2, act_space_p2, {}),
        },
        policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
    )
)

algo = config.build()

for i in range(1000):
    results = algo.train()
    print(
        f"Iter {i}: reward mean = {results['env_runners'].get('episode_return_mean', results['env_runners'].get('episode_reward_mean'))}")
    if i % 50 == 0:
        checkpoint = algo.save()
        print(f"✅ Checkpoint salvato: {checkpoint}")

ray.shutdown()
