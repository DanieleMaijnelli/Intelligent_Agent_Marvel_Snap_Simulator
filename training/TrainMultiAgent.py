import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from environment.EnvironmentMarvelSnapSimulator import TestEnvironmentMarvelSnapSimulator
from ray import tune
from pathlib import Path
from StatisticsPrinter import print_multiagent_training_summary

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

checkpoint_dir = Path("checkpoints").resolve()
checkpoint_dir.mkdir(parents=True, exist_ok=True)

config = (
    PPOConfig()
    .environment(env="marvel_snap_env")
    .framework("torch")
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
    .env_runners(num_env_runners=1)
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

results = None
for i in range(31):
    results = algo.train()

    if i % 10 == 0:
        checkpoint_path = algo.save(checkpoint_dir=str(checkpoint_dir))
        print(f"✅ Checkpoint salvato in: {checkpoint_path}")

print_multiagent_training_summary(results)

ray.shutdown()
