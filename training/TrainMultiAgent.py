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

results = None
for i in range(21):
    results = algo.train()
    print(results)

    if i % 5 == 0:
        checkpoint_path = algo.save(checkpoint_dir="checkpoints")
        print(f"✅ Checkpoint salvato in: {checkpoint_path}")


r_p1 = results["env_runners"]["agent_episode_returns_mean"]["player_1"]
r_p2 = results["env_runners"]["agent_episode_returns_mean"]["player_2"]
print(f"reward_mean P1={r_p1:.3f} P2={r_p2:.3f}")

env_stats = results["env_runners"]
agent_returns = env_stats["agent_episode_returns_mean"]

r_p1 = agent_returns["player_1"]
r_p2 = agent_returns["player_2"]
len_mean = env_stats["episode_len_mean"]

print(f"P1 mean return = {r_p1:.3f} | P2 = {r_p2:.3f} | ep_len = {len_mean:.1f}")

learner_p1 = results["learners"]["player_1"]
entropy_p1 = learner_p1["entropy"]
vf_var_p1 = learner_p1["vf_explained_var"]

print(f"    P1: entropy={entropy_p1:.2f}, vf_explained_var={vf_var_p1:.2f}")


ray.shutdown()
