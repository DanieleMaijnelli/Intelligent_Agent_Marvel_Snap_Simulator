import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from environment.EnvironmentMarvelSnapSimulator import TestEnvironmentMarvelSnapSimulator
from ray import tune
from pathlib import Path


def print_metrics(results):
    envr = results["env_runners"]
    learner_p1 = results["info"]["learner"]["player_1"]["learner_stats"]
    learner_p2 = results["info"]["learner"]["player_2"]["learner_stats"]

    rew_p1 = envr["policy_reward_mean"]["player_1"]
    rew_p2 = envr["policy_reward_mean"]["player_2"]
    ep_len = envr["episode_len_mean"]

    vfvar_p1 = learner_p1["vf_explained_var"]
    vfvar_p2 = learner_p2["vf_explained_var"]

    ent_p1 = learner_p1["entropy"]
    ent_p2 = learner_p2["entropy"]

    kl_p1 = learner_p1["kl"]
    kl_p2 = learner_p2["kl"]

    print(
        f"P1_return={rew_p1:+.3f}  P2_return={rew_p2:+.3f}  "
        f"ep_len={ep_len:.1f} | "
        f"vf_var P1={vfvar_p1:.2f} P2={vfvar_p2:.2f} | "
        f"entropy P1={ent_p1:.2f} P2={ent_p2:.2f} | "
        f"kl P1={kl_p1:.3f} P2={kl_p2:.3f}"
    )


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
for i in range(21):
    results = algo.train()
    print(results)

    if i % 5 == 0:
        checkpoint_path = algo.save(checkpoint_dir=str(checkpoint_dir))
        print(f"✅ Checkpoint salvato in: {checkpoint_path}")

print_metrics(results)

ray.shutdown()
