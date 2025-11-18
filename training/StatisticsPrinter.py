def print_multiagent_training_summary(result: dict):
    it          = result.get("training_iteration")
    date        = result.get("date")
    env_steps   = result.get("num_env_steps_sampled")
    agent_steps = result.get("num_agent_steps_sampled")
    time_total  = result.get("time_total_s")

    print("\n" + "=" * 60)
    print(f"  TRAINING ITERATION {it}")
    print("=" * 60)
    print(f"Data:                     {date}")
    print(f"Env steps totali:         {env_steps}")
    print(f"Agent steps totali:       {agent_steps}")
    print(f"Tempo totale (s):         {time_total:.1f}" if isinstance(time_total, (int, float)) else f"Tempo totale: {time_total}")

    env_runners = result.get("env_runners", {})
    ep_rew_mean = env_runners.get("episode_reward_mean")
    ep_rew_min  = env_runners.get("episode_reward_min")
    ep_rew_max  = env_runners.get("episode_reward_max")
    ep_len_mean = env_runners.get("episode_len_mean")
    num_eps     = env_runners.get("num_episodes")

    pol_rew_mean = env_runners.get("policy_reward_mean", {})

    print("\n[EPISODI]")
    print(f"Num episodi totali (lifetime): {num_eps}")
    print(f"Reward episodio media:         {float(ep_rew_mean):+.3f}" if ep_rew_mean is not None else "Reward episodio media: N/A")
    print(f"Reward episodio min / max:     {float(ep_rew_min):+.3f} / {float(ep_rew_max):+.3f}"
          if ep_rew_min is not None and ep_rew_max is not None else "Reward episodio min / max: N/A")
    print(f"Lunghezza episodio media:      {float(ep_len_mean):.2f}" if ep_len_mean is not None else "Lunghezza episodio media: N/A")

    if pol_rew_mean:
        print("\n[REWARD MEDI PER AGENTE]")
        for policy_id, rew in pol_rew_mean.items():
            print(f"  {policy_id}: {float(rew):+.3f}")

    learners = result.get("info", {}).get("learner", {})

    def _print_learner_stats(player_key: str, label: str):
        data = learners.get(player_key)
        if not data:
            print(f"\n[LEARNER {label}] Nessun dato.")
            return

        stats = data.get("learner_stats", {})
        total_loss   = stats.get("total_loss")
        policy_loss  = stats.get("policy_loss")
        vf_loss      = stats.get("vf_loss")
        vf_ev        = stats.get("vf_explained_var")
        entropy      = stats.get("entropy")
        kl           = stats.get("kl")
        cur_lr       = stats.get("cur_lr")
        grad_gnorm   = stats.get("grad_gnorm")

        num_steps_tr = data.get("num_agent_steps_trained")
        num_updates  = data.get("num_grad_updates_lifetime")

        print(f"\n[LEARNER {label} - {player_key}]")
        print(f"  Steps allenati (policy):    {num_steps_tr}")
        print(f"  Grad updates totali:        {num_updates}")
        print(f"  Learning rate attuale:      {cur_lr}")
        print(f"  Total loss:                 {float(total_loss):.4f}"   if total_loss   is not None else "  Total loss: N/A")
        print(f"  Policy loss:                {float(policy_loss):.4f}"  if policy_loss  is not None else "  Policy loss: N/A")
        print(f"  Value loss:                 {float(vf_loss):.4f}"      if vf_loss      is not None else "  Value loss: N/A")
        print(f"  VF explained var:           {float(vf_ev):+.3f}"       if vf_ev        is not None else "  VF explained var: N/A")
        print(f"  Entropy:                    {float(entropy):.3f}"      if entropy      is not None else "  Entropy: N/A")
        print(f"  KL divergence:              {float(kl):.5f}"           if kl           is not None else "  KL divergence: N/A")
        print(f"  Grad global norm:           {float(grad_gnorm):.3f}"   if grad_gnorm   is not None else "  Grad global norm: N/A")

    _print_learner_stats("player_1", "PLAYER 1")
    _print_learner_stats("player_2", "PLAYER 2")

    env_sps = result.get("num_env_steps_sampled_throughput_per_sec")
    timers  = result.get("timers", {})
    learn_throughput = timers.get("learn_throughput")

    print("\n[PERFORMANCE]")
    print(f"Env steps/s (sample):         {float(env_sps):.1f}" if env_sps is not None else "Env steps/s (sample): N/A")
    print(f"Throughput learning (steps/s):{float(learn_throughput):.1f}" if learn_throughput is not None else "Throughput learning: N/A")
    print("=" * 60 + "\n")