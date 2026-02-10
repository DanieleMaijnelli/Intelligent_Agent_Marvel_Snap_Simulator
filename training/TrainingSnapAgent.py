import torch.nn as nn
import torch.optim as optim
import time
from training.TrainingUtilityFunctions import *
from environment.SnapAgentEnvironment import SnapAgentEnvironment
from training.TrainingNetwork import QNetworkB, load_q_network, save_q_network


def generate_episode_snap_agent(environment, snap_q_network, epsilon):
    environment.reset()
    episode_ally_state_actions = []
    ally_reward_list = []

    done = False

    while not done:
        action, state_action_vector = choose_snap_action_epsilon_greedy(
            environment, snap_q_network, True, epsilon
        )
        done = environment.step(action)
        episode_ally_state_actions.append(state_action_vector)
        ally_reward_list.append(0.0)

        if done:
            winner = environment.game_state.passStatus["winner"]
            if winner == "Ally":
                final_reward_ally = float(environment.game_state.status["cubes"])
            elif winner == "Enemy":
                final_reward_ally = -float(environment.game_state.status["cubes"])
            else:
                final_reward_ally = 0.0

            if len(ally_reward_list) > 0:
                ally_reward_list[len(ally_reward_list) - 1] += final_reward_ally

    ally_return_list = compute_monte_carlo_returns(ally_reward_list, discount_factor=0.9)

    return (
        episode_ally_state_actions,
        ally_return_list
    )


def train_snap_agent_deep_monte_carlo_with_logging(
    number_of_episodes,
    learning_rate=5e-5,
    epsilon_start=0.99,
    epsilon_end=0.01,
    seed_value=None,
    evaluation_interval=100,
    evaluation_games=50,
    decay_fraction=0.5,
    log_csv_path=None,
    save_model_path=None,
):
    if seed_value is not None:
        set_global_seed(seed_value)
    start_time_seconds = time.time()

    input_dimension = get_input_dimension()
    player_q_network = load_q_network(
        f"trained_q_network_DMC_2000000_episodes.pt",
        input_dimension=input_dimension,
        architecture="B",
    )

    snap_input_dimension = get_snap_input_dimension()
    snap_q_network = QNetworkB(snap_input_dimension)
    optimizer = optim.Adam(snap_q_network.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    environment = SnapAgentEnvironment(player_q_network)

    csv_file = None
    csv_writer = None
    if log_csv_path is not None:
        csv_file, csv_writer = create_snap_training_csv_writer(log_csv_path)
        csv_file.flush()

    episode_index = 0
    decay_episodes = int(number_of_episodes * decay_fraction)

    while episode_index < number_of_episodes:
        if episode_index < decay_episodes:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * (
                1.0 - float(episode_index) / float(decay_episodes - 1)
            )
        else:
            epsilon = epsilon_end

        snap_q_network.eval()
        (
            episode_ally_state_actions,
            ally_return_list
        ) = generate_episode_snap_agent(environment, snap_q_network, epsilon)
        snap_q_network.train()

        training_state_action_list = []
        target_return_list = []

        for state_action_vector, target_return in zip(episode_ally_state_actions, ally_return_list):
            training_state_action_list.append(state_action_vector)
            target_return_list.append(target_return)

        if len(training_state_action_list) > 0:
            state_action_array = numpy.stack(training_state_action_list).astype(
                numpy.float32
            )
            target_array = numpy.array(target_return_list, dtype=numpy.float32)

            state_action_tensor = torch.from_numpy(state_action_array)
            target_tensor = torch.from_numpy(target_array)

            predicted_q_tensor = snap_q_network(state_action_tensor)
            loss_value = loss_function(predicted_q_tensor, target_tensor)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        if evaluation_interval is not None and evaluation_interval > 0:
            if (episode_index + 1) % evaluation_interval == 0:
                snap_q_network.eval()
                eval_results = evaluate_snap_agent(
                    player_q_network, snap_q_network, evaluation_games
                )
                snap_q_network.train()
                elapsed_minutes = (time.time() - start_time_seconds) / 60.0
                ally_win_rate = eval_results["ally_win_rate"]
                enemy_win_rate = eval_results["enemy_win_rate"]
                tie_rate = eval_results["tie_rate"]
                average_cubes_won = eval_results["average_cubes_won"]

                if csv_writer is not None:
                    write_snap_training_csv_row(
                        csv_writer,
                        episode_index + 1,
                        elapsed_minutes,
                        epsilon,
                        ally_win_rate,
                        enemy_win_rate,
                        tie_rate,
                        average_cubes_won
                    )
                    csv_file.flush()

        episode_index += 1

    if csv_file is not None:
        csv_file.close()

    if save_model_path is not None:
        save_q_network(snap_q_network, save_model_path)

    return snap_q_network


if __name__ == "__main__":
    number_of_episodes = 100000
    network = train_snap_agent_deep_monte_carlo_with_logging(
        number_of_episodes=number_of_episodes,
        learning_rate=5e-5,
        epsilon_start=0.99,
        epsilon_end=0.01,
        seed_value=45,
        evaluation_interval=5000,
        evaluation_games=1000,
        decay_fraction=0.5,
        log_csv_path=f"training_log_snap_DMC_{number_of_episodes}_episodes.csv",
        save_model_path=f"trained_q_network_snap_DMC_{number_of_episodes}_episodes.pt",
    )
    print("Training finished.")

    snap_input_dimension = get_snap_input_dimension()
    input_dimension = get_input_dimension()

    player_q_network = load_q_network(
        f"trained_q_network_DMC_2000000_episodes.pt",
        input_dimension=input_dimension,
        architecture="B",
    )

    snap_q_network = load_q_network(
        f"trained_q_network_snap_DMC_{number_of_episodes}_episodes.pt",
        input_dimension=snap_input_dimension,
        architecture="B",
    )

    player_q_network.eval()
    snap_q_network.eval()
    final_eval_results = evaluate_snap_agent(
        player_q_network,
        snap_q_network,
        number_of_games=1000
    )

    print("Final evaluation:")
    for key, value in final_eval_results.items():
        print(f"{key}: {format_csv_value(value)}")
