import torch.nn as nn
import torch.optim as optim
import time
from training.TrainingUtilityFunctions import *
from training.TrainingNetwork import QNetwork, QNetworkA, QNetworkB, load_q_network, save_q_network
import multiprocessing
import io


def serialize_state_dictionary(state_dictionary):
    memory_buffer = io.BytesIO()
    torch.save(state_dictionary, memory_buffer)
    return memory_buffer.getvalue()


def deserialize_state_dictionary(state_dictionary_bytes):
    memory_buffer = io.BytesIO(state_dictionary_bytes)
    return torch.load(memory_buffer, map_location="cpu")


def generate_episode(environment, q_network, epsilon, epsilon_enemy):
    environment.reset()
    episode_ally_state_actions = []
    episode_enemy_state_actions = []

    ally_reward_list = []
    enemy_reward_list = []

    final_reward_ally = 0.0
    final_reward_enemy = 0.0

    done = False
    is_ally = True

    last_ally_action_index_in_turn = None
    last_enemy_action_index_in_turn = None

    previous_turn_counter = int(environment.game_state.status["turncounter"])

    while not done:
        if is_ally:
            action, state_action_vector = choose_action_epsilon_greedy(
                environment, q_network, is_ally, epsilon
            )
            action_type, done = environment.step(action, is_ally)
            episode_ally_state_actions.append(state_action_vector)
            ally_reward_list.append(0.0)
            last_ally_action_index_in_turn = len(ally_reward_list) - 1

            if action_type == "Passed":
                is_ally = False
        else:
            action, _ = choose_action_epsilon_greedy(environment, q_network, is_ally, epsilon_enemy)
            action_type, done = environment.step(action, is_ally)
            if action_type == "Passed":
                is_ally = True

        current_turn_counter = int(environment.game_state.status["turncounter"])
        if current_turn_counter != previous_turn_counter:
            end_of_turn_reward = 0.7 * float(previous_turn_counter) * compute_end_of_turn_location_reward(
                environment.game_state)

            if last_ally_action_index_in_turn is not None:
                ally_reward_list[last_ally_action_index_in_turn] += end_of_turn_reward

            if last_enemy_action_index_in_turn is not None:
                enemy_reward_list[last_enemy_action_index_in_turn] -= end_of_turn_reward

            last_ally_action_index_in_turn = None
            last_enemy_action_index_in_turn = None
            previous_turn_counter = current_turn_counter

        if done:
            winner = environment.game_state.passStatus["winner"]
            if winner == "Ally":
                final_reward_ally = 2.0
                final_reward_enemy = -2.0
            elif winner == "Enemy":
                final_reward_ally = -2.0
                final_reward_enemy = 2.0
            else:
                final_reward_ally = 0.0
                final_reward_enemy = 0.0

            if len(ally_reward_list) > 0:
                ally_reward_list[len(ally_reward_list) - 1] += final_reward_ally
            if len(enemy_reward_list) > 0:
                enemy_reward_list[len(enemy_reward_list) - 1] += final_reward_enemy

    ally_return_list = compute_monte_carlo_returns(ally_reward_list, discount_factor=1.0)
    enemy_return_list = compute_monte_carlo_returns(enemy_reward_list, discount_factor=1.0)

    return (
        episode_ally_state_actions,
        episode_enemy_state_actions,
        ally_return_list,
        enemy_return_list,
        final_reward_ally,
        final_reward_enemy,
    )


def actor_task(seed, tasks_queue, results_queue):
    set_global_seed(seed)
    environment = SingleAgentTestEnvironment()
    input_dimension = get_input_dimension()
    q_network = QNetworkB(input_dimension)
    q_network.eval()

    while True:
        task_dictionary = tasks_queue.get()

        loaded_state_dictionary = deserialize_state_dictionary(task_dictionary["serialized_q_network"])
        q_network.load_state_dict(loaded_state_dictionary)
        q_network.eval()
        training_state_action_list = []
        target_return_list = []

        for i in range(task_dictionary["n_of_episodes"]):
            (
                episode_ally_state_actions,
                episode_enemy_state_actions,
                ally_return_list,
                enemy_return_list,
                final_reward_ally,
                final_reward_enemy,
            ) = generate_episode(environment, q_network, task_dictionary["epsilon"], task_dictionary["epsilon_enemy"])

            for state_action_vector, target_return in zip(episode_ally_state_actions, ally_return_list):
                training_state_action_list.append(state_action_vector)
                target_return_list.append(target_return)

            for state_action_vector, target_return in zip(episode_enemy_state_actions, enemy_return_list):
                training_state_action_list.append(state_action_vector)
                target_return_list.append(target_return)

        state_action_array = numpy.stack(training_state_action_list).astype(numpy.float32)
        target_array = numpy.array(target_return_list, dtype=numpy.float32)
        results_queue.put(
            {"state_action_array": state_action_array, "target_array": target_array}
        )


def train_deep_monte_carlo_with_logging(
    number_of_episodes,
    learning_rate=1e-4,
    epsilon_start=0.9,
    epsilon_end=0.05,
    seed_value=None,
    evaluation_interval=100,
    evaluation_games=50,
    decay_fraction=0.5,
    log_csv_path=None,
    save_model_path=None,
    n_of_actor_processes=5
):
    if seed_value is not None:
        set_global_seed(seed_value)
    start_time_seconds = time.time()

    input_dimension = get_input_dimension()
    q_network = QNetworkB(input_dimension)
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    csv_file = None
    csv_writer = None
    if log_csv_path is not None:
        csv_file, csv_writer = create_training_csv_writer(log_csv_path)

    episode_index = 0
    decay_episodes = int(number_of_episodes * decay_fraction)

    multiprocessing_context = multiprocessing.get_context("spawn")
    tasks_queue = multiprocessing_context.Queue()
    results_queue = multiprocessing_context.Queue()
    actors_list = []
    actor_index = 1
    while actor_index <= n_of_actor_processes:
        actor_seed = seed_value + (actor_index * 100)
        actor_process = multiprocessing_context.Process(
            target=actor_task,
            args=(actor_seed, tasks_queue, results_queue)
        )
        actor_process.start()
        actors_list.append(actor_process)
        actor_index += 1

    while True:
        if episode_index < decay_episodes:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * (
                1.0 - float(episode_index) / float(decay_episodes - 1)
            )
        else:
            epsilon = epsilon_end

        decay_episodes_enemy = int(number_of_episodes * 0.5)
        if episode_index >= decay_episodes_enemy:
            epsilon_enemy = 0.05
        else:
            epsilon_enemy = 1.0 - float(episode_index) / float(decay_episodes_enemy)

        q_network.eval()
        q_network_state_dictionary_bytes = serialize_state_dictionary(q_network.state_dict())

        episodes_per_actor = 200

        for i in range(n_of_actor_processes):
            tasks_queue.put({
                "n_of_episodes": episodes_per_actor,
                "serialized_q_network": q_network_state_dictionary_bytes,
                "epsilon": epsilon,
                "epsilon_enemy": epsilon_enemy
            })

        q_network.train()
        for i in range(n_of_actor_processes):
            result = results_queue.get()
            state_action_array = result["state_action_array"]
            target_array = result["target_array"]
            episode_index += episodes_per_actor

            if len(state_action_array) > 0:

                state_action_tensor = torch.from_numpy(state_action_array)
                target_tensor = torch.from_numpy(target_array)

                predicted_q_tensor = q_network(state_action_tensor)
                loss_value = loss_function(predicted_q_tensor, target_tensor)

                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
            deck_pair_ally_win_rate_value_list = []
            deck_pair_value_index = 0
            while deck_pair_value_index < 16:
                deck_pair_ally_win_rate_value_list.append("")
                deck_pair_value_index += 1

            if evaluation_interval is not None and evaluation_interval > 0:
                if (episode_index % evaluation_interval) == 0:
                    q_network.eval()
                    eval_results = evaluate_against_chosen_opponent(
                        q_network, evaluation_games, epsilon_agent=0.0
                    )
                    q_network.train()
                    elapsed_minutes = (time.time() - start_time_seconds) / 60.0
                    ally_win_rate = eval_results["ally_win_rate"]
                    enemy_win_rate = eval_results["enemy_win_rate"]
                    tie_rate = eval_results["tie_rate"]
                    deck_pair_ally_win_rate_value_list = extract_deck_pair_ally_win_rate_list(eval_results)

                    if csv_writer is not None:
                        write_training_csv_row(
                            csv_writer,
                            episode_index,
                            elapsed_minutes,
                            epsilon,
                            ally_win_rate,
                            enemy_win_rate,
                            tie_rate,
                            deck_pair_ally_win_rate_value_list,
                        )
                        csv_file.flush()

        if episode_index >= number_of_episodes:
            break

    if csv_file is not None:
        csv_file.close()

    if save_model_path is not None:
        save_q_network(q_network, save_model_path)

    for actor_process in actors_list:
        if actor_process.is_alive():
            actor_process.terminate()

    for actor_process in actors_list:
        actor_process.join()

    return q_network


if __name__ == "__main__":
    number_of_episodes = 900000
    network = train_deep_monte_carlo_with_logging(
        number_of_episodes=number_of_episodes,
        learning_rate=5e-5,
        epsilon_start=0.9,
        epsilon_end=0.01,
        seed_value=14,
        evaluation_interval=20000,
        evaluation_games=3000,
        decay_fraction=0.35,
        log_csv_path=f"training_log_DMC_{number_of_episodes}_episodes.csv",
        save_model_path=f"trained_q_network_DMC_{number_of_episodes}_episodes.pt",
    )
    print("Training finished.")

    input_dimension = get_input_dimension()

    loaded_q_network = load_q_network(
        f"trained_q_network_DMC_{number_of_episodes}_episodes.pt",
        input_dimension=input_dimension,
        architecture="B",
    )

    loaded_q_network.eval()
    final_eval_results = evaluate_against_chosen_opponent(
        loaded_q_network,
        number_of_games=10000,
        epsilon_agent=0.0,
        verbose=False
    )

    final_eval_results_with_self = evaluate_against_chosen_opponent(
        loaded_q_network,
        number_of_games=10000,
        epsilon_agent=0.0,
        verbose=False,
        opponent_type="Self"
    )
    print("Final evaluation vs random opponent:")
    for key, value in final_eval_results.items():
        print(f"{key}: {format_csv_value(value)}")

    print("Final evaluation vs self:")
    for key, value in final_eval_results_with_self.items():
        print(f"{key}: {format_csv_value(value)}")
