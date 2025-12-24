import random
import torch
import torch.nn as nn
import torch.optim as optim
from environment.TestUtilityFunctions import *
import time
from training.TrainingUtilityFunctions import *
from training.TrainingNetwork import QNetwork, load_q_network, save_q_network


def set_global_seed(seed_value):
    random.seed(seed_value)
    numpy.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def choose_random_action(environment, is_ally):
    game_state = environment.game_state
    legal_actions_list = get_legal_actions(game_state, is_ally)
    return random.choice(legal_actions_list)


def choose_action_epsilon_greedy(environment, q_network, is_ally, epsilon):
    game_state = environment.game_state
    legal_actions_list = get_legal_actions(game_state, is_ally)
    if random.random() < epsilon:
        chosen_action = random.choice(legal_actions_list)
        state_action_vector = build_observation_with_chosen_action(
            game_state, is_ally, chosen_action
        )
        return chosen_action, numpy.array(state_action_vector, dtype=numpy.float32)

    state_action_vector_list = []
    for action_tuple in legal_actions_list:
        state_action_vector = build_observation_with_chosen_action(
            game_state, is_ally, action_tuple
        )
        state_action_vector_list.append(state_action_vector)

    state_action_array = numpy.stack(state_action_vector_list).astype(numpy.float32)
    state_action_tensor = torch.from_numpy(state_action_array)

    with torch.no_grad():
        q_value_tensor = q_network(state_action_tensor)

    best_action_index = int(torch.argmax(q_value_tensor).item())
    best_action = legal_actions_list[best_action_index]
    best_state_action_vector = state_action_vector_list[best_action_index]

    return best_action, numpy.array(best_state_action_vector, dtype=numpy.float32)


def generate_episode(environment, q_network, epsilon, enemy_type):
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
            if enemy_type == "Self-Play":
                action, state_action_vector = choose_action_epsilon_greedy(
                    environment, q_network, is_ally, epsilon
                )
                action_type, done = environment.step(action, is_ally)
                episode_enemy_state_actions.append(state_action_vector)
                enemy_reward_list.append(0.0)
                last_enemy_action_index_in_turn = len(enemy_reward_list) - 1

                if action_type == "Passed":
                    is_ally = True
            else:
                action = choose_random_action(environment, False)
                action_type, done = environment.step(action, is_ally)
                if action_type == "Passed":
                    is_ally = True

        current_turn_counter = int(environment.game_state.status["turncounter"])
        if current_turn_counter != previous_turn_counter:
            end_of_turn_reward = float(previous_turn_counter) * compute_end_of_turn_location_reward(environment.game_state)

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


def train_deep_monte_carlo_with_logging(
    number_of_episodes,
    learning_rate=1e-4,
    epsilon_start=0.9,
    epsilon_end=0.05,
    seed_value=None,
    evaluation_interval=100,
    evaluation_games=50,
    log_csv_path=None,
    save_model_path=None,
):
    if seed_value is not None:
        set_global_seed(seed_value)
    start_time_seconds = time.time()

    input_dimension = get_input_dimension()
    q_network = QNetwork(input_dimension)
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    environment = SingleAgentTestEnvironment()

    training_loss_history = []
    ally_win_rate_history = []
    enemy_win_rate_history = []
    tie_rate_history = []

    csv_file = None
    csv_writer = None
    if log_csv_path is not None:
        csv_file, csv_writer = create_training_csv_writer(log_csv_path)

    episode_index = 0
    while episode_index < number_of_episodes:
        if number_of_episodes > 1:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * (
                1.0 - float(episode_index) / float(number_of_episodes - 1)
            )
        else:
            epsilon = epsilon_start

        block_size = 10
        block_index = int(episode_index / block_size)

        if block_index % 2 == 0:
            enemy_type = "Self-Play"
        else:
            enemy_type = "Random"
        (
            episode_ally_state_actions,
            episode_enemy_state_actions,
            ally_return_list,
            enemy_return_list,
            final_reward_ally,
            final_reward_enemy,
        ) = generate_episode(environment, q_network, epsilon, enemy_type)

        training_state_action_list = []
        target_return_list = []

        for state_action_vector, target_return in zip(episode_ally_state_actions, ally_return_list):
            training_state_action_list.append(state_action_vector)
            target_return_list.append(target_return)

        for state_action_vector, target_return in zip(episode_enemy_state_actions, enemy_return_list):
            training_state_action_list.append(state_action_vector)
            target_return_list.append(target_return)

        loss_value_float = 0.0

        if len(training_state_action_list) > 0:
            state_action_array = numpy.stack(training_state_action_list).astype(
                numpy.float32
            )
            target_array = numpy.array(target_return_list, dtype=numpy.float32)

            state_action_tensor = torch.from_numpy(state_action_array)
            target_tensor = torch.from_numpy(target_array)

            predicted_q_tensor = q_network(state_action_tensor)
            loss_value = loss_function(predicted_q_tensor, target_tensor)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            loss_value_float = float(loss_value.item())
            training_loss_history.append(loss_value_float)

        ally_win_rate = ""
        enemy_win_rate = ""
        tie_rate = ""
        elapsed_minutes = ""
        deck_pair_ally_win_rate_value_list = []
        deck_pair_value_index = 0
        while deck_pair_value_index < 16:
            deck_pair_ally_win_rate_value_list.append("")
            deck_pair_value_index += 1

        if evaluation_interval is not None and evaluation_interval > 0:
            if (episode_index + 1) % evaluation_interval == 0:
                eval_results = evaluate_against_random_opponent(
                    q_network, evaluation_games, epsilon_agent=0.0
                )
                elapsed_minutes = (time.time() - start_time_seconds) / 60.0
                ally_win_rate = eval_results["ally_win_rate"]
                enemy_win_rate = eval_results["enemy_win_rate"]
                tie_rate = eval_results["tie_rate"]
                deck_pair_ally_win_rate_value_list = extract_deck_pair_ally_win_rate_list(eval_results)

                ally_win_rate_history.append(ally_win_rate)
                enemy_win_rate_history.append(enemy_win_rate)
                tie_rate_history.append(tie_rate)

        if csv_writer is not None:
            write_training_csv_row(
                csv_writer,
                episode_index + 1,
                elapsed_minutes,
                epsilon,
                loss_value_float,
                final_reward_ally,
                final_reward_enemy,
                ally_win_rate,
                enemy_win_rate,
                tie_rate,
                deck_pair_ally_win_rate_value_list,
            )

        episode_index += 1

    if csv_file is not None:
        csv_file.close()

    if save_model_path is not None:
        save_q_network(q_network, save_model_path)

    results_dictionary = {
        "q_network": q_network,
        "training_loss_history": training_loss_history,
        "ally_win_rate_history": ally_win_rate_history,
        "enemy_win_rate_history": enemy_win_rate_history,
        "tie_rate_history": tie_rate_history,
    }
    return results_dictionary


def evaluate_against_random_opponent(q_network, number_of_games, epsilon_agent=0.0, verbose=False):
    environment = SingleAgentTestEnvironment(verbose)

    ally_wins = 0
    enemy_wins = 0
    ties = 0

    deck_pair_total_game_count_matrix = []
    deck_pair_ally_win_count_matrix = []

    deck_index_row = 0
    while deck_index_row < 4:
        deck_pair_total_game_count_matrix.append([0, 0, 0, 0])
        deck_pair_ally_win_count_matrix.append([0, 0, 0, 0])
        deck_index_row += 1

    game_index = 0
    while game_index < number_of_games:
        environment.reset()
        done = False
        is_ally = True

        while not done:
            if is_ally:
                action, _ = choose_action_epsilon_greedy(
                    environment, q_network, True, epsilon_agent
                )
            else:
                action = choose_random_action(environment, False)

            action_type, done = environment.step(action, is_ally)

            if action_type == "Passed":
                is_ally = not is_ally

        ally_deck_number = int(environment.game_state.ally_deck_number)
        enemy_deck_number = int(environment.game_state.enemy_deck_number)

        ally_deck_index = ally_deck_number - 1
        enemy_deck_index = enemy_deck_number - 1

        deck_pair_total_game_count_matrix[ally_deck_index][enemy_deck_index] += 1

        winner = environment.game_state.passStatus["winner"]
        if winner == "Ally":
            ally_wins += 1
            deck_pair_ally_win_count_matrix[ally_deck_index][enemy_deck_index] += 1
        elif winner == "Enemy":
            enemy_wins += 1
        else:
            ties += 1

        game_index += 1

    total_games = float(number_of_games)
    ally_win_rate = ally_wins / total_games
    enemy_win_rate = enemy_wins / total_games
    tie_rate = ties / total_games

    results_dictionary = {
        "ally_wins": ally_wins,
        "enemy_wins": enemy_wins,
        "ties": ties,
        "ally_win_rate": ally_win_rate,
        "enemy_win_rate": enemy_win_rate,
        "tie_rate": tie_rate,
    }

    compute_individual_decks_win_rate(deck_pair_ally_win_count_matrix, deck_pair_total_game_count_matrix,
                                      results_dictionary)

    return results_dictionary


if __name__ == "__main__":
    number_of_episodes = 250000
    results = train_deep_monte_carlo_with_logging(
        number_of_episodes=number_of_episodes,
        learning_rate=3e-4,
        epsilon_start=0.9,
        epsilon_end=0.05,
        seed_value=54,
        evaluation_interval=5000,
        evaluation_games=2000,
        log_csv_path=f"training_log_{number_of_episodes}_episodes.csv",
        save_model_path=f"trained_q_network_{number_of_episodes}_episodes.pt",
    )

    trained_q_network = results["q_network"]
    print("Training finished.")

    input_dimension = trained_q_network.linear_layer_1.in_features

    loaded_q_network = load_q_network(
        f"trained_q_network_{number_of_episodes}_episodes.pt",
        input_dimension=input_dimension,
        hidden_dimension=512,
    )

    final_eval_results = evaluate_against_random_opponent(
        loaded_q_network,
        number_of_games=10000,
        epsilon_agent=0.0,
        verbose=False
    )
    print("Final evaluation vs random opponent:")
    for key, value in final_eval_results.items():
        print(f"{key}: {format_csv_value(value)}")
