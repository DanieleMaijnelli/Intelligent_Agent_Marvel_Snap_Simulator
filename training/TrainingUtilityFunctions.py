import csv
from environment.SingleAgentTestEnvironment import SingleAgentTestEnvironment
from environment.TestUtilityFunctions import build_observation_with_chosen_action, get_legal_actions
import random
import torch
import numpy


def set_global_seed(seed_value):
    random.seed(seed_value)
    numpy.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def create_training_csv_writer(log_csv_path):
    csv_file = open(log_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)

    header_row = [
        "episode",
        "elapsed_minutes",
        "epsilon",
        "ally_win_rate",
        "enemy_win_rate",
        "tie_rate",
    ]

    ally_deck_number = 1
    while ally_deck_number <= 4:
        enemy_deck_number = 1
        while enemy_deck_number <= 4:
            header_row.append(
                "ally_win_rate_deck_" + str(ally_deck_number) + "_vs_" + str(enemy_deck_number)
            )
            enemy_deck_number += 1
        ally_deck_number += 1

    csv_writer.writerow(header_row)
    return csv_file, csv_writer


def extract_deck_pair_ally_win_rate_list(eval_results):
    deck_pair_ally_win_rate_value_list = []
    ally_deck_number = 1
    while ally_deck_number <= 4:
        enemy_deck_number = 1
        while enemy_deck_number <= 4:
            key_name = "ally_win_rate_deck_" + str(ally_deck_number) + "_vs_" + str(enemy_deck_number)
            deck_pair_ally_win_rate_value_list.append(eval_results[key_name])
            enemy_deck_number += 1
        ally_deck_number += 1
    return deck_pair_ally_win_rate_value_list


def write_training_csv_row(
    csv_writer,
    episode_number,
    elapsed_minutes,
    epsilon,
    ally_win_rate,
    enemy_win_rate,
    tie_rate,
    deck_pair_ally_win_rate_value_list,
):
    if csv_writer is None:
        return

    row_value_list = [
                         episode_number,
                         elapsed_minutes,
                         epsilon,
                         ally_win_rate,
                         enemy_win_rate,
                         tie_rate,
                     ] + deck_pair_ally_win_rate_value_list

    formatted_row_value_list = []
    value_index = 0
    while value_index < len(row_value_list):
        formatted_row_value_list.append(format_csv_value(row_value_list[value_index]))
        value_index += 1

    csv_writer.writerow(formatted_row_value_list)


def format_csv_value(value_object):
    if value_object == "":
        return ""
    if value_object is None:
        return ""
    if isinstance(value_object, float):
        return "{:.3f}".format(value_object)
    if isinstance(value_object, int):
        return str(value_object)
    return str(value_object)


def get_input_dimension():
    dummy_environment = SingleAgentTestEnvironment()
    dummy_environment.reset()
    dummy_action = (-1, -1)
    dummy_state_action_vector = build_observation_with_chosen_action(
        dummy_environment.game_state, True, dummy_action
    )
    return len(dummy_state_action_vector)


def compute_individual_decks_win_rate(deck_pair_ally_win_count_matrix, deck_pair_total_game_count_matrix,
                                      results_dictionary):
    ally_deck_number = 1
    while ally_deck_number <= 4:
        enemy_deck_number = 1
        while enemy_deck_number <= 4:
            ally_deck_index = ally_deck_number - 1
            enemy_deck_index = enemy_deck_number - 1

            total_count = int(deck_pair_total_game_count_matrix[ally_deck_index][enemy_deck_index])
            ally_win_count = int(deck_pair_ally_win_count_matrix[ally_deck_index][enemy_deck_index])

            if total_count > 0:
                deck_pair_ally_win_rate = float(ally_win_count) / float(total_count)
            else:
                deck_pair_ally_win_rate = 0.0

            key_name = "ally_win_rate_deck_" + str(ally_deck_number) + "_vs_" + str(enemy_deck_number)
            results_dictionary[key_name] = deck_pair_ally_win_rate

            enemy_deck_number += 1
        ally_deck_number += 1


def compute_end_of_turn_location_reward(game_state):
    reward_value = 0.0

    location_1 = game_state.locationList["location1"]
    location_2 = game_state.locationList["location2"]
    location_3 = game_state.locationList["location3"]

    for location_object in [location_1, location_2, location_3]:
        if location_object.winning == "Ally":
            reward_value += 0.1
        elif location_object.winning == "Enemy":
            reward_value -= 0.1

    return reward_value


def compute_monte_carlo_returns(reward_list, discount_factor=1.0):
    returns_list = [0.0] * len(reward_list)

    running_return = 0.0
    index_value = len(reward_list) - 1
    while index_value >= 0:
        running_return = reward_list[index_value] + discount_factor * running_return
        returns_list[index_value] = running_return
        index_value -= 1

    return returns_list


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


def evaluate_against_chosen_opponent(q_network, number_of_games, epsilon_agent=0.0, verbose=False, opponent_type="Random"):
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
                if opponent_type == "Random":
                    action = choose_random_action(environment, False)
                else:
                    action, _ = choose_action_epsilon_greedy(environment, q_network, False, epsilon_agent)

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
