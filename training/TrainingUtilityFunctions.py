import csv
from environment.PlayerAgentEnvironment import PlayerAgentEnvironment
from environment.SnapAgentEnvironment import Action, SnapAgentEnvironment
from environment.EnvironmentUtilityFunctions import build_observation_with_chosen_action, get_legal_actions, \
    build_observation_snap_agent_with_chosen_action
import random
import torch
import numpy


def set_global_seed(seed_value):
    random.seed(seed_value)
    numpy.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def create_snap_training_csv_writer(log_csv_path):
    csv_file = open(log_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)

    header_row = [
        "episode",
        "elapsed_minutes",
        "epsilon",
        "ally_win_rate",
        "enemy_win_rate",
        "tie_rate",
        "average_cubes_won",
    ]

    csv_writer.writerow(header_row)
    return csv_file, csv_writer


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


def write_snap_training_csv_row(
    csv_writer,
    episode_number,
    elapsed_minutes,
    epsilon,
    ally_win_rate,
    enemy_win_rate,
    tie_rate,
    average_cubes_won
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
        average_cubes_won
    ]

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
    dummy_environment = PlayerAgentEnvironment()
    dummy_environment.reset()
    dummy_action = (-1, -1)
    dummy_state_action_vector = build_observation_with_chosen_action(
        dummy_environment.game_state, True, dummy_action
    )
    return len(dummy_state_action_vector)


def get_snap_input_dimension():
    dummy_environment = PlayerAgentEnvironment()
    dummy_environment.reset()
    dummy_state_action_vector = build_observation_snap_agent_with_chosen_action(
        dummy_environment.game_state, True, Action.NOTHING
    )
    return len(dummy_state_action_vector)


def compute_individual_decks_win_rate(deck_pair_ally_win_count_matrix, deck_pair_total_game_count_matrix,
                                      results_dictionary):
    ally_deck_number = 1
    while ally_deck_number <= 6:
        enemy_deck_number = 1
        while enemy_deck_number <= 6:
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


def choose_greedy_action(environment, is_ally: bool):
    game_state = environment.game_state
    legal_actions_list = get_legal_actions(game_state, is_ally)

    playable_actions_list = []
    for action_tuple in legal_actions_list:
        if action_tuple != (-1, -1):
            playable_actions_list.append(action_tuple)

    if len(playable_actions_list) == 0:
        chosen_action = (-1, -1)
        return chosen_action

    if random.random() < 0.01:
        chosen_action = random.choice(legal_actions_list)
        return chosen_action

    status_dictionary = game_state.status
    location_dictionary = game_state.locationList
    hand = status_dictionary["allyhand"] if is_ally else status_dictionary["enemyhand"]

    best_action = None
    best_diff = None
    best_power = None

    action_index = 0
    while action_index < len(playable_actions_list):
        hand_index, location_index = playable_actions_list[action_index]

        location_key = "location" + str(location_index + 1)
        location = location_dictionary[location_key]

        if is_ally:
            my_power = int(location.alliesPower)
            enemy_power = int(location.enemiesPower)
        else:
            my_power = int(location.enemiesPower)
            enemy_power = int(location.alliesPower)

        diff = abs(my_power - enemy_power)
        card_power = int(hand[hand_index].cur_power)

        if (best_action is None) or (diff < best_diff) or (diff == best_diff and card_power > best_power):
            best_action = (hand_index, location_index)
            best_diff = diff
            best_power = card_power

        action_index += 1

    return best_action


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


def choose_snap_action_epsilon_greedy(environment, snap_q_network, is_ally, epsilon):
    game_state = environment.game_state
    if (game_state.status["allysnapped"] and is_ally) or (game_state.status["enemysnapped"] and (not is_ally)):
        legal_actions_list = (Action.NOTHING, Action.RETREAT)
    else:
        legal_actions_list = (Action.NOTHING, Action.RETREAT, Action.SNAP)

    if random.random() < epsilon:
        if len(legal_actions_list) >= 3:
            r = random.random()
            if r < 0.88:
                chosen_action = Action.NOTHING
            elif r < 0.90:
                chosen_action = Action.RETREAT
            else:
                chosen_action = Action.SNAP
        else:
            r = random.random()
            chosen_action = Action.NOTHING if r < 0.98 else Action.RETREAT

        state_action_vector = build_observation_snap_agent_with_chosen_action(
            game_state, is_ally, chosen_action
        )
        return chosen_action, numpy.array(state_action_vector, dtype=numpy.float32)

    state_action_vector_list = []
    for action_tuple in legal_actions_list:
        state_action_vector = build_observation_snap_agent_with_chosen_action(
            game_state, is_ally, action_tuple
        )
        state_action_vector_list.append(state_action_vector)

    state_action_array = numpy.stack(state_action_vector_list).astype(numpy.float32)
    state_action_tensor = torch.from_numpy(state_action_array)

    with torch.no_grad():
        q_value_tensor = snap_q_network(state_action_tensor)

    best_action_index = int(torch.argmax(q_value_tensor).item())
    best_action = legal_actions_list[best_action_index]
    best_state_action_vector = state_action_vector_list[best_action_index]

    return best_action, numpy.array(best_state_action_vector, dtype=numpy.float32)


def evaluate_against_chosen_opponent(q_network, number_of_games, epsilon_agent=0.01, verbose=False,
                                     opponent_type="Random", agent_deck_type=0):
    environment = PlayerAgentEnvironment(verbose, agent_deck_type)

    ally_wins = 0
    enemy_wins = 0
    ties = 0

    NUM_DECKS = 6

    deck_pair_total_game_count_matrix = [[0] * NUM_DECKS for _ in range(NUM_DECKS)]
    deck_pair_ally_win_count_matrix = [[0] * NUM_DECKS for _ in range(NUM_DECKS)]

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
                elif opponent_type == "Greedy":
                    action = choose_greedy_action(environment, False)
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


def evaluate_snap_agent(player_q_network, snap_q_network, number_of_games, agent_deck_type=0):
    environment = SnapAgentEnvironment(player_q_network, agent_deck_type=agent_deck_type)

    ally_wins = 0
    enemy_wins = 0
    ties = 0
    cubes_won = 0.0

    game_index = 0
    while game_index < number_of_games:
        environment.reset()
        done = False

        while not done:
            action, _ = choose_snap_action_epsilon_greedy(
                environment, snap_q_network, True, 0.0
            )
            done = environment.step(action)

        winner = environment.game_state.passStatus["winner"]
        if winner == "Ally":
            ally_wins += 1
            cubes_won += environment.game_state.status["cubes"]
        elif winner == "Enemy":
            enemy_wins += 1
            cubes_won -= environment.game_state.status["cubes"]
        else:
            ties += 1

        game_index += 1

    total_games = float(number_of_games)
    ally_win_rate = ally_wins / total_games
    enemy_win_rate = enemy_wins / total_games
    tie_rate = ties / total_games
    average_cubes_won = cubes_won / total_games

    results_dictionary = {
        "ally_win_rate": ally_win_rate,
        "enemy_win_rate": enemy_win_rate,
        "tie_rate": tie_rate,
        "average_cubes_won": average_cubes_won
    }

    return results_dictionary


def evaluate_cube_samples(player_q_network, snap_q_network, number_of_games):
    environment = SnapAgentEnvironment(player_q_network)

    samples = []
    for game_index in range(number_of_games):
        environment.reset()
        done = False
        while not done:
            action, _ = choose_snap_action_epsilon_greedy(environment, snap_q_network, True, 0.0)
            done = environment.step(action)

        winner = environment.game_state.passStatus["winner"]
        cubes = float(environment.game_state.status["cubes"])

        if winner == "Ally":
            cubes_net = cubes
        elif winner == "Enemy":
            cubes_net = -cubes
        else:
            cubes_net = 0.0

        samples.append((game_index, cubes, winner, cubes_net))

    return samples


def append_cube_samples_to_csv(csv_path, checkpoint, episode, samples):
    try:
        with open(csv_path, "r"):
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["checkpoint", "episode", "game_index", "cubes_stake", "winner", "cubes_net"])

        for (game_index, cubes, winner, cubes_net) in samples:
            w.writerow([checkpoint, episode, game_index, cubes, winner, cubes_net])
