import csv
import random

from environment.PlayerAgentEnvironment import PlayerAgentEnvironment
from environment.EnvironmentUtilityFunctions import get_legal_actions

PASS_ACTION = (-1, -1)


def format_csv_value(value_object):
    if value_object == "" or value_object is None:
        return ""
    if isinstance(value_object, float):
        return "{:.3f}".format(value_object)
    if isinstance(value_object, int):
        return str(value_object)
    return str(value_object)


def compute_individual_decks_win_rate(num_decks, deck_pair_ally_win_count_matrix, deck_pair_total_game_count_matrix,
                                      results_dictionary):
    ally_deck_number = 1
    while ally_deck_number <= num_decks:
        enemy_deck_number = 1
        while enemy_deck_number <= num_decks:
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


def choose_action_random_baseline(environment, is_ally):
    game_state = environment.game_state

    # se hai già passato, ritorna pass
    if is_ally and game_state.status["allypass"]:
        return PASS_ACTION
    if (not is_ally) and game_state.status["enemypass"]:
        return PASS_ACTION

    legal_actions_list = get_legal_actions(game_state, is_ally)

    playable_actions_list = []
    action_index = 0
    while action_index < len(legal_actions_list):
        action_tuple = legal_actions_list[action_index]
        if action_tuple != PASS_ACTION:
            playable_actions_list.append(action_tuple)
        action_index += 1

    if len(playable_actions_list) == 0:
        return PASS_ACTION

    return random.choice(playable_actions_list)


def choose_action_greedy_baseline(environment, is_ally, epsilon_random=0.01):
    game_state = environment.game_state
    legal_actions_list = get_legal_actions(game_state, is_ally)

    playable_actions_list = []
    action_index = 0
    while action_index < len(legal_actions_list):
        action_tuple = legal_actions_list[action_index]
        if action_tuple != PASS_ACTION:
            playable_actions_list.append(action_tuple)
        action_index += 1

    if len(playable_actions_list) == 0:
        return PASS_ACTION

    # 1% random tra quelle giocabili (non includo PASS)
    if random.random() < epsilon_random:
        return random.choice(playable_actions_list)

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

        # per sicurezza, aggiorna i field
        location.countPower()

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


def choose_baseline_action(environment, is_ally, policy_type):
    if policy_type == "Random":
        return choose_action_random_baseline(environment, is_ally)
    elif policy_type == "Greedy":
        return choose_action_greedy_baseline(environment, is_ally, epsilon_random=0.01)
    else:
        raise ValueError("Unknown policy_type: " + str(policy_type))


def create_baseline_eval_csv_writer(csv_path, num_decks=6):
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)

    header_row = [
        "ally_policy",
        "enemy_policy",
        "agent_deck_type",
        "number_of_games",
        "ally_win_rate",
        "enemy_win_rate",
        "tie_rate",
    ]

    ally_deck_number = 1
    while ally_deck_number <= num_decks:
        enemy_deck_number = 1
        while enemy_deck_number <= num_decks:
            header_row.append("ally_win_rate_deck_" + str(ally_deck_number) + "_vs_" + str(enemy_deck_number))
            enemy_deck_number += 1
        ally_deck_number += 1

    csv_writer.writerow(header_row)
    return csv_file, csv_writer


def write_baseline_eval_csv_row(csv_writer, ally_policy, enemy_policy, agent_deck_type, number_of_games, eval_results,
                                num_decks=6):
    row_value_list = [
        ally_policy,
        enemy_policy,
        agent_deck_type,
        number_of_games,
        eval_results.get("ally_win_rate", 0.0),
        eval_results.get("enemy_win_rate", 0.0),
        eval_results.get("tie_rate", 0.0),
    ]

    ally_deck_number = 1
    while ally_deck_number <= num_decks:
        enemy_deck_number = 1
        while enemy_deck_number <= num_decks:
            key_name = "ally_win_rate_deck_" + str(ally_deck_number) + "_vs_" + str(enemy_deck_number)
            row_value_list.append(eval_results.get(key_name, 0.0))
            enemy_deck_number += 1
        ally_deck_number += 1

    formatted_row_value_list = []
    value_index = 0
    while value_index < len(row_value_list):
        formatted_row_value_list.append(format_csv_value(row_value_list[value_index]))
        value_index += 1

    csv_writer.writerow(formatted_row_value_list)


def evaluate_baseline_vs_baseline(
    number_of_games,
    ally_policy,
    enemy_policy,
    agent_deck_type=0,
    verbose=False,
    num_decks=6
):
    # Se nel tuo progetto PlayerAgentEnvironment accetta decktype, qui lo passi.
    # Se NON lo accetta, lascia solo verbose=False (adegua in base al tuo costruttore).
    environment = PlayerAgentEnvironment(verbose=verbose)

    ally_wins = 0
    enemy_wins = 0
    ties = 0

    deck_pair_total_game_count_matrix = [[0] * num_decks for _ in range(num_decks)]
    deck_pair_ally_win_count_matrix = [[0] * num_decks for _ in range(num_decks)]

    game_index = 0
    while game_index < number_of_games:
        environment.reset()
        done = False
        is_ally_turn = True

        while not done:
            if is_ally_turn:
                action = choose_baseline_action(environment, True, ally_policy)
            else:
                action = choose_baseline_action(environment, False, enemy_policy)

            action_type, done = environment.step(action, is_ally_turn)

            if action_type == "Passed":
                is_ally_turn = not is_ally_turn

        winner = environment.game_state.passStatus["winner"]
        if winner == "Ally":
            ally_wins += 1
        elif winner == "Enemy":
            enemy_wins += 1
        else:
            ties += 1

        # matchup matrix (se disponibile)
        if hasattr(environment.game_state, "ally_deck_number") and hasattr(environment.game_state, "enemy_deck_number"):
            ally_deck_number = int(environment.game_state.ally_deck_number)
            enemy_deck_number = int(environment.game_state.enemy_deck_number)

            if 1 <= ally_deck_number <= num_decks and 1 <= enemy_deck_number <= num_decks:
                ally_deck_index = ally_deck_number - 1
                enemy_deck_index = enemy_deck_number - 1

                deck_pair_total_game_count_matrix[ally_deck_index][enemy_deck_index] += 1
                if winner == "Ally":
                    deck_pair_ally_win_count_matrix[ally_deck_index][enemy_deck_index] += 1

        game_index += 1

    total_games = float(number_of_games)
    results_dictionary = {
        "ally_win_rate": ally_wins / total_games,
        "enemy_win_rate": enemy_wins / total_games,
        "tie_rate": ties / total_games,
    }

    compute_individual_decks_win_rate(num_decks, deck_pair_ally_win_count_matrix, deck_pair_total_game_count_matrix,
                                      results_dictionary)
    return results_dictionary


def evaluate_random_greedy_baselines_to_csv(
    csv_path,
    number_of_games=10000,
    verbose=False,
    num_decks=6
):
    csv_file, csv_writer = create_baseline_eval_csv_writer(csv_path, num_decks=num_decks)

    ally_policy_list = ["Random", "Greedy"]
    enemy_policy_list = ["Random", "Greedy"]
    agent_deck_type_list = [0, 1, 2]

    ally_policy_index = 0
    while ally_policy_index < len(ally_policy_list):
        ally_policy = ally_policy_list[ally_policy_index]

        enemy_policy_index = 0
        while enemy_policy_index < len(enemy_policy_list):
            enemy_policy = enemy_policy_list[enemy_policy_index]

            deck_type_index = 0
            while deck_type_index < len(agent_deck_type_list):
                agent_deck_type = agent_deck_type_list[deck_type_index]

                eval_results = evaluate_baseline_vs_baseline(
                    number_of_games=number_of_games,
                    ally_policy=ally_policy,
                    enemy_policy=enemy_policy,
                    agent_deck_type=agent_deck_type,
                    verbose=verbose,
                    num_decks=num_decks
                )

                write_baseline_eval_csv_row(
                    csv_writer,
                    ally_policy=ally_policy,
                    enemy_policy=enemy_policy,
                    agent_deck_type=agent_deck_type,
                    number_of_games=number_of_games,
                    eval_results=eval_results,
                    num_decks=num_decks
                )
                csv_file.flush()

                deck_type_index += 1

            enemy_policy_index += 1

        ally_policy_index += 1

    csv_file.close()
