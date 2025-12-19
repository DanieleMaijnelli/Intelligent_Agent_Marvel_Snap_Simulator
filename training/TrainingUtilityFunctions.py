import csv
from environment.SingleAgentTestEnvironment import SingleAgentTestEnvironment
from environment.TestUtilityFunctions import build_observation_with_chosen_action


def create_training_csv_writer(log_csv_path):
    csv_file = open(log_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)

    header_row = [
        "episode",
        "elapsed_hours",
        "epsilon",
        "loss",
        "final_reward_ally",
        "final_reward_enemy",
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
    elapsed_hours,
    epsilon,
    loss_value_float,
    final_reward_ally,
    final_reward_enemy,
    ally_win_rate,
    enemy_win_rate,
    tie_rate,
    deck_pair_ally_win_rate_value_list,
):
    if csv_writer is None:
        return

    row_value_list = [
                         episode_number,
                         elapsed_hours,
                         epsilon,
                         loss_value_float,
                         final_reward_ally,
                         final_reward_enemy,
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
