import csv


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
