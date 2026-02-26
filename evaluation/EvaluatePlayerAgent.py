import os
import torch

from training.TrainingNetwork import load_q_network
from training.TrainingUtilityFunctions import format_csv_value
from training.TrainingUtilityFunctions import get_input_dimension

from training.TrainingDeepMonteCarlo import evaluate_against_chosen_opponent

import csv


def create_full_matchup_eval_csv_writer(csv_path, num_decks=6):
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)

    header_row = [
        "opponent_type",
        "agent_deck_type",
        "number_of_games",
        "epsilon_agent",
        "ally_win_rate",
        "enemy_win_rate",
        "tie_rate",
    ]

    ally_deck_number = 1
    while ally_deck_number <= num_decks:
        enemy_deck_number = 1
        while enemy_deck_number <= num_decks:
            header_row.append(
                "ally_win_rate_deck_" + str(ally_deck_number) + "_vs_" + str(enemy_deck_number)
            )
            enemy_deck_number += 1
        ally_deck_number += 1

    csv_writer.writerow(header_row)
    return csv_file, csv_writer


def write_full_matchup_eval_csv_row(
    csv_writer,
    opponent_type,
    agent_deck_type,
    number_of_games,
    epsilon_agent,
    eval_results,
    num_decks=6,
):
    row_value_list = [
        opponent_type,
        agent_deck_type,
        number_of_games,
        epsilon_agent,
        eval_results.get("ally_win_rate", ""),
        eval_results.get("enemy_win_rate", ""),
        eval_results.get("tie_rate", ""),
    ]

    ally_deck_number = 1
    while ally_deck_number <= num_decks:
        enemy_deck_number = 1
        while enemy_deck_number <= num_decks:
            key_name = "ally_win_rate_deck_" + str(ally_deck_number) + "_vs_" + str(enemy_deck_number)
            row_value_list.append(eval_results.get(key_name, ""))
            enemy_deck_number += 1
        ally_deck_number += 1

    formatted_row_value_list = []
    value_index = 0
    while value_index < len(row_value_list):
        formatted_row_value_list.append(format_csv_value(row_value_list[value_index]))
        value_index += 1

    csv_writer.writerow(formatted_row_value_list)


def evaluate_6_tests_to_csv(
    q_network,
    csv_path,
    number_of_games=10000,
    epsilon_agent=0.01,
    verbose=False,
    num_decks=6,
):
    csv_file, csv_writer = create_full_matchup_eval_csv_writer(csv_path, num_decks=num_decks)

    opponent_type_list = ["Random", "Greedy"]
    agent_deck_type_list = [0, 1, 2]

    opp_i = 0
    while opp_i < len(opponent_type_list):
        opponent_type = opponent_type_list[opp_i]

        dt_i = 0
        while dt_i < len(agent_deck_type_list):
            agent_deck_type = agent_deck_type_list[dt_i]

            eval_results = evaluate_against_chosen_opponent(
                q_network,
                number_of_games=number_of_games,
                epsilon_agent=epsilon_agent,
                verbose=verbose,
                opponent_type=opponent_type,
                agent_deck_type=agent_deck_type,
            )

            write_full_matchup_eval_csv_row(
                csv_writer,
                opponent_type=opponent_type,
                agent_deck_type=agent_deck_type,
                number_of_games=number_of_games,
                epsilon_agent=epsilon_agent,
                eval_results=eval_results,
                num_decks=num_decks,
            )
            csv_file.flush()

            dt_i += 1

        opp_i += 1

    csv_file.close()


if __name__ == "__main__":
    input_dimension = get_input_dimension()

    loaded_q_network = load_q_network(
        "trained_q_network_DMC_2000000_episodes.pt",
        input_dimension=input_dimension,
        architecture="B",
    )
    loaded_q_network.eval()

    evaluate_6_tests_to_csv(
        loaded_q_network,
        csv_path="model_evaluation_DMC_2000000_episodes.csv",
        number_of_games=10000,
        epsilon_agent=0.01,
        verbose=False,
        num_decks=6,
    )
