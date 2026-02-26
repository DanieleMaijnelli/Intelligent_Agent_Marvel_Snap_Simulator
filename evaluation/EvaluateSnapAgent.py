import csv
import inspect
import random
from training.TrainingNetwork import load_q_network
from training.TrainingUtilityFunctions import (
    format_csv_value,
    get_input_dimension,
    get_snap_input_dimension,
    choose_action_epsilon_greedy,
    choose_snap_action_epsilon_greedy,
)
from environment.SnapAgentEnvironment import SnapAgentEnvironment


def create_snap_eval_csv_writer(csv_path: str):
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)

    header_row = [
        "opponent_type",
        "agent_deck_type",
        "enemy_deck_type",
        "number_of_games",
        "epsilon_snap",
        "ally_win_rate",
        "average_cubes_won",
    ]
    csv_writer.writerow(header_row)
    return csv_file, csv_writer


def write_snap_eval_csv_row(
    csv_writer,
    opponent_type: str,
    agent_deck_type: int,
    enemy_deck_type: int,
    number_of_games: int,
    epsilon_snap: float,
    ally_win_rate: float,
    average_cubes_won: float,
):
    row = [
        opponent_type,
        agent_deck_type,
        enemy_deck_type,
        number_of_games,
        epsilon_snap,
        ally_win_rate,
        average_cubes_won,
    ]
    csv_writer.writerow([format_csv_value(v) for v in row])


def evaluate_snap_agent_against_opponent(
    player_q_network,
    snap_q_network,
    number_of_games: int,
    opponent_type: str,
    agent_deck_type: int,
    epsilon_snap: float = 0.0
):
    env = SnapAgentEnvironment(
        player_q_network,
        agent_deck_type=agent_deck_type,
        enemy_type=opponent_type,
    )

    ally_wins = 0
    enemy_wins = 0
    ties = 0
    cubes_net = 0.0

    for _ in range(number_of_games):
        env.reset()
        done = False

        while not done:
            action, _ = choose_snap_action_epsilon_greedy(env, snap_q_network, True, epsilon_snap)
            done = env.step(action)

        winner = env.game_state.passStatus.get("winner", "Tie")
        cubes = float(env.game_state.status.get("cubes", 1))

        if winner == "Ally":
            ally_wins += 1
            cubes_net += cubes
        elif winner == "Enemy":
            enemy_wins += 1
            cubes_net -= cubes
        else:
            ties += 1

    total = float(number_of_games)
    ally_win_rate = ally_wins / total
    average_cubes_won = cubes_net / total

    return ally_win_rate, average_cubes_won


def evaluate_6_tests_to_csv_snap_agent(
    player_q_network,
    snap_q_network,
    csv_path: str,
    number_of_games: int = 10000,
    epsilon_snap: float = 0.0,
    verbose: bool = False,
):
    csv_file, csv_writer = create_snap_eval_csv_writer(csv_path)

    opponent_type_list = ["Random", "Greedy"]
    agent_deck_type_list = [0, 1, 2]
    enemy_deck_type = 1

    for opponent_type in opponent_type_list:
        for agent_deck_type in agent_deck_type_list:
            ally_win_rate, average_cubes_won = evaluate_snap_agent_against_opponent(
                player_q_network=player_q_network,
                snap_q_network=snap_q_network,
                number_of_games=number_of_games,
                opponent_type=opponent_type,
                agent_deck_type=agent_deck_type,
                epsilon_snap=epsilon_snap
            )

            write_snap_eval_csv_row(
                csv_writer,
                opponent_type=opponent_type,
                agent_deck_type=agent_deck_type,
                enemy_deck_type=enemy_deck_type,
                number_of_games=number_of_games,
                epsilon_snap=epsilon_snap,
                ally_win_rate=ally_win_rate,
                average_cubes_won=average_cubes_won,
            )
            csv_file.flush()

    csv_file.close()


if __name__ == "__main__":
    snap_input_dimension = get_snap_input_dimension()
    input_dimension = get_input_dimension()

    player_q_network = load_q_network(
        f"trained_q_network_DMC_2000000_episodes.pt",
        input_dimension=input_dimension,
        architecture="B",
    )

    snap_q_network = load_q_network(
        f"trained_q_network_snap_DMC_1500010_episodes.pt",
        input_dimension=snap_input_dimension,
        architecture="B",
    )

    player_q_network.eval()
    snap_q_network.eval()

    evaluate_6_tests_to_csv_snap_agent(
        player_q_network=player_q_network,
        snap_q_network=snap_q_network,
        csv_path="model_evaluation_SNAP_AGENT.csv",
        number_of_games=10000,
        epsilon_snap=0.0,
        verbose=False,
    )
