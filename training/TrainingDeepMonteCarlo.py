import random
import torch
import torch.nn as nn
import torch.optim as optim
from environment.SingleAgentTestEnvironment import SingleAgentTestEnvironment
from environment.TestUtilityFunctions import *


def set_global_seed(seed_value):
    random.seed(seed_value)
    numpy.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


class QNetwork(nn.Module):
    def __init__(self, input_dimension, hidden_dimension=512):
        super(QNetwork, self).__init__()
        self.linear_layer_1 = nn.Linear(input_dimension, hidden_dimension)
        self.linear_layer_2 = nn.Linear(hidden_dimension, hidden_dimension)
        self.linear_layer_3 = nn.Linear(hidden_dimension, 1)

    def forward(self, state_action_tensor):
        x = torch.relu(self.linear_layer_1(state_action_tensor))
        x = torch.relu(self.linear_layer_2(x))
        x = self.linear_layer_3(x)
        return x.squeeze(-1)


def save_q_network(q_network, file_path):
    torch.save(q_network.state_dict(), file_path)


def load_q_network(file_path, input_dimension, hidden_dimension=512):
    q_network = QNetwork(input_dimension, hidden_dimension)
    state_dictionary = torch.load(file_path, map_location=torch.device("cpu"))
    q_network.load_state_dict(state_dictionary)
    q_network.eval()
    return q_network


def choose_random_action(environment, is_ally):
    game_state = environment.game_state
    legal_actions_list = get_legal_actions(game_state, is_ally)
    return random.choice(legal_actions_list)


def choose_action_epsilon_greedy(environment, q_network, is_ally, epsilon):
    game_state = environment.game_state
    legal_actions_list = get_legal_actions(game_state, is_ally)
    if random.random() < epsilon:
        chosen_action = random.choice(legal_actions_list)
        state_action_vector = build_observation_with_chosen_action(game_state, is_ally, chosen_action)
        return chosen_action, numpy.array(state_action_vector, dtype=numpy.float32)

    best_q_value = None
    best_action = None
    best_state_action_vector = None

    for action_tuple in legal_actions_list:
        state_action_vector = build_observation_with_chosen_action(game_state, is_ally, action_tuple)
        state_action_tensor = torch.from_numpy(numpy.array(state_action_vector, dtype=numpy.float32)).unsqueeze(0)
        with torch.no_grad():
            q_value_tensor = q_network(state_action_tensor)
        q_value = float(q_value_tensor.item())
        if best_q_value is None or q_value > best_q_value:
            best_q_value = q_value
            best_action = action_tuple
            best_state_action_vector = state_action_vector

    return best_action, numpy.array(best_state_action_vector, dtype=numpy.float32)


def generate_episode(environment, q_network, epsilon):
    environment.reset()
    episode_ally_state_actions = []
    episode_enemy_state_actions = []
    final_reward_ally = 0.0
    final_reward_enemy = 0.0
    done = False
    is_ally = True

    while not done:
        action, state_action_vector = choose_action_epsilon_greedy(environment, q_network, is_ally, epsilon)
        action_type, done = environment.step(action, is_ally)

        if is_ally:
            episode_ally_state_actions.append(state_action_vector)
            if action_type == "Passed":
                is_ally = False
        else:
            episode_enemy_state_actions.append(state_action_vector)
            if action_type == "Passed":
                is_ally = True
        if done:
            winner = environment.game_state.passStatus["winner"]
            if winner == "Ally":
                final_reward_ally = 1.0
                final_reward_enemy = -1.0
            elif winner == "Enemy":
                final_reward_ally = -1.0
                final_reward_enemy = 1.0
            else:
                final_reward_ally = 0.0
                final_reward_enemy = 0.0

    return episode_ally_state_actions, episode_enemy_state_actions, final_reward_ally, final_reward_enemy


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

    dummy_environment = SingleAgentTestEnvironment()
    dummy_environment.reset()
    dummy_action = (-1, -1)
    dummy_state_action_vector = build_observation_with_chosen_action(
        dummy_environment.game_state, True, dummy_action
    )
    input_dimension = len(dummy_state_action_vector)

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
        import csv
        csv_file = open(log_csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "episode",
                "epsilon",
                "loss",
                "final_reward_ally",
                "final_reward_enemy",
                "ally_win_rate",
                "enemy_win_rate",
                "tie_rate",
            ]
        )

    episode_index = 0
    while episode_index < number_of_episodes:
        if number_of_episodes > 1:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * (
                1.0 - float(episode_index) / float(number_of_episodes - 1)
            )
        else:
            epsilon = epsilon_start

        (
            episode_ally_state_actions,
            episode_enemy_state_actions,
            final_reward_ally,
            final_reward_enemy,
        ) = generate_episode(environment, q_network, epsilon)

        training_state_action_list = []
        target_return_list = []

        for episode_actions, final_reward in [
            (episode_ally_state_actions, final_reward_ally),
            (episode_enemy_state_actions, final_reward_enemy),
        ]:
            for state_action_vector in episode_actions:
                training_state_action_list.append(state_action_vector)
                target_return_list.append(final_reward)

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

        if evaluation_interval is not None and evaluation_interval > 0:
            if (episode_index + 1) % evaluation_interval == 0:
                eval_results = evaluate_against_random_opponent(
                    q_network, evaluation_games, epsilon_agent=0.0
                )
                ally_win_rate = eval_results["ally_win_rate"]
                enemy_win_rate = eval_results["enemy_win_rate"]
                tie_rate = eval_results["tie_rate"]

                ally_win_rate_history.append(ally_win_rate)
                enemy_win_rate_history.append(enemy_win_rate)
                tie_rate_history.append(tie_rate)

                print(
                    "Episode:",
                    episode_index + 1,
                    "Loss:",
                    loss_value_float,
                    "Epsilon:",
                    epsilon,
                    "Ally win rate vs random:",
                    ally_win_rate,
                    "Enemy win rate:",
                    enemy_win_rate,
                    "Tie rate:",
                    tie_rate,
                )

        if csv_writer is not None:
            csv_writer.writerow(
                [
                    episode_index + 1,
                    epsilon,
                    loss_value_float,
                    final_reward_ally,
                    final_reward_enemy,
                    ally_win_rate,
                    enemy_win_rate,
                    tie_rate,
                ]
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


def evaluate_against_random_opponent(q_network, number_of_games, epsilon_agent=0.0):
    environment = SingleAgentTestEnvironment(True)

    ally_wins = 0
    enemy_wins = 0
    ties = 0

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

        winner = environment.game_state.passStatus["winner"]
        if winner == "Ally":
            ally_wins += 1
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
    return results_dictionary


if __name__ == "__main__":
    results = train_deep_monte_carlo_with_logging(
        number_of_episodes=40000,
        learning_rate=1e-4,
        epsilon_start=0.9,
        epsilon_end=0.05,
        seed_value=42,
        evaluation_interval=1000,
        evaluation_games=100,
        log_csv_path="training_log.csv",
        save_model_path="trained_q_network.pt",
    )

    trained_q_network = results["q_network"]
    print("Training finished.")

    input_dimension = trained_q_network.linear_layer_1.in_features

    loaded_q_network = load_q_network(
        "trained_q_network.pt",
        input_dimension=input_dimension,
        hidden_dimension=512,
    )

    final_eval_results = evaluate_against_random_opponent(
        loaded_q_network,
        number_of_games=2000,
        epsilon_agent=0.0,
    )
    print("Final evaluation vs random opponent:", final_eval_results)
