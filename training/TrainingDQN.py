import random
import torch
import torch.nn as nn
import torch.optim as optim
from environment.TestUtilityFunctions import *
import time
from training.TrainingUtilityFunctions import *
from training.TrainingNetwork import QNetwork, load_q_network, save_q_network
from training.TrainingDeepMonteCarlo import choose_action_epsilon_greedy, choose_random_action, evaluate_against_random_opponent


def set_global_seed(seed_value):
    random.seed(seed_value)
    numpy.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


class ReplayBuffer:
    def __init__(self, maximum_size):
        self.maximum_size = int(maximum_size)
        self.transition_list = []
        self.write_index = 0

    def __len__(self):
        return len(self.transition_list)

    def add_transition(self, transition_tuple):
        if len(self.transition_list) < self.maximum_size:
            self.transition_list.append(transition_tuple)
        else:
            self.transition_list[self.write_index] = transition_tuple
        self.write_index = (self.write_index + 1) % self.maximum_size

    def sample_batch(self, batch_size):
        batch_size = int(batch_size)
        return random.sample(self.transition_list, batch_size)


def compute_terminal_rewards(game_state):
    winner = game_state.passStatus["winner"]
    if winner == "Ally":
        return 1.0, -1.0
    if winner == "Enemy":
        return -1.0, 1.0
    return 0.0, 0.0


def compute_max_next_q_value(q_network_target, game_state, is_ally):
    legal_actions_list = get_legal_actions(game_state, is_ally)
    if len(legal_actions_list) == 0:
        return 0.0

    state_action_vector_list = []
    for action_tuple in legal_actions_list:
        state_action_vector = build_observation_with_chosen_action(
            game_state, is_ally, action_tuple
        )
        state_action_vector_list.append(state_action_vector)

    state_action_array = numpy.stack(state_action_vector_list).astype(numpy.float32)
    state_action_tensor = torch.from_numpy(state_action_array)

    with torch.no_grad():
        q_value_tensor = q_network_target(state_action_tensor)

    return float(torch.max(q_value_tensor).item())


def choose_enemy_action(environment, q_network, enemy_type, epsilon):
    if enemy_type == "Self-Play":
        action, _ = choose_action_epsilon_greedy(environment, q_network, False, epsilon)
        return action
    return choose_random_action(environment, False)


def step_from_ally_perspective(environment, q_network, epsilon, enemy_type, use_end_of_turn_reward):
    game_state_before = copy.deepcopy(environment.game_state)
    action, state_action_vector = choose_action_epsilon_greedy(environment, q_network, True, epsilon)

    previous_turn_counter = int(environment.game_state.status["turncounter"])
    reward_value = 0.0

    action_type, done = environment.step(action, True)
    is_ally = True
    if action_type == "Passed":
        is_ally = False

    current_turn_counter = int(environment.game_state.status["turncounter"])
    if use_end_of_turn_reward and current_turn_counter != previous_turn_counter:
        reward_value += compute_end_of_turn_location_reward(environment.game_state)
        previous_turn_counter = current_turn_counter

    while not done and not is_ally:
        enemy_action = choose_enemy_action(environment, q_network, enemy_type, epsilon)
        action_type, done = environment.step(enemy_action, False)
        if action_type == "Passed":
            is_ally = True

        current_turn_counter = int(environment.game_state.status["turncounter"])
        if use_end_of_turn_reward and current_turn_counter != previous_turn_counter:
            reward_value += compute_end_of_turn_location_reward(environment.game_state)
            previous_turn_counter = current_turn_counter

    if done:
        final_reward_ally, final_reward_enemy = compute_terminal_rewards(environment.game_state)
        reward_value += final_reward_ally

    next_game_state = copy.deepcopy(environment.game_state)
    return (
        numpy.array(state_action_vector, dtype=numpy.float32),
        float(reward_value),
        next_game_state,
        bool(done),
    )


def train_deep_q_learning_with_logging(
    number_of_episodes,
    learning_rate=1e-4,
    discount_factor=0.99,
    epsilon_start=0.9,
    epsilon_end=0.05,
    seed_value=None,
    evaluation_interval=100,
    evaluation_games=500,
    replay_buffer_size=200000,
    batch_size=256,
    target_update_interval=2000,
    updates_per_episode=1,
    enemy_type="Random",
    use_end_of_turn_reward=True,
    log_csv_path=None,
    save_model_path=None,
):
    if seed_value is not None:
        set_global_seed(seed_value)
    start_time_seconds = time.time()

    input_dimension = get_input_dimension()
    q_network = QNetwork(input_dimension)
    q_network_target = QNetwork(input_dimension)
    q_network_target.load_state_dict(q_network.state_dict())
    q_network_target.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    replay_buffer = ReplayBuffer(replay_buffer_size)
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
    update_step_counter = 0

    while episode_index < number_of_episodes:
        if number_of_episodes > 1:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * (
                1.0 - float(episode_index) / float(number_of_episodes - 1)
            )
        else:
            epsilon = epsilon_start

        environment.reset()
        done = False

        while not done:
            state_action_vector, reward_value, next_game_state, done = step_from_ally_perspective(
                environment,
                q_network,
                epsilon,
                enemy_type,
                use_end_of_turn_reward,
            )
            replay_buffer.add_transition((state_action_vector, reward_value, next_game_state, done))

        loss_value_float = 0.0

        update_iteration = 0
        while update_iteration < updates_per_episode:
            if len(replay_buffer) >= batch_size:
                batch_transition_list = replay_buffer.sample_batch(batch_size)

                batch_state_action_list = []
                batch_target_list = []

                for state_action_vector, reward_value, next_game_state, transition_done in batch_transition_list:
                    if transition_done:
                        target_value = float(reward_value)
                    else:
                        max_next_q_value = compute_max_next_q_value(q_network_target, next_game_state, True)
                        target_value = float(reward_value) + float(discount_factor) * float(max_next_q_value)

                    batch_state_action_list.append(state_action_vector)
                    batch_target_list.append(target_value)

                state_action_array = numpy.stack(batch_state_action_list).astype(numpy.float32)
                target_array = numpy.array(batch_target_list, dtype=numpy.float32)

                state_action_tensor = torch.from_numpy(state_action_array)
                target_tensor = torch.from_numpy(target_array)

                predicted_q_tensor = q_network(state_action_tensor)
                loss_value = loss_function(predicted_q_tensor, target_tensor)

                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()

                loss_value_float = float(loss_value.item())
                training_loss_history.append(loss_value_float)

                update_step_counter += 1
                if target_update_interval is not None and target_update_interval > 0:
                    if update_step_counter % target_update_interval == 0:
                        q_network_target.load_state_dict(q_network.state_dict())
                        q_network_target.eval()

            update_iteration += 1

        if evaluation_interval is not None and evaluation_interval > 0:
            if (episode_index + 1) % evaluation_interval == 0:
                eval_results = evaluate_against_random_opponent(
                    q_network, evaluation_games, epsilon_agent=0.0
                )
                ally_win_rate = eval_results["ally_win_rate"]
                ally_win_rate_history.append(ally_win_rate)

                elapsed_minutes = (time.time() - start_time_seconds) / 60.0

                if csv_writer is not None:
                    csv_writer.writerow(
                        [
                            episode_index + 1,
                            elapsed_minutes,
                            epsilon,
                            loss_value_float,
                            eval_results["ally_win_rate"],
                            eval_results["enemy_win_rate"],
                            eval_results["tie_rate"],
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
    }
    return results_dictionary
