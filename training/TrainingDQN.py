import time
import torch.nn as nn
import torch.optim as optim
from training.TrainingUtilityFunctions import *
from training.TrainingNetwork import QNetwork, load_q_network, save_q_network


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
            self.write_index += 1
            if self.write_index >= self.maximum_size:
                self.write_index = 0

    def sample_batch(self, batch_size):
        batch_size = int(batch_size)
        return random.sample(self.transition_list, batch_size)


def _build_next_state_action_array(game_state, is_ally):
    legal_actions_list = get_legal_actions(game_state, is_ally)

    state_action_vector_list = []
    for action_tuple in legal_actions_list:
        v = build_observation_with_chosen_action(game_state, is_ally, action_tuple)
        state_action_vector_list.append(v)

    if len(state_action_vector_list) <= 0:
        return None

    return numpy.stack(state_action_vector_list).astype(numpy.float32)


def generate_episode_transitions(environment, q_network, epsilon, enemy_type):
    environment.reset()

    transition_dict_list = []

    final_reward_ally = 0.0
    final_reward_enemy = 0.0

    done = False
    is_ally = True

    pending_transition_ally = None
    pending_transition_enemy = None

    last_ally_transition_in_turn = None
    last_enemy_transition_in_turn = None

    previous_turn_counter = int(environment.game_state.status["turncounter"])

    while not done:
        if is_ally:
            if pending_transition_ally is not None:
                pending_transition_ally["next_state_action_array"] = _build_next_state_action_array(
                    environment.game_state, True
                )
                pending_transition_ally["done"] = False
                pending_transition_ally = None

            action, state_action_vector = choose_action_epsilon_greedy(
                environment, q_network, True, epsilon
            )

            new_transition = {
                "state_action_vector": numpy.array(state_action_vector, dtype=numpy.float32),
                "reward": 0.0,
                "next_state_action_array": None,
                "done": False,
                "player_is_ally": True,
            }
            transition_dict_list.append(new_transition)
            pending_transition_ally = new_transition
            last_ally_transition_in_turn = new_transition

            action_type, done = environment.step(action, True)

            if action_type == "Passed":
                is_ally = False

        else:
            if enemy_type == "Self-Play":
                if pending_transition_enemy is not None:
                    pending_transition_enemy["next_state_action_array"] = _build_next_state_action_array(
                        environment.game_state, False
                    )
                    pending_transition_enemy["done"] = False
                    pending_transition_enemy = None

                action, state_action_vector = choose_action_epsilon_greedy(
                    environment, q_network, False, epsilon
                )

                new_transition = {
                    "state_action_vector": numpy.array(state_action_vector, dtype=numpy.float32),
                    "reward": 0.0,
                    "next_state_action_array": None,
                    "done": False,
                    "player_is_ally": False,
                }
                transition_dict_list.append(new_transition)
                pending_transition_enemy = new_transition
                last_enemy_transition_in_turn = new_transition

                action_type, done = environment.step(action, False)

                if action_type == "Passed":
                    is_ally = True
            else:
                action = choose_random_action(environment, False)
                action_type, done = environment.step(action, False)
                if action_type == "Passed":
                    is_ally = True

        current_turn_counter = int(environment.game_state.status["turncounter"])
        if current_turn_counter != previous_turn_counter:
            end_of_turn_reward = 0.25 * float(previous_turn_counter) * compute_end_of_turn_location_reward(
                environment.game_state
            )

            if last_ally_transition_in_turn is not None:
                last_ally_transition_in_turn["reward"] += end_of_turn_reward

            if last_enemy_transition_in_turn is not None:
                last_enemy_transition_in_turn["reward"] -= end_of_turn_reward

            last_ally_transition_in_turn = None
            last_enemy_transition_in_turn = None
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

            if last_ally_transition_in_turn is not None:
                last_ally_transition_in_turn["reward"] += final_reward_ally
            else:
                if pending_transition_ally is not None:
                    pending_transition_ally["reward"] += final_reward_ally

            if last_enemy_transition_in_turn is not None:
                last_enemy_transition_in_turn["reward"] += final_reward_enemy
            else:
                if pending_transition_enemy is not None:
                    pending_transition_enemy["reward"] += final_reward_enemy

            if pending_transition_ally is not None:
                pending_transition_ally["next_state_action_array"] = None
                pending_transition_ally["done"] = True
                pending_transition_ally = None

            if pending_transition_enemy is not None:
                pending_transition_enemy["next_state_action_array"] = None
                pending_transition_enemy["done"] = True
                pending_transition_enemy = None

    transition_tuple_list = []
    for t in transition_dict_list:
        transition_tuple_list.append(
            (
                t["state_action_vector"],
                float(t["reward"]),
                t["next_state_action_array"],
                bool(t["done"]),
            )
        )

    return transition_tuple_list, final_reward_ally, final_reward_enemy


def _compute_dqn_batch_loss(q_network, target_network, batch_transition_list, discount_factor):
    if batch_transition_list is None or len(batch_transition_list) <= 0:
        return None

    state_action_list = []
    reward_list = []
    done_list = []

    next_arrays = []
    next_owner_index = []
    sample_index = 0

    for (state_action_vector, reward_value, next_state_action_array, done_flag) in batch_transition_list:
        state_action_list.append(state_action_vector)
        reward_list.append(float(reward_value))
        done_list.append(bool(done_flag))

        if (not done_flag) and (next_state_action_array is not None) and (len(next_state_action_array) > 0):
            next_arrays.append(next_state_action_array)
            next_owner_index.append(sample_index)

        sample_index += 1

    state_action_array = numpy.stack(state_action_list).astype(numpy.float32)
    state_action_tensor = torch.from_numpy(state_action_array)

    reward_tensor = torch.from_numpy(numpy.array(reward_list, dtype=numpy.float32))
    done_mask_tensor = torch.from_numpy(numpy.array(done_list, dtype=numpy.float32))

    predicted_q_tensor = q_network(state_action_tensor).view(-1)

    next_max_q_tensor = torch.zeros((len(batch_transition_list),), dtype=torch.float32)

    if len(next_arrays) > 0:
        concatenated_next = numpy.concatenate(next_arrays, axis=0).astype(numpy.float32)
        concatenated_next_tensor = torch.from_numpy(concatenated_next)

        with torch.no_grad():
            concatenated_q = target_network(concatenated_next_tensor).view(-1)

        start_index = 0
        block_index = 0
        while block_index < len(next_arrays):
            block = next_arrays[block_index]
            block_len = int(block.shape[0])
            end_index = start_index + block_len

            owner = int(next_owner_index[block_index])
            block_max = torch.max(concatenated_q[start_index:end_index])
            next_max_q_tensor[owner] = block_max

            start_index = end_index
            block_index += 1

    target_q_tensor = reward_tensor + (1.0 - done_mask_tensor) * float(discount_factor) * next_max_q_tensor
    loss_function = nn.MSELoss()
    loss_value = loss_function(predicted_q_tensor, target_q_tensor)

    return loss_value


def train_dqn_with_logging(
    number_of_episodes,
    learning_rate=1e-4,
    epsilon_start=0.9,
    epsilon_end=0.05,
    seed_value=None,
    evaluation_interval=100,
    evaluation_games=50,
    log_csv_path=None,
    save_model_path=None,
    replay_buffer_size=200000,
    batch_size=64,
    discount_factor=1.0,
    updates_per_episode=1,
    target_update_interval=2000,
    gradient_clip_norm=1.0,
):
    if seed_value is not None:
        set_global_seed(seed_value)

    start_time_seconds = time.time()

    input_dimension = get_input_dimension()
    q_network = QNetwork(input_dimension)
    target_network = QNetwork(input_dimension)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

    environment = SingleAgentTestEnvironment()

    replay_buffer = ReplayBuffer(replay_buffer_size)

    training_loss_history = []
    ally_win_rate_history = []
    enemy_win_rate_history = []
    tie_rate_history = []

    csv_file = None
    csv_writer = None
    if log_csv_path is not None:
        csv_file, csv_writer = create_training_csv_writer(log_csv_path)

    episode_index = 0
    gradient_step_counter = 0

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

        transition_tuple_list, final_reward_ally, final_reward_enemy = generate_episode_transitions(
            environment, q_network, epsilon, enemy_type
        )

        for transition_tuple in transition_tuple_list:
            replay_buffer.add_transition(transition_tuple)

        loss_value_float = 0.0
        if len(replay_buffer) >= int(batch_size):
            update_index = 0
            while update_index < int(updates_per_episode):
                batch_transition_list = replay_buffer.sample_batch(batch_size)
                loss_value = _compute_dqn_batch_loss(
                    q_network, target_network, batch_transition_list, discount_factor
                )
                if loss_value is not None:
                    optimizer.zero_grad()
                    loss_value.backward()
                    if gradient_clip_norm is not None and gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(q_network.parameters(), float(gradient_clip_norm))
                    optimizer.step()

                    loss_value_float = float(loss_value.item())
                    training_loss_history.append(loss_value_float)

                    gradient_step_counter += 1
                    if target_update_interval is not None and target_update_interval > 0:
                        if gradient_step_counter % int(target_update_interval) == 0:
                            target_network.load_state_dict(q_network.state_dict())
                            target_network.eval()

                update_index += 1

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


if __name__ == "__main__":
    number_of_episodes = 210000

    results = train_dqn_with_logging(
        number_of_episodes=number_of_episodes,
        learning_rate=3e-4,
        epsilon_start=0.9,
        epsilon_end=0.05,
        seed_value=59,
        evaluation_interval=10000,
        evaluation_games=2000,
        log_csv_path=f"training_log_DQN_{number_of_episodes}_episodes.csv",
        save_model_path=f"trained_q_network_DQN_{number_of_episodes}_episodes.pt",
        replay_buffer_size=10000,
        batch_size=32,
        discount_factor=0.99,
        updates_per_episode=3,
        target_update_interval=5000,
        gradient_clip_norm=1.0,
    )

    trained_q_network = results["q_network"]
    print("Training finished.")

    input_dimension = trained_q_network.linear_layer_1.in_features

    loaded_q_network = load_q_network(
        f"trained_q_network_DQN_{number_of_episodes}_episodes.pt",
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
