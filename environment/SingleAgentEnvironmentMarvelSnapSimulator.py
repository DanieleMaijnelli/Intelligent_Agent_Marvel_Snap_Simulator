import gymnasium as gym
from gymnasium import spaces
import numpy
from gameManager import GameState
import Decks


class MarvelSnapSingleAgentEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.game_state = GameState(verbose=False)
        self.card_pool_list = Decks.ALL_CARDS
        self.number_of_cards = len(self.card_pool_list)
        self.observation_space = spaces.Box(
            low=-200.0,
            high=200.0,
            shape=(15,),
            dtype=numpy.float32
        )
        self.action_space = spaces.Discrete(3 * self.number_of_cards)

    def get_observation_array(self):
        status_dictionary = self.game_state.status
        location_dictionary = self.game_state.locationList
        feature_list = []
        for location_key in ["location1", "location2", "location3"]:
            location_object = location_dictionary[location_key]
            feature_list.append(float(location_object.alliesPower) / 10.0)
            feature_list.append(float(location_object.enemiesPower) / 10.0)
            feature_list.append(float(len(location_object.allies)) / 4.0)
            feature_list.append(float(len(location_object.enemies)) / 4.0)
        feature_list.append(float(status_dictionary["allyenergy"]) / 10.0)
        feature_list.append(float(status_dictionary["turncounter"]) / 7.0)
        feature_list.append(1.0 if status_dictionary["allypriority"] else 0.0)
        return numpy.array(feature_list, dtype=numpy.float32)

    def decode_action_index(self, action_index):
        card_index = action_index // 3
        location_index = action_index % 3
        card_identifier = self.card_pool_list[card_index]
        location_number = location_index + 1
        return card_identifier, location_number

    def find_card_index_in_hand(self, card_identifier, is_ally):
        hand_list = self.game_state.status["allyhand"] if is_ally else self.game_state.status["enemyhand"]
        for card_index, card_object in enumerate(hand_list):
            if getattr(card_object, "name", None) == card_identifier:
                return card_index
        return None

    def compute_total_ally_power(self):
        location_dictionary = self.game_state.locationList
        total_power_value = 0.0
        for location_key in ["location1", "location2", "location3"]:
            location_object = location_dictionary[location_key]
            total_power_value += float(location_object.alliesPower)
        return total_power_value

    def is_action_valid_for_ally(self, action_index):
        card_identifier, location_number = self.decode_action_index(action_index)
        index_in_hand = self.find_card_index_in_hand(card_identifier, is_ally=True)
        if index_in_hand is None:
            return False
        hand_list = self.game_state.status["allyhand"]
        card_object = hand_list[index_in_hand]
        if self.game_state.status["allyenergy"] < card_object.cur_cost:
            return False
        location_key = f"location{location_number}"
        location_object = self.game_state.locationList[location_key]
        if hasattr(location_object, "max_units"):
            if len(location_object.allies) >= location_object.max_units:
                return False
        return True

    def apply_action_for_ally(self, action_index):
        if not self.is_action_valid_for_ally(action_index):
            return False
        card_identifier, location_number = self.decode_action_index(action_index)
        index_in_hand = self.find_card_index_in_hand(card_identifier, is_ally=True)
        if index_in_hand is None:
            return False
        self.game_state.addUnit(index_in_hand, ally=True, locNum=location_number)
        return True

    def valid_enemy_moves_list(self):
        move_list = []
        hand_list = self.game_state.status["enemyhand"]
        enemy_energy_value = self.game_state.status["enemyenergy"]
        for hand_index, card_object in enumerate(hand_list):
            if enemy_energy_value < card_object.cur_cost:
                continue
            for location_number in [1, 2, 3]:
                location_key = f"location{location_number}"
                location_object = self.game_state.locationList[location_key]
                if hasattr(location_object, "max_units"):
                    if len(location_object.enemies) >= location_object.max_units:
                        continue
                move_list.append((hand_index, location_number))
        return move_list

    def apply_random_enemy_move(self):
        move_list = self.valid_enemy_moves_list()
        if not move_list:
            return False
        random_index = numpy.random.randint(len(move_list))
        hand_index, location_number = move_list[random_index]
        self.game_state.addUnit(hand_index, ally=False, locNum=location_number)
        return True

    def build_action_mask_for_ally(self):
        mask_array = numpy.zeros(self.action_space.n, dtype=numpy.int8)
        for action_index in range(self.action_space.n):
            if self.is_action_valid_for_ally(action_index):
                mask_array[action_index] = 1
        return mask_array

    def compute_reward_value(self, energy_spent_value, power_gained_value):
        location_dictionary = self.game_state.locationList
        number_of_locations_won = 0
        for location_key in ["location1", "location2", "location3"]:
            location_object = location_dictionary[location_key]
            if getattr(location_object, "winning", None) == "Ally":
                number_of_locations_won += 1
        location_bonus_value = 0.1 * float(number_of_locations_won)
        reward_value = float(energy_spent_value) + float(power_gained_value) + location_bonus_value
        return reward_value

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game_state = GameState(verbose=False)
        observation_array = self.get_observation_array()
        info_dictionary = {
            "action_mask": self.build_action_mask_for_ally()
        }
        return observation_array, info_dictionary

    def step(self, action):
        integer_action = int(action)
        ally_energy_before = float(self.game_state.status["allyenergy"])
        ally_power_before = self.compute_total_ally_power()
        self.apply_action_for_ally(integer_action)
        self.apply_random_enemy_move()
        ally_energy_after = float(self.game_state.status["allyenergy"])
        ally_power_after = self.compute_total_ally_power()
        energy_spent_value = max(0.0, ally_energy_before - ally_energy_after)
        power_gained_value = max(0.0, ally_power_after - ally_power_before)
        self.game_state.endOfTurn()
        terminated_flag = self.game_state.game_end
        truncated_flag = False
        if not terminated_flag:
            self.game_state.startOfTurn()
        observation_array = self.get_observation_array()
        reward_value = self.compute_reward_value(energy_spent_value, power_gained_value)
        info_dictionary = {
            "action_mask": self.build_action_mask_for_ally()
        }
        return observation_array, reward_value, terminated_flag, truncated_flag, info_dictionary
