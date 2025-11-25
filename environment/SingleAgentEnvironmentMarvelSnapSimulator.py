import gymnasium as gym
from gymnasium import spaces
import numpy
import Decks


class MarvelSnapSingleAgentEnv(gym.Env):

    def __init__(self, game_state):
        super().__init__()
        self.game_state = game_state
        self.card_pool_list = Decks.ALL_CARDS
        self.number_of_cards = len(self.card_pool_list)
        self.ACTION_SPACE_LENGTH = 3 * self.number_of_cards + 1
        self.observation_space = spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(9 + self.ACTION_SPACE_LENGTH,),
            dtype=numpy.float32
        )
        self.action_space = spaces.Discrete(self.ACTION_SPACE_LENGTH)

    def get_observation_array(self):
        status_dictionary = self.game_state.status
        location_dictionary = self.game_state.locationList
        feature_list = []
        for location_key in ["location1", "location2", "location3"]:
            location = location_dictionary[location_key]
            feature_list.append(float(location.alliesPower) / 10.0)
            feature_list.append(float(location.enemiesPower) / 10.0)
        feature_list.append(float(status_dictionary["allyenergy"]) / 10.0)
        feature_list.append(float(status_dictionary["turncounter"]) / 7.0)
        feature_list.append(1.0 if status_dictionary["allypriority"] else 0.0)

        features = numpy.array(feature_list, dtype=numpy.float32)
        action_mask = self.build_action_mask(True).astype(numpy.float32)
        return numpy.concatenate([features, action_mask], axis=0)

    def decode_action_index(self, action_index):
        card_index = action_index // 3
        location_index = action_index % 3
        card = self.card_pool_list[card_index]
        location_number = location_index + 1
        return card, location_number

    def can_play(self, ally: bool, card, location):
        if ally:
            energy = self.game_state.status["allyenergy"]
            energy_check = card.cur_cost <= energy
            location_check = (not location.checkIfLocationFull(ally)) and location.can_play_cards_allies
            unit_check = location.canCardBePlayed(card)
            return energy_check and location_check and unit_check
        else:
            energy = self.game_state.status["enemyenergy"]
            energy_check = card.cur_cost <= energy
            location_check = (not location.checkIfLocationFull(ally)) and location.can_play_cards_enemies
            unit_check = location.canCardBePlayed(card)
            return energy_check and location_check and unit_check

    def build_action_mask(self, is_ally):
        action_mask = numpy.zeros(self.ACTION_SPACE_LENGTH, dtype=numpy.int8)
        hand = self.game_state.status["allyhand"] if is_ally else self.game_state.status["enemyhand"]
        location_dictionary = self.game_state.locationList

        for card in hand:
            card_index = Decks.CLASS_TO_INDEX[card.__class__]
            for location_index, location_key in enumerate(["location1", "location2", "location3"]):
                location = location_dictionary[location_key]
                if self.can_play(is_ally, card, location):
                    action_index = (card_index * 3) + location_index
                    action_mask[action_index] = 1
        action_mask[self.ACTION_SPACE_LENGTH - 1] = 1
        return action_mask

    def play_enemy(self):
        while not self.game_state.status["enemypass"]:
            enemy_action_mask = self.build_action_mask(False)
            valid_actions = numpy.flatnonzero(enemy_action_mask)
            random_valid_action = numpy.random.choice(valid_actions)
            if random_valid_action == (self.ACTION_SPACE_LENGTH - 1):
                self.game_state.status["enemypass"] = True
                break

            card_type, location_number = self.decode_action_index(random_valid_action)
            for card_index, card in enumerate(self.game_state.status["enemyhand"]):
                if isinstance(card, card_type):
                    if not self.game_state.addUnit(card_index, False, location_number):
                        continue
                    break

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game_state.reset()
        observation_array = self.get_observation_array()
        info_dictionary = {}
        return observation_array, info_dictionary

    def step(self, action):
        terminated_flag = False
        truncated_flag = False
        reward = 0.0
        integer_action = int(action)
        self.play_enemy()
        if self.build_action_mask(True)[integer_action] == 1:
            if integer_action == (self.ACTION_SPACE_LENGTH - 1):
                self.game_state.status["allypass"] = True
            else:
                card_type, location_number = self.decode_action_index(integer_action)
                for card_index, card in enumerate(self.game_state.status["allyhand"]):
                    if isinstance(card, card_type):
                        if not self.game_state.addUnit(card_index, True, location_number):
                            continue
                        reward += 3.0
                        reward += card.cur_power / 2.0
                        break
        else:
            reward -= 1.0
            self.game_state.status["allypass"] = True

        if self.game_state.status["allypass"] and self.game_state.status["enemypass"]:
            reward -= self.game_state.status["allyenergy"]
            self.game_state.turnEnd(True)
            if self.game_state.game_end:
                for location in self.game_state.locationList.values():
                    if location.alliesPower > location.enemiesPower:
                        reward += 1.0
                    elif location.alliesPower < location.enemiesPower:
                        reward -= 1.0
                terminated_flag = True

        observation_array = self.get_observation_array()
        info_dictionary = {}
        return observation_array, reward, terminated_flag, truncated_flag, info_dictionary
