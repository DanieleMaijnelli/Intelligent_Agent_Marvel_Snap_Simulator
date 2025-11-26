import gymnasium as gym
from gymnasium import spaces
from UtilityFunctions import *


class MarvelSnapSingleSnapAgentEnv(gym.Env):

    def __init__(self, game_state):
        super().__init__()
        self.game_state = game_state
        self.card_pool_list = Decks.ALL_CARDS
        self.number_of_cards = len(self.card_pool_list)
        self.PLAYERS_ACTION_SPACE_LENGTH = 3 * self.number_of_cards + 1
        self.observation_space = spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(17 + self.number_of_cards,),
            dtype=numpy.float32
        )
        self.action_space = spaces.Discrete(3)

    def get_observation_array(self):
        status_dictionary = self.game_state.status
        location_dictionary = self.game_state.locationList
        feature_list = []
        for location_key in ["location1", "location2", "location3"]:
            location = location_dictionary[location_key]
            feature_list.append(float(location.alliesPower) / 10.0)
            feature_list.append(float(location.enemiesPower) / 10.0)
            feature_list.append(float(len(location.allies)) / 4.0)
            feature_list.append(float(len(location.enemies)) / 4.0)
        feature_list.append(float(status_dictionary["allyenergy"]) / 10.0)
        feature_list.append(float(status_dictionary["turncounter"]) / 7.0)
        feature_list.append(1.0 if status_dictionary["allypriority"] else 0.0)
        feature_list.append(float(status_dictionary["cubes"]) / 8.0)
        feature_list.append(float(status_dictionary["tempcubes"]) / 8.0)

        features = numpy.array(feature_list, dtype=numpy.float32)
        action_mask = self.build_owned_cards_vector(True).astype(numpy.float32)
        return numpy.concatenate([features, action_mask], axis=0)

    def build_owned_cards_vector(self, is_ally):
        owned_cards_vector = numpy.zeros(self.number_of_cards, dtype=numpy.int8)
        hand = self.game_state.status["allyhand"] if is_ally else self.game_state.status["enemyhand"]
        for card in hand:
            card_index = Decks.CLASS_TO_INDEX[card.__class__]
            owned_cards_vector[card_index] = 1

        return owned_cards_vector

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

        play_randomly(self.game_state, False, self.PLAYERS_ACTION_SPACE_LENGTH, self.card_pool_list)

        observation_array = self.get_observation_array()
        info_dictionary = {}
        return observation_array, reward, terminated_flag, truncated_flag, info_dictionary
