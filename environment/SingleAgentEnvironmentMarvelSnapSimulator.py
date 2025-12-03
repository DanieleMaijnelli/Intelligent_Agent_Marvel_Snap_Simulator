import gymnasium as gym
from gymnasium import spaces
from environment.UtilityFunctions import *


class MarvelSnapSingleAgentEnv(gym.Env):

    def __init__(self, game_state):
        super().__init__()
        self.game_state = game_state
        self.card_pool_list = Decks.ALL_CARDS
        self.number_of_cards = len(self.card_pool_list)
        self.ACTION_SPACE_LENGTH = 3 * self.number_of_cards + 1
        self.LOCATION_MASK_LENGTH = len(Decks.ALL_LOCATIONS) * 3
        self.observation_space = spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(9 + self.ACTION_SPACE_LENGTH + self.LOCATION_MASK_LENGTH, ),
            dtype=numpy.float32
        )
        self.action_space = spaces.Discrete(self.ACTION_SPACE_LENGTH)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game_state.reset()
        observation_array = get_enriched_observation_array_single_agent(self.game_state, self.ACTION_SPACE_LENGTH, self.LOCATION_MASK_LENGTH)
        info_dictionary = {}
        return observation_array, info_dictionary

    def step(self, action):
        terminated_flag = False
        truncated_flag = False
        reward = 0.0
        integer_action = int(action)
        play_randomly(self.game_state, False, self.ACTION_SPACE_LENGTH, self.card_pool_list)
        if build_player_action_mask(self.game_state, True, self.ACTION_SPACE_LENGTH)[integer_action] == 1:
            if integer_action == (self.ACTION_SPACE_LENGTH - 1):
                self.game_state.status["allypass"] = True
            else:
                card_type, location_number = decode_action_index(integer_action, self.card_pool_list)
                for card_index, card in enumerate(self.game_state.status["allyhand"]):
                    if isinstance(card, card_type):
                        if not self.game_state.addUnit(card_index, True, location_number):
                            continue
                        reward += 1.5
                        reward += card.cur_power / 1.5
                        break
        else:
            reward -= 1.0
            self.game_state.status["allypass"] = True

        if self.game_state.status["allypass"] and self.game_state.status["enemypass"]:
            reward -= (self.game_state.status["allyenergy"] / 3.0)
            self.game_state.turnEnd(True)

            for location in self.game_state.locationList.values():
                if location.alliesPower > location.enemiesPower:
                    reward += 0.5
                elif location.alliesPower < location.enemiesPower:
                    reward -= 0.5

            if self.game_state.game_end:
                if self.game_state.passStatus["winner"] == "Ally":
                    reward += 7.0
                elif self.game_state.passStatus["winner"] == "Enemy":
                    reward -= 7.0
                terminated_flag = True

        observation_array = get_enriched_observation_array_single_agent(self.game_state, self.ACTION_SPACE_LENGTH, self.LOCATION_MASK_LENGTH)
        info_dictionary = {}
        return observation_array, reward, terminated_flag, truncated_flag, info_dictionary
