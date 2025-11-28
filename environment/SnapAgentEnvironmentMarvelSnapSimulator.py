import gymnasium as gym
from gymnasium import spaces
from environment.UtilityFunctions import *


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
            shape=(18 + self.number_of_cards,),
            dtype=numpy.float32
        )
        self.action_space = spaces.Discrete(3)
        self.ACTION_SNAP = 0
        self.ACTION_RETREAT = 1
        self.ACTION_NOTHING = 2

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game_state.reset()
        play_randomly(self.game_state, False, self.PLAYERS_ACTION_SPACE_LENGTH, self.card_pool_list)
        observation_array = get_observation_array_snap_agent(self.game_state, self.number_of_cards)
        info_dictionary = {}
        return observation_array, info_dictionary

    def step(self, action):
        terminated_flag = False
        truncated_flag = False
        reward = 0.0
        integer_action = int(action)

        if integer_action == self.ACTION_RETREAT:
            self.game_state.retreat(True)
            terminated_flag = True
        else:
            if integer_action == self.ACTION_SNAP:
                self.game_state.snap(True)
            play_randomly(self.game_state, True, self.PLAYERS_ACTION_SPACE_LENGTH, self.card_pool_list)
            self.game_state.turnEnd(True)

        if self.game_state.game_end:
            cubes = float(self.game_state.status["cubes"])
            if self.game_state.passStatus["winner"] == "Ally":
                reward += cubes
            elif self.game_state.passStatus["winner"] == "Enemy":
                reward -= cubes
            terminated_flag = True
        else:
            play_randomly(self.game_state, False, self.PLAYERS_ACTION_SPACE_LENGTH, self.card_pool_list)
            snap_randomly(self.game_state, False, 0.15)
        observation_array = get_observation_array_snap_agent(self.game_state, self.number_of_cards)
        info_dictionary = {}
        return observation_array, reward, terminated_flag, truncated_flag, info_dictionary
