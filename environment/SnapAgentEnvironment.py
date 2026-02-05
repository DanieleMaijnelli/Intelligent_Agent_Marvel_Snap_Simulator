from environment.UtilityFunctions import *
from gameManager import GameState
from EnvironmentUtilityFunctions import build_hand_observation, build_locations_observation, \
    build_played_cards_observation


class SnapAgentEnvironment:
    def __init__(self, verbose=False):
        self.game_state = GameState(verbose=verbose)

    def build_basic_observation_snap_agent(self, is_ally):
        status_dictionary = self.game_state.status
        feature_list = [float(status_dictionary["turncounter"]) / 7.0, float(status_dictionary["cubes"]) / 8.0,
                        float(status_dictionary["tempcubes"]) / 8.0]

        if is_ally:
            feature_list.append(float(status_dictionary["allyenergy"]) / 10.0)
            feature_list.append(1.0 if status_dictionary["allypriority"] else 0.0)
            feature_list.append(1.0 if status_dictionary["allysnapped"] else 0.0)
            feature_list.append(1.0 if status_dictionary["enemysnapped"] else 0.0)
        else:
            feature_list.append(float(status_dictionary["enemyenergy"]) / 10.0)
            feature_list.append(1.0 if not status_dictionary["allypriority"] else 0.0)
            feature_list.append(1.0 if status_dictionary["enemysnapped"] else 0.0)
            feature_list.append(1.0 if status_dictionary["allysnapped"] else 0.0)

        return feature_list

    def build_observation_snap_agent(self, is_ally):
        game_state = self.game_state
        feature_list = []
        basic_observation = self.build_basic_observation_snap_agent(is_ally)
        locations_observation = build_locations_observation(game_state, is_ally, -1)
        hand_observation = build_hand_observation(game_state, is_ally, -1)
        played_cards_observation = build_played_cards_observation(game_state, is_ally)
        feature_list.extend(basic_observation)
        feature_list.extend(locations_observation)
        feature_list.extend(hand_observation)
        feature_list.extend(played_cards_observation)
        return numpy.array(feature_list, dtype=numpy.float32)

    def reset(self):
        self.game_state.reset()

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
