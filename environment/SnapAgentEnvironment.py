import random
from environment.PlayerAgentEnvironment import PlayerAgentEnvironment
from enum import Enum


class Action(Enum):
    NOTHING = 0
    RETREAT = 1
    SNAP = 2


class SnapAgentEnvironment:
    def __init__(self, player_network):
        self.player_network = player_network
        self.player_agent_environment = PlayerAgentEnvironment()
        self.game_state = self.player_agent_environment.game_state
        self.player_network.eval()

    def snap_randomly(self, is_ally: bool, snap_probability):
        game_state = self.game_state
        if random.random() < snap_probability:
            if is_ally:
                game_state.snap(True)
            else:
                game_state.snap(False)

    def execute_player_turn(self, is_ally: bool):
        from training.TrainingUtilityFunctions import choose_action_epsilon_greedy
        player_action_type = None
        while player_action_type != "Passed":
            player_action, state_action_vector = choose_action_epsilon_greedy(
                self.player_agent_environment, self.player_network, is_ally, 0.01
            )
            player_action_type, done = self.player_agent_environment.step(player_action, is_ally)

    def reset(self):
        self.player_agent_environment.reset()
        self.snap_randomly(False, 0.01)

    def step(self, action):
        if action == Action.RETREAT:
            self.game_state.retreat(True)
            done = True
        else:
            if action == Action.SNAP:
                self.game_state.snap(True)
            self.execute_player_turn(False)
            self.execute_player_turn(True)
            done = self.game_state.game_end

        if not done:
            self.snap_randomly(False, 0.10)

        return done
