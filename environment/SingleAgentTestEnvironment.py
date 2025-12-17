from gameManager import GameState


class SingleAgentTestEnvironment:
    def __init__(self, verbose=False):
        self.game_state = GameState(verbose=verbose)

    def reset(self):
        self.game_state.reset()

    def apply_single_action(self, is_ally, action_tuple):
        hand_index, location_index = action_tuple

        if hand_index == -1:
            if is_ally:
                self.game_state.status["allypass"] = True
            else:
                self.game_state.status["enemypass"] = True
            return "Passed"

        if is_ally:
            if self.game_state.status["allypass"]:
                return "Passed"
            self.game_state.addUnit(hand_index, True, location_index + 1)
            return "Played"
        else:
            if self.game_state.status["enemypass"]:
                return "Passed"
            self.game_state.addUnit(hand_index, False, location_index + 1)
            return "Played"

    def step(self, action, is_ally):
        executed_action = self.apply_single_action(is_ally, action)

        if self.game_state.status["allypass"] and self.game_state.status["enemypass"]:
            self.game_state.turnEnd(True)

        done = bool(self.game_state.game_end)

        return executed_action, done
