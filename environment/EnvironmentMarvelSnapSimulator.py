import numpy as np
import random
from gameManager import GameState
from gymnasium import spaces
from pettingzoo.utils import ParallelEnv

MAX_HAND = 7
MAX_ACTIONS = MAX_HAND * 3 + 3

AGENTS = ["player_1", "player_2"]


class TestEnvironmentMarvelSnapSimulator(ParallelEnv):
    metadata = {"name": "marvel_snap_parallel_v0"}

    def __init__(self):
        super().__init__()
        self.possible_agents = AGENTS
        self.agents = list(AGENTS)
        self._obs_dim = 6 + 3 * 4 + 2 * MAX_HAND
        self.observation_spaces = {
            a: spaces.Box(-np.inf, np.inf, shape=(self._obs_dim,), dtype=np.float32) for a in self.agents
        }
        self.action_spaces = {a: spaces.Discrete(MAX_ACTIONS) for a in self.agents}

        self.game = GameState()
        self._action_maps = {a: [] for a in self.agents}

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
        self.game.reset()
        self.agents = self.possible_agents[:]
        self._action_maps = {a: [] for a in self.agents}

        obs = {a: self._encode_obs(a) for a in self.agents}
        infos = {a: {"action_mask": self._mask(a), "action_meanings": self._pretty(a)} for a in self.agents}
        return obs, infos

    def step(self, actions):
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        rewards = {a: 0.0 for a in self.agents}
        infos = {}

        intents = {a: self._resolve_action(a, actions[a]) for a in self.agents}

        if intents["player_1"][0] == "retreat":
            self.game.retreat(True)
            terminations = {a: True for a in self.agents}
        if intents["player_2"][0] == "retreat":
            self.game.retreat(False)
            terminations = {a: True for a in self.agents}

        if intents["player_1"][0] == "snap":
            self.game.snap(True)
        if intents["player_2"][0] == "snap":
            self.game.snap(False)

        for agent, ally_flag in (("player_1", True), ("player_2", False)):
            kind = intents[agent][0]
            if kind == "play":
                _, hand_index, loc_key = intents[agent]
                hand = self._hand(ally_flag)
                if hand_index < len(hand):
                    loc_num = int(loc_key[-1])
                    self.game.addUnit(hand_index, ally_flag, loc_num)

        p1_pass = intents["player_1"][0] in ("pass", "retreat")
        p2_pass = intents["player_2"][0] in ("pass", "retreat")

        if not any(terminations.values()) and p1_pass and p2_pass:
            self.game.turnEnd(True)
            if getattr(self.game, "game_end", False):
                terminations = {a: True for a in self.agents}

        if any(terminations.values()):
            rew = self._terminal_reward()
            rewards["player_1"] = rew
            rewards["player_2"] = -rew

        obs = {a: self._encode_obs(a) for a in self.agents}
        infos = {a: {"action_mask": self._mask(a), "action_meanings": self._pretty(a)} for a in self.agents}
        return obs, rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _terminal_reward(self) -> float:
        winner = self.game.passStatus['winner']
        cubes = float(self.game.status.get("cubes"))
        if winner == "Ally":
            return cubes
        if winner == "Enemy":
            return -cubes
        return 0.0

    def _hand(self, ally: bool):
        return self.game.status["allyhand"] if ally else self.game.status["enemyhand"]

    def _can_play(self, ally: bool, card, location):
        if ally:
            energy = self.game.status["allyenergy"]
            energy_check = getattr(card, "cur_cost", 99) <= energy
            location_check = (not location.checkIfLocationFull(ally)) and location.can_play_cards_allies
            unit_check = location.canCardBePlayed(card)
            return energy_check and location_check and unit_check
        else:
            energy = self.game.status["enemyenergy"]
            energy_check = getattr(card, "cur_cost", 99) <= energy
            location_check = (not location.checkIfLocationFull(ally)) and location.can_play_cards_enemies
            unit_check = location.canCardBePlayed(card)
            return energy_check and location_check and unit_check

    def _resolve_action(self, agent_name, chosen_action_index):
        legal_action_mask = self._mask(agent_name)

        is_out_of_bounds = chosen_action_index >= len(legal_action_mask)
        is_illegal = not is_out_of_bounds and (legal_action_mask[chosen_action_index] == 0)

        if is_out_of_bounds or is_illegal:
            return ("pass",)

        action_map = self._action_maps.get(agent_name, [])
        if chosen_action_index < len(action_map):
            action_description = action_map[chosen_action_index]
        else:
            action_description = ("pass",)

        return action_description

    def _mask(self, agent):
        ally = (agent == "player_1")
        hand = self._hand(ally)
        locations = [
            ("location1", self.game.locationList["location1"]),
            ("location2", self.game.locationList["location2"]),
            ("location3", self.game.locationList["location3"]),
        ]
        actions_map, mask = [], []

        for h in range(MAX_HAND):
            if h < len(hand):
                card = hand[h]
            else:
                card = None
            for name, location in locations:
                if card is not None:
                    legal = self._can_play(ally, card, location)
                else:
                    legal = False
                actions_map.append(("play", h, name))
                mask.append(1 if legal else 0)

        actions_map.append(("pass",))
        mask.append(1)

        can_snap = not (self.game.status["allysnapped"] if ally else self.game.status["enemysnapped"])
        actions_map.append(("snap",))
        mask.append(1 if can_snap else 0)

        can_retreat = not getattr(self.game, "game_end", False)
        actions_map.append(("retreat",))
        mask.append(1 if can_retreat else 0)

        self._action_maps[agent] = actions_map
        return np.array(mask, dtype=np.int8)

    def _encode_obs(self, agent):
        ally = (agent == "player_1")
        status = self.game.status
        locations = self.game.locationList

        def loc_feats(loc):
            ally_power = getattr(loc, "alliesPower", 0)
            enemy_power = getattr(loc, "enemiesPower", 0)
            aC = len(getattr(loc, "allies", []))
            eC = len(getattr(loc, "enemies", []))
            return [ally_power, enemy_power, aC, eC]

        base = [
            status.get("turncounter", 1),
            status.get("allyenergy", 0) if ally else status.get("enemyenergy", 0),
            status.get("enemyenergy", 0) if ally else status.get("allyenergy", 0),
            status.get("cubes", 1),
            status.get("tempcubes", 1),
            1.0 if (status.get("allypriority", True) if ally else not status.get("allypriority", True)) else 0.0,
        ]
        vec = base
        vec += loc_feats(locations["location1"])
        vec += loc_feats(locations["location2"])
        vec += loc_feats(locations["location3"])

        hand = self._hand(ally)
        hand_feats = []
        for i in range(MAX_HAND):
            if i < len(hand):
                c = hand[i]
                hand_feats += [getattr(c, "cur_cost", 0), getattr(c, "cur_power", 0)]
            else:
                hand_feats += [0, 0]
        vec += hand_feats
        return np.array(vec, dtype=np.float32)

    def _pretty(self, agent):
        readable_labels = []
        action_descriptions = self._action_maps.get(agent, [])

        for action_description in action_descriptions:
            action_type = action_description[0]

            if action_type == "play":
                hand_index = action_description[1]
                target_location = action_description[2]
                readable_labels.append(f"PLAY(hand[{hand_index}] → {target_location})")
            else:
                readable_labels.append(action_type.upper())

        return readable_labels
