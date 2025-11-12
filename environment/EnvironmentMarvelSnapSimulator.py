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
        self.agents = AGENTS
        self.possible_agents = AGENTS
        self._obs_dim = 6 + 3 * 4 + 2 * MAX_HAND  # base + locations + hand feats
        self.observation_spaces = {
            a: spaces.Box(-np.inf, np.inf, shape=(self._obs_dim,), dtype=np.float32) for a in self.agents
        }
        self.action_spaces = {a: spaces.Discrete(MAX_ACTIONS) for a in self.agents}

        self.game = GameState()
        self._action_maps = {a: [] for a in self.agents}

    # ---------- API ----------
    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
        self.game.reset()  # il simulatore fa gameStart() dentro reset()

        obs = {a: self._encode_obs(a) for a in self.agents}
        infos = {a: {"action_mask": self._mask(a), "action_meanings": self._pretty(a)} for a in self.agents}
        return obs, infos

    def step(self, actions):
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        rewards = {a: 0.0 for a in self.agents}
        infos = {}

        # --- 1) interpreta le azioni simultanee ---
        intents = {a: self._resolve_action(a, actions[a]) for a in self.agents}

        # --- 2) RETREAT ha priorità: termina subito ---
        if intents["player_1"][0] == "retreat":
            self.game.retreat(True)
            terminations = {a: True for a in self.agents}
        if intents["player_2"][0] == "retreat":
            self.game.retreat(False)
            terminations = {a: True for a in self.agents}

        # --- 3) SNAP (non terminale) ---
        if intents["player_1"][0] == "snap":
            self.game.snap(True)
        if intents["player_2"][0] == "snap":
            self.game.snap(False)

        # --- 4) PLAY: gioca carta (mano index, location) ---
        for agent, ally_flag in (("player_1", True), ("player_2", False)):
            kind = intents[agent][0]
            if kind == "play":
                _, h_idx, loc_key = intents[agent]
                hand = self._hand(ally_flag)
                if h_idx < len(hand):
                    card = hand[h_idx]
                    loc_num = int(loc_key[-1])  # "location1" -> 1
                    self.game.addUnit(card, ally_flag, loc_num)

        # --- 5) controllo di fine turno ---
        # ora un turno finisce SOLO se entrambi hanno passato o ritirato
        p1_pass = intents["player_1"][0] in ("pass", "retreat")
        p2_pass = intents["player_2"][0] in ("pass", "retreat")

        if not any(terminations.values()) and p1_pass and p2_pass:
            # chiudi il turno corrente
            self.game.turnEnd(True)
            if getattr(self.game, "game_end", False):
                terminations = {a: True for a in self.agents}
            else:
                # PREPARA IL TURNO SUCCESSIVO (energia, pesca, ecc.)
                self.game.startOfTurn()

        # --- 6) reward terminale zero-sum ---
        if any(terminations.values()):
            rew = self._terminal_reward()  # POV Ally
            rewards["player_1"] = rew
            rewards["player_2"] = -rew

        # --- 7) osservazioni, mask, info ---
        obs = {a: self._encode_obs(a) for a in self.agents}
        infos = {a: {"action_mask": self._mask(a), "action_meanings": self._pretty(a)} for a in self.agents}
        return obs, rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # ---------- Helpers ----------

    def _terminal_reward(self) -> float:
        # Coerente con l'implementazione corrente del simulatore:
        # - al turno 6 tempcubes viene già raddoppiato in startOfTurn()
        # - endOfTurn copia tempcubes -> cubes
        # - endGame usa cubes "liscio"
        try:
            winner = self.game.checkWinner()
        except Exception:
            winner = self.game.passStatus.get("winner", "Tie")
        cubes = float(self.game.status.get("cubes", self.game.status.get("tempcubes", 1)))
        if winner == "Ally":
            return cubes
        if winner == "Enemy":
            return -cubes
        return 0.0

    def _hand(self, ally: bool):
        return self.game.status["allyhand"] if ally else self.game.status["enemyhand"]

    def _can_play(self, ally: bool):
        energy = self.game.status["allyenergy"] if ally else self.game.status["enemyenergy"]
        hand = self._hand(ally)
        locs = [
            self.game.locationList["location1"],
            self.game.locationList["location2"],
            self.game.locationList["location3"],
        ]
        for c in hand:
            if getattr(c, "cur_cost", 99) <= energy and any(not L.checkIfLocationFull(ally) for L in locs):
                return True
        return False

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
        energy = self.game.status["allyenergy"] if ally else self.game.status["enemyenergy"]
        locs = [
            ("location1", self.game.locationList["location1"]),
            ("location2", self.game.locationList["location2"]),
            ("location3", self.game.locationList["location3"]),
        ]
        amap, mask = [], []

        # PLAY(card,loc) fino a MAX_HAND carte
        for h in range(MAX_HAND):
            if (h < len(hand)):
                card = hand[h]
            else:
                card = None
            for name, L in locs:
                if (card is not None):
                    legal = (getattr(card, "cur_cost", 99) <= energy) and (not L.checkIfLocationFull(ally))
                else:
                    legal = False
                amap.append(("play", h, name))
                mask.append(1 if legal else 0)

        # PASS sempre disponibile
        amap.append(("pass",))
        mask.append(1)

        # SNAP disponibile se non già snappato da quel lato
        can_snap = not (self.game.status["allysnapped"] if ally else self.game.status["enemysnapped"])
        amap.append(("snap",))
        mask.append(1 if can_snap else 0)

        # RETREAT sempre disponibile finché la partita non è finita
        can_retreat = not getattr(self.game, "game_end", False)
        amap.append(("retreat",))
        mask.append(1 if can_retreat else 0)

        self._action_maps[agent] = amap
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
