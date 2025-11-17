from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
from gameManager import GameState
import random
import numpy


class SnapEnv(ParallelEnv):
    metadata = {"render.modes": ["human"], "name": "marvel_snap_v0"}

    def __init__(self):
        super().__init__()
        self.possible_agents = ["player_1", "player_2"]
        self.agents = self.possible_agents[:]
        self.gm = GameState()
        self.gm.reset()
        self.tracked_player = None
        self.card_names = [

            "Agent 13",
            "America Chavez",
            "Angel",
            "Antman",
            "Apocalypse",
            "Armor",
            "Bishop",
            "Blade",
            "Blue Marvel",
            "Bucky Barnes",
            "Winter Soldier",
            "Cable",
            "Captain America",
            "Carnage",
            "Cloak",
            "Coleen Wing",
            "Colossus",
            "Cosmo",
            "Death",
            "Deathlok",
            "Devil Dinosaur",
            "Doctor Strange",
            "Ebony Maw",
            "Elektra",
            "Enchantress",
            "Forge",
            "Gamora",
            "Groot",
            "Hawkeye",
            "Heimdall",
            "Hobgoblin",
            "Hulk Buster",
            "Iceman",
            "The Infinaut",
            "Iron Fist",
            "Iron Man",
            "Jessica Jones",
            "Jubilee",
            "Kazar",
            "Killmonger",
            "Klaw",
            "Knull",
            "Korg",
            "Kraven",
            "Lady Sif",
            "Lizard",
            "Magik",
            "Mantis",
            "Medusa",
            "Mister Sinister",
            "Sinister Clone",
            "Namor",
            "Nightcrawler",
            "Onslaught",
            "Punisher",
            "Modok",
            "Moongirl",
            "Morbius",
            "Morph",
            "Mr Fantastic",
            "Multiple Man",
            "Nakia",
            "Nova",
            "Odin",
            "Okoye",
            "Professor X",
            "Psylocke",
            "Rhino",
            "Rocket Racoon",
            "Sabretooth",
            "Scarlet Witch",
            "Scorpion",
            "Sentinel",
            "Shang-Chi",
            "Squirrel Girl",
            "Squirrel",
            "Star Lord",
            "Storm",
            "Strong Guy",
            "Sunspot",
            "Swarm",
            "Sword Master",
            "Uatu The Watcher",
            "Vision",
            "Vulture",
            "Warpath",
            "White Queen",
            "White Tiger",
            "Tiger",
            "Wolfsbane",
            "Wolvering",
            "Yondu"
        ]
        self.CARD_NAME_TO_ID = {name: idx for idx, name in enumerate(self.card_names)}
        self.location_names = [
            "Revealing Location in 1 turns",
            "Revealing Location in 2 turns",
            "Revealing Location in 3 turns",
            "Altar of Death",
            "Asgard",
            "Asteroid M",
            "Atlantis",
            "Bar with no name",
            "Castle Blackstone",
            "Crimson Cosmos",
            "Crown City",
            "Elysium",
            "Fisk Tower",
            "Hellfire Club",
            "Jotunheim",
            "Kamar-Taj",
            "Kyln",
            "Lake Heldas",
            "Limbo",
            "Milano",
            "Mojo World",
            "Necrosha",
            "Negative Zone",
            "New York",
            "Nidavellir",
            "Onslaught Citadel",
            "Plunder Castle",
            "Project Pegasus",
            "Sanctum Sanctorum",
            "Sewer System",
            "Wakanda"
        ]
        self.LOCATION_NAME_TO_ID = {name: idx for idx, name in enumerate(self.location_names)}
        max_cards = len(self.card_names)
        max_hand_size = 7
        num_location = 3
        max_turns = 7
        self.games = 0
        self.games_won = 0
        self.winrate = 0
        self.passing = {"player_1": False, "player_2": False}
        self.counter = 0

        card_space = spaces.Dict({
            "id": spaces.Discrete(max_cards),
            "cost": spaces.Box(low=0, high=15, shape=(), dtype=np.int32),
            "power": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.int32),
        })

        location_card_space = spaces.Dict({
            "id": spaces.Discrete(max_cards),
            "cost": spaces.Box(low=0, high=15, shape=(), dtype=np.int32),
            "power": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.int32),
        })

        location_space = spaces.Dict({
            "id": spaces.Discrete(len(self.location_names)),
            "ally_power": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.int32),
            "enemy_power": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.int32),
            "ally_cards": spaces.Sequence(location_card_space),
            "ally_cards_pre_reveal": spaces.Sequence(location_card_space),
            "enemy_cards": spaces.Sequence(location_card_space),
        })

        self._observation_space = spaces.Dict({
            "hand": spaces.Sequence(card_space),
            "energy": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.int32),
            "turn": spaces.Discrete(max_turns + 1),
            "locations": spaces.Dict({
                "location1": location_space,
                "location2": location_space,
                "location3": location_space,
            })
        })
        self.observation_spaces = {agent: self._observation_space for agent in self.possible_agents}

        action_size = max_hand_size * num_location + 1
        self.pass_action = 21
        self.action_spaces = {agent: spaces.Discrete(action_size) for agent in self.possible_agents}

    def flatten_obs(self, obs, agent="player_1"):
        hand = self.flatten_hand(obs['hand'], max_size=7)
        energy = np.array([obs["energy"]])
        turn = np.array([obs["turn"]])
        locations = []
        for location in obs["locations"].values():
            flat_loc = np.array(self.flatten_location(location))
            locations.append(flat_loc)
        return np.concatenate((hand, energy, turn, locations[0], locations[1], locations[2]))

    def flatten_hand(self, hand, max_size):
        flat_hand = np.full(max_size * 3, -1, dtype=np.float32)
        for i, card in enumerate(hand[:max_size]):
            flat_hand[i * 3:(i + 1) * 3] = card["id"], card["cost"], card["power"]
        return flat_hand

    def flatten_location(self, location):
        loc_id = np.array([location["id"]])
        ally_power = np.array([location["ally_power"]])
        enemy_power = np.array([location["enemy_power"]])
        ally_cards = self.flatten_hand(location["ally_cards"], 4)
        ally_cards_pre_reveal = self.flatten_hand(location["ally_cards_pre_reveal"], 4)
        enemy_cards = self.flatten_hand(location["enemy_cards"], 4)
        return np.concatenate((loc_id, ally_power, enemy_power, ally_cards, ally_cards_pre_reveal, enemy_cards))

    def encode_action(self, card_idx, location_idx):
        return card_idx * self.num_location + location_idx

    def decode_action(self, action):
        card_idx = action // 3
        location_idx = action % 3
        return (card_idx, location_idx)

    def reset(self, seed=None, options=None):
        self.gm.reset()
        self.games += 1
        self.winrate = self.games_won / self.games
        self.agents = ["player_1", "player_2"]
        if seed is not None:
            random.seed(seed)
        observations = {agent: self.observe(agent) for agent in self.agents}
        return observations

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def fromCardToDict(self, card):
        if card is None:
            return {"id": -1, "cost": 0, "power": 0}
        return {
            "id": self.CARD_NAME_TO_ID.get(card.name, -1),
            "cost": card.cur_cost,
            "power": card.cur_power}

    def calculateRewards(self):
        player1count, player2count = 0, 0
        for location in self.gm.locationList.values():
            if location.winning == "Ally":
                player1count += 1
            elif location.winning == "Enemy":
                player2count += 1
        reward = {"player_1": player1count, "player_2": player2count}
        return reward

    def step(self, actions):
        rewards = {agent: 0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        if self.gm.game_end == False:
            for agent, agent_actions in actions.items():
                if isinstance(agent_actions, np.int64):
                    agent_actions = [agent_actions.item()]
                else:
                    agent_actions = [agent_actions]
                for action in agent_actions:
                    if action == self.pass_action or self.counter >= 7:
                        if self.counter >= 7:
                            for agentpass in self.agents:
                                if self.passing[agentpass] == False:
                                    rewards[agentpass] -= 0.4
                                    infos[agentpass]["nopass"] = True
                        self.passing[agent] = True
                        if (self.passing["player_1"] and self.passing["player_2"]) or self.counter >= 7:
                            self.counter = 0
                            print("Both players passed")
                            for card in self.gm.status[
                                "allyhand"]:  # reward negative per ogni carta in mano rimasta che non hanno giocato nonostante avessero l'energia
                                if card.cost < self.gm.status["allyenergy"]:
                                    rewards["player_1"] -= card.cost * 0.1
                            for card in self.gm.status["enemyhand"]:
                                if card.cost < self.gm.status["enemyenergy"]:
                                    rewards["player_2"] -= card.cost * 0.1
                            self.gm.turnEnd(True)
                            if self.gm.game_end:
                                break
                    cardidx, locationidx = self.decode_action(
                        action)  # in base al valore in input, ricava la carta e le location in cui viene giocata
                    current_hand = self.gm.getHand(agent=agent)
                    if cardidx > len(current_hand) - 1:
                        rewards[agent] -= 0.3
                        infos[agent]["invalid_action"] = True
                        print("invalid action")
                        continue
                    else:
                        card = current_hand[cardidx]
                        if agent == "player_1":
                            ally = True
                            if card.cost <= self.gm.status["allyenergy"]:
                                result = self.gm.addUnit(card, ally,
                                                         self.gm.locationList[f"location{locationidx + 1}"].locationNum)
                                if result:
                                    self.gm.status["allyenergy"] -= card.cost
                                    self.gm.status["allyhand"].remove(card)
                                    rewards[agent] += 0.5
                                    infos[agent]["card_played"] = True
                                    if self.gm.status["allyenergy"] == 0:
                                        rewards[agent] += 0.5
                                        infos[agent]["0_energy_bonus"] = True
                                else:
                                    rewards[agent] -= 0.5
                                    infos[agent]["illegal_move"] = True
                                    print("illegal move")
                        elif agent == "player_2":
                            ally = False
                            if card.cost <= self.gm.status["enemyenergy"]:
                                result = self.gm.addUnit(card, ally,
                                                         self.gm.locationList[f"location{locationidx + 1}"].locationNum)
                                if result:
                                    self.gm.status["enemyenergy"] -= card.cost
                                    self.gm.status["enemyhand"].remove(card)
                                    rewards[agent] += 0.5
                                    infos[agent]["card_played"] = True
                                    if self.gm.status["enemyenergy"] == 0:
                                        rewards[agent] += 0.5
                                        infos[agent]["0_energy_bonus"] = True
                                else:
                                    rewards[agent] -= 0.5
                                    infos[agent]["illegal_move"] = True
                                    print("illegal move")
        if self.gm.game_end or self.gm.status["turncounter"] > 7:
            terminations = {agent: True for agent in self.agents}
            if self.gm.checkWinner() == "Ally":
                rewards["player_1"] += 3
                rewards["player_2"] -= 3
                if self.tracked_player == "player_1":
                    self.games_won += 1
                infos["player_1"]["win"] = True
                infos["player_2"]["loss"] = True
            elif self.gm.checkWinner() == "Enemy":
                rewards["player_1"] -= 3
                rewards["player_2"] += 3
                if self.tracked_player == "player_2":
                    self.games_won += 1
                infos["player_1"]["loss"] = True
                infos["player_2"]["win"] = True
            else:
                rewards["player_1"] -= 0.5
                rewards["player_2"] -= 0.5
                infos["player_1"]["draw"] = True
                infos["player_2"]["draw"] = True
        if (self.gm.locationList["location1"].alliesPower + self.gm.locationList["location2"].alliesPower +
            self.gm.locationList["location3"].alliesPower) / 3 >= 8 and self.gm.checkWinner() == "Ally":
            rewards["player_1"] += 2
            infos["player_1"]["bonus_power"] = True
        self.counter += 1
        observations = {agent: self.observe(agent) for agent in self.agents}
        self.agents = [] if all(terminations.values()) else self.agents
        self.render()
        return observations, rewards, terminations, truncations, infos

    def observe(self, agent):
        hand = self.gm.getHand(agent=agent)
        obs_hand = [self.fromCardToDict(card) for card in hand]  # Rimosso `{}` esterno inutile

        obs_locations = {}  # raccogli tutte le location

        for locnum in range(3):
            loc = self.gm.locationList["location" + str(locnum + 1)]
            if agent == "player_1":
                obs_locations[f"location{locnum + 1}"] = {
                    "id": self.LOCATION_NAME_TO_ID.get(loc.name, -1),
                    "ally_power": loc.alliesPower,
                    "enemy_power": loc.enemiesPower,
                    "ally_cards": [self.fromCardToDict(card) for card in loc.allies],
                    "ally_cards_pre_reveal": [self.fromCardToDict(card) for card in loc.preRevealAllies],
                    "enemy_cards": [self.fromCardToDict(card) for card in loc.enemies],
                }
            elif agent == "player_2":
                # Inverti le prospettive
                obs_locations[f"location{locnum + 1}"] = {
                    "id": self.LOCATION_NAME_TO_ID.get(loc.name, -1),
                    "ally_power": loc.enemiesPower,
                    "enemy_power": loc.alliesPower,
                    "ally_cards": [self.fromCardToDict(card) for card in loc.enemies],
                    "ally_cards_pre_reveal": [self.fromCardToDict(card) for card in loc.preRevealEnemies],
                    "enemy_cards": [self.fromCardToDict(card) for card in loc.allies],
                }

        return {
            "hand": obs_hand,
            "energy": self.gm.status["allyenergy"] if agent == "player_1" else self.gm.status["enemyenergy"],
            "turn": self.gm.status["turncounter"],
            "locations": obs_locations
        }

    def render(self):
        print(self.gm.print_board_state())
