import importlib
import random
import cards
import Locations
from Locations.Location import *
from nanoid import generate
import uuid
import json
import jsonschema
import os
import time
from datetime import datetime
import inspect


class GameState:
    def __init__(self):
        self.game = {
            "game_id": "",
            "winner": "None",
            "start_time": "",
            "end_time": "",
        }
        self.version = '1.1.0'
        self.game_id = ''
        self.exit = False
        self.turnAlly = False
        self.allymaxenergy, self.enemymaxenergy = 1, 1
        self.turncounter = 1
        self.maxturns = 6
        self.game_end = False
        self.locationList = {"location1": 0, "location2": 0, "location3": 0}
        self.status = {"maxturns": self.maxturns, "allymaxenergy": self.allymaxenergy,
                       "enemymaxenergy": self.enemymaxenergy, "allyenergy": 1,
                       "enemyenergy": 1, "turncounter": self.turncounter,
                       "tempenergyally": 0, "tempenergyenemy": 0,
                       "allyhand": [], "enemyhand": [],
                       "allydeck": [], "enemydeck": [],
                       "alliesdestroyed": [], "enemiesdestroyed": [],
                       "alliesdiscarded": [], "enemiesdiscarded": [],
                       "allypriority": True,
                       "cubes": 1, "tempcubes": 1,
                       "allysnapped": False, "enemysnapped": False,
                       "cardsplayed": [], "onnextcardbeingplayed": [],
                       "allypass": False, "enemypass": False,
                       "endofturncounterally": 0, "endofturncounterenemy": 0,
                       "locationlist": self.locationList}
        self.passStatus = {
            'turnpassally': self.status['allypass'],
            'turnpassenemy': self.status['enemypass'],
            'winner': "None",
            'retreatally': False,
            'retreatenemy': False,
            'turnend': False
        }
        self.locationList["location1"] = TemporaryLocation(1, self.status, self.locationList)
        self.locationList["location2"] = TemporaryLocation(2, self.status, self.locationList)
        self.locationList["location3"] = TemporaryLocation(3, self.status, self.locationList)

    def reset(self):
        self.exit = False
        self.turnAlly = False
        self.allymaxenergy, self.enemymaxenergy = 1, 1
        self.turncounter = 1
        self.maxturns = 6
        self.game_end = False
        self.locationList = {"location1": 0, "location2": 0, "location3": 0}
        self.status = {"maxturns": self.maxturns, "allymaxenergy": self.allymaxenergy,
                       "enemymaxenergy": self.enemymaxenergy, "allyenergy": 1,
                       "enemyenergy": 1, "turncounter": self.turncounter,
                       "tempenergyally": 0, "tempenergyenemy": 0,
                       "allyhand": [], "enemyhand": [],
                       "allydeck": [], "enemydeck": [],
                       "alliesdestroyed": [], "enemiesdestroyed": [],
                       "alliesdiscarded": [], "enemiesdiscarded": [],
                       "allypriority": True,
                       "cubes": 1, "tempcubes": 1,
                       "allysnapped": False, "enemysnapped": False,
                       "cardsplayed": [], "onnextcardbeingplayed": [],
                       "allypass": False, "enemypass": False,
                       "endofturncounterally": 0, "endofturncounterenemy": 0,
                       "locationlist": self.locationList}
        self.passStatus = {
            'turnpassally': self.status['allypass'],
            'turnpassenemy': self.status['enemypass'],
            'winner': "None",
            'retreatally': False,
            'retreatenemy': False,
            'turnend': False
        }
        self.locationList["location1"] = TemporaryLocation(1, self.status, self.locationList)
        self.locationList["location2"] = TemporaryLocation(2, self.status, self.locationList)
        self.locationList["location3"] = TemporaryLocation(3, self.status, self.locationList)
        self.gameStart()

    def resolveTie(self):
        allypower = self.locationList["location1"].alliesPower + self.locationList["location2"].alliesPower + \
                    self.locationList["location3"].alliesPower
        enemypower = self.locationList["location1"].enemiesPower + self.locationList["location2"].enemiesPower + \
                     self.locationList["location3"].enemiesPower
        if allypower > enemypower:
            return "Ally"
        elif allypower < enemypower:
            return "Enemy"
        else:
            return "Tie"

    def checkWinner(self):
        self.locationList["location1"].locationWinner(), self.locationList["location2"].locationWinner(), \
            self.locationList["location3"].locationWinner()
        results = [self.locationList["location1"].winning, self.locationList["location2"].winning,
                   self.locationList["location3"].winning]
        allywin, enemywin = 0, 0
        for string in results:
            if string == "Ally":
                allywin += 1
            elif string == "Enemy":
                enemywin += 1
        if allywin > enemywin:
            return "Ally"
        elif allywin < enemywin:
            return "Enemy"
        else:
            return self.resolveTie()

    def addUnit(self, unit_index, ally, locNum):
        selectedLoc = "location" + str(locNum)
        if ally:
            unit = self.status["allyhand"][unit_index]
            if self.status["allyenergy"] < unit.cur_cost:
                print("not enough energy")
                was_added = False
            else:
                was_added = self.locationList[selectedLoc].addToAllies(unit)
            if was_added:
                self.status["allyenergy"] -= unit.cur_cost
                del self.status["allyhand"][unit_index]

            print(self.locationList[selectedLoc].preRevealAllies)
            if was_added:
                print(unit, " was added to ", self.locationList[selectedLoc].name)
                unit.playCard(self.locationList[selectedLoc])
                move = {
                    "move_id": str(uuid.uuid4()),
                    "game_id": self.game_id,
                    "player": "player1",
                    "turn": self.status["turncounter"],
                    "card_played": unit.name,
                    "location": {
                        "name": self.locationList[selectedLoc].name,
                        "position": self.locationList[selectedLoc].locationNum,
                        "ally_cards": [
                            card.name for card in self.locationList[selectedLoc].allies
                        ],
                        "enemy_cards": [
                            card.name for card in self.locationList[selectedLoc].enemies
                        ],
                    }
                }
            print(unit, " was added?", was_added)
            return was_added
        else:
            unit = self.status["enemyhand"][unit_index]
            if self.status["enemyenergy"] < unit.cur_cost:
                print("not enough energy")
                was_added = False
            else:
                was_added = self.locationList[selectedLoc].addToEnemies(unit)
            if was_added:
                self.status["enemyenergy"] -= unit.cur_cost
                del self.status["enemyhand"][unit_index]

            if was_added:
                print(unit, " was added to ", self.locationList[selectedLoc].name)
                unit.playCard(self.locationList[selectedLoc])
                move = {
                    "move_id": str(uuid.uuid4()),
                    "game_id": self.game_id,
                    "player": "player2",
                    "turn": self.status["turncounter"],
                    "card_played": unit.name,
                    "location": {
                        "name": self.locationList[selectedLoc].name,
                        "position": self.locationList[selectedLoc].locationNum,
                        "ally_cards": [
                            card.name for card in self.locationList[selectedLoc].enemies
                        ],
                        "enemy_cards": [
                            card.name for card in self.locationList[selectedLoc].allies
                        ],
                    }
                }
                # self.registerMove(move)
            print(unit, " was added?", was_added)
            return was_added

    def undoActions(self, turnAlly, hand):
        loc1temp = self.locationList["location1"].undoActions(turnAlly)
        loc2temp = self.locationList["location2"].undoActions(turnAlly)
        loc3temp = self.locationList["location3"].undoActions(turnAlly)
        print("temps:", loc1temp, loc2temp, loc3temp)
        refund = 0
        for unit in loc1temp + loc2temp + loc3temp:
            refund += unit.cur_cost
        hand += loc1temp + loc2temp + loc3temp
        return refund

    def boardStatus(self):  # ritorna una stringa che definisce lo stato di ogni location
        print(self.locationList["location1"].name, "[", self.locationList["location1"].description, "]: ",
              self.locationList["location1"].locationStatus(), "")
        print(self.locationList["location2"].name, "[", self.locationList["location2"].description, "]: ",
              self.locationList["location2"].locationStatus(), "")
        print(self.locationList["location3"].name, "[", self.locationList["location3"].description, "]: ",
              self.locationList["location3"].locationStatus(), "")

    def draw(self, hand, deck, num):  # pesca un numero di carte dal deck
        i = 0
        if deck == []:
            print("No more cards in the deck!")
        else:
            while (i < num and len(hand) < 7):
                hand.append(deck[-1])
                del deck[-1]
                i += 1

    def gameStart(self):  # inserisci carte nel deck e pesca le carte
        ALL_CARDS = [
            cls for name, cls in inspect.getmembers(cards, inspect.isclass)
            if cls.__module__.startswith("cards") and cls is not cards.Card
        ]
        player_deck_classes = random.sample(ALL_CARDS, 12)
        enemy_deck_classes = random.sample(ALL_CARDS, 12)
        self.game_id = str(generate(size=10))
        self.game = {
            "game_id": self.game_id,
            "winner": "None",
            "start_time": datetime.utcfromtimestamp(time.time()).isoformat() + "Z",
            "end_time": '',
        }
        self.status["allydeck"] = [cls(True, self.status) for cls in player_deck_classes]
        self.status["enemydeck"] = [cls(False, self.status) for cls in enemy_deck_classes]
        random.shuffle(self.status["allydeck"])
        random.shuffle(self.status["enemydeck"])
        self.draw(self.status["allyhand"], self.status["allydeck"], 4)
        self.draw(self.status["enemyhand"], self.status["enemydeck"], 4)
        for location in self.locationList.values():
            location.startOfTurn()

    def moveSelection(self, card, location):
        for moves in card.location.cards_to_move:
            if moves[0] == card:
                print("You already moved that card")
                return "error"
        if location == card:
            print("You can't move the card to the same location")
        else:
            if card.moves_number > 0 or location.location_can_be_moved_to:
                if not location.checkIfLocationFull(card.ally):
                    card.location.cards_to_move.append([card, location])
                    return "success"
                else:
                    return "Location full"
            else:
                return "You can't move that card!"

    def snap(self, turnAlly):
        if (turnAlly and not self.status["allysnapped"]):
            self.status["allysnapped"] = True
            self.status["tempcubes"] *= 2

        elif (not turnAlly and not self.status["enemysnapped"]):
            self.status["enemysnapped"] = True
            self.status["tempcubes"] *= 2

    def startOfTurn(self):
        if self.status["turncounter"] == self.maxturns:
            self.status["tempcubes"] *= 2
        self.locationList["location1"].startOfTurn()
        self.locationList["location2"].startOfTurn()
        self.locationList["location3"].startOfTurn()
        self.status["allyenergy"] = self.status["allymaxenergy"] + self.status["tempenergyally"]
        self.status["enemyenergy"] = self.status["enemymaxenergy"] + self.status["tempenergyenemy"]
        self.status["tempenergyally"], self.status["tempenergyenemy"] = 0, 0
        winning = self.checkWinner()
        for card in self.locationList["location1"].allies + self.locationList["location2"].allies + self.locationList[
            "location3"].allies + self.locationList["location1"].enemies + self.locationList["location2"].enemies + \
                    self.locationList["location3"].enemies:
            card.startOfTurn()
        match winning:
            case "Ally" | "Tie":
                self.status["allypriority"] = True
                print("Allies have priority")
            case "Enemy":
                self.status["allypriority"] = False
                print("Enemies have priority")
        for card in self.status["allyhand"] + self.status["allydeck"] + self.status["enemyhand"] + self.status[
            "enemydeck"]:
            card.updateCard(self.locationList)
        self.draw(self.status["allyhand"], self.status["allydeck"], 1)
        self.draw(self.status["enemyhand"], self.status["enemydeck"], 1)

    def announcer(self):
        match self.status["allypriority"]:
            case True:
                print("Revealing ally cards")
            case False:
                print("Revealing enemy cards")

    def endOfTurn(self):
        if not (self.passStatus["retreatally"] or self.passStatus["retreatenemy"]): self.status["cubes"] = self.status[
            "tempcubes"]
        self.announcer()
        self.locationList["location1"].startOfTurnMoves(), self.locationList["location2"].startOfTurnMoves(), \
            self.locationList["location3"].startOfTurnMoves()
        self.locationList["location1"].revealCards(), self.locationList["location2"].revealCards(), self.locationList[
            "location3"].revealCards()
        self.status["allypriority"] = not self.status["allypriority"]
        self.announcer()
        self.locationList["location1"].revealCards(), self.locationList["location2"].revealCards(), self.locationList[
            "location3"].revealCards()
        print("End of turn ", self.status["turncounter"])
        self.locationList["location1"].endOfTurn(), self.locationList["location2"].endOfTurn(), self.locationList[
            "location3"].endOfTurn()
        self.status["turncounter"] += 1
        self.status["allymaxenergy"] += 1
        self.status["enemymaxenergy"] += 1

    def endGame(self):
        self.boardStatus()
        if (not self.passStatus["retreatally"] and not self.passStatus["retreatenemy"]):
            winner = self.checkWinner()
        else:
            winner = self.passStatus['winner']
        self.game_end = True
        match winner:
            case "Ally":
                print("Allies have won ", int(self.status["cubes"]))
                print("Enemies have lost ", int(self.status["cubes"]))
                self.game['winner'] = 'player1'
                self.passStatus['winner'] = 'Ally'
            case "Enemy":
                print("Allies have lost ", int(self.status["cubes"]))
                print("Enemies have won ", int(self.status["cubes"]))
                self.game['winner'] = 'player2'
                self.passStatus['winner'] = 'Enemy'
            case "Tie":
                print("Tie!")
                self.game['winner'] = 'Tie'
                self.passStatus['winner'] = 'Tie'

        self.game['end_time'] = datetime.utcfromtimestamp(time.time()).isoformat() + "Z"

    def turnEnd(self, training):
        last_turn = (self.status["turncounter"] == self.status["maxturns"])
        self.endOfTurn()
        self.passStatus['turnend'] = True
        if training and last_turn:
            self.endGame()
        elif training:
            self.startOfTurn()

    def retreat(self, ally: bool):
        if ally:
            self.passStatus['retreatally'] = True
            self.passStatus['winner'] = "Enemy"
        else:
            self.passStatus['retreatenemy'] = True
            self.passStatus['winner'] = "Ally"
        self.endGame()

    def getHand(self, agent):
        if agent == "player_1":
            return self.status["allyhand"]
        elif agent == "player_2":
            return self.status["enemyhand"]


'''
game = GameState()

game.gameStart()
'''
