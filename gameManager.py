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

class GameState():
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
        self.allymaxenergy, self.enemymaxenergy =1,1
        self.turncounter = 1
        self.maxturns = 6
        self.game_end = False
        self.locationList = {"location1": 0,"location2": 0, "location3": 0}
        self.status = {"maxturns": self.maxturns,"allymaxenergy": self.allymaxenergy,
                "enemymaxenergy": self.enemymaxenergy, "allyenergy": 1,
                    "enemyenergy":1, "turncounter":self.turncounter,
                    "tempenergyally":0, "tempenergyenemy":0,
                    "allyhand":[], "enemyhand":[],
                    "allydeck":[], "enemydeck":[],
                    "alliesdestroyed":[], "enemiesdestroyed":[],
                    "alliesdiscarded":[], "enemiesdiscarded":[],
                    "allypriority": True,
                    "cubes":1, "tempcubes":1,
                    "allysnapped":False, "enemysnapped": False,
                    "cardsplayed": [], "onnextcardbeingplayed": [],
                    "allypass": False, "enemypass": False,
                    "endofturncounterally":0, "endofturncounterenemy":0,
                    "locationlist": self.locationList}
        self.passStatus = {
                'turnpassally': self.status['allypass'],  
                'turnpassenemy': self.status['enemypass'],
                'winner': "None",
                'retreatally': False,
                'retreatenemy': False,
                'turnend': False 
            }
        self.locationList["location1"]=TemporaryLocation(1,self.status,self.locationList)
        self.locationList["location2"]= TemporaryLocation(2,self.status,self.locationList)
        self.locationList["location3"]= TemporaryLocation(3,self.status,self.locationList)
        

    def reset(self):
            self.exit = False
            self.turnAlly = False
            self.allymaxenergy, self.enemymaxenergy = 1, 1
            self.turncounter = 1
            self.maxturns = 6
            self.game_end = False
            self.locationList = {"location1": 0, "location2": 0, "location3": 0}
            self.status = {"maxturns": self.maxturns,"allymaxenergy":self.allymaxenergy,
                "enemymaxenergy": self.enemymaxenergy, "allyenergy": 1,
                    "enemyenergy":1, "turncounter":self.turncounter,
                    "tempenergyally":0, "tempenergyenemy":0,
                    "allyhand":[], "enemyhand":[],
                    "allydeck":[], "enemydeck":[],
                    "alliesdestroyed":[], "enemiesdestroyed":[],
                    "alliesdiscarded":[], "enemiesdiscarded":[],
                    "allypriority": True,
                    "cubes":1, "tempcubes":1,
                    "allysnapped":False, "enemysnapped": False,
                    "cardsplayed": [], "onnextcardbeingplayed": [],
                    "allypass": False, "enemypass": False,
                    "endofturncounterally":0, "endofturncounterenemy":0,
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
        allypower = self.locationList["location1"].alliesPower + self.locationList["location2"].alliesPower + self.locationList["location3"].alliesPower
        enemypower = self.locationList["location1"].enemiesPower + self.locationList["location2"].enemiesPower + self.locationList["location3"].enemiesPower
        if allypower > enemypower:
            return "Ally"
        elif allypower < enemypower:
            return "Enemy"  
        else:
            return "Tie"

    def checkWinner(self):
        self.locationList["location1"].locationWinner(), self.locationList["location2"].locationWinner(), self.locationList["location3"].locationWinner()
        results = [self.locationList["location1"].winning,self.locationList["location2"].winning,self.locationList["location3"].winning]
        allywin, enemywin = 0,0
        for string in results:
            if string == "Ally":
                allywin +=1
            elif string == "Enemy":
                enemywin +=1
        if allywin > enemywin:
            return "Ally"
        elif allywin < enemywin:
            return "Enemy"
        else:
            return self.resolveTie()

    def registerMove(self,move):
        print("Registering move")
        with open("matchlogs/schema/schema-move.json", "r") as schema_file:
            schema = json.load(schema_file)
        try:
            jsonschema.validate(move, schema)
        except jsonschema.exceptions.ValidationError as e:
            print(e)
        directory = "matchlogs"
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, f"moves/move-{self.version}-data.json")
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
        else:
            data = []
        
        if any(existing["move_id"] == move["move_id"] for existing in data):
            print("ignored")
            return
        data.append(move)
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    def addUnit(self,unit,ally, locNum):
        selectedLoc = "location" + str(locNum)
        if ally:
            was_added = self.locationList[selectedLoc].addToAllies(unit)
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
                #self.registerMove(move)
            return was_added
        else:
            was_added = self.locationList[selectedLoc].addToEnemies(unit)
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
                        "enemy_cards":[
                            card.name for card in self.locationList[selectedLoc].allies
                        ],
                    }
                }
                #self.registerMove(move)
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

    def boardStatus(self): #ritorna una stringa che definisce lo stato di ogni location 
        print(self.locationList["location1"].name,"[", self.locationList["location1"].description, "]: ", self.locationList["location1"].locationStatus(),"")
        print(self.locationList["location2"].name,"[", self.locationList["location2"].description, "]: ", self.locationList["location2"].locationStatus(),"")
        print(self.locationList["location3"].name,"[", self.locationList["location3"].description, "]: ", self.locationList["location3"].locationStatus(),"")

    def draw(self,hand,deck,num): #pesca un numero di carte dal deck 
        i=0
        if deck == []:
            print("No more cards in the deck!")
        else:
            while i<num:
                hand.append(deck[-1])
                del deck[-1]
                i+=1

    def gameStart(self): #inserisci carte nel deck e pesca le carte
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
        self.draw(self.status["allyhand"],self.status["allydeck"],4)
        self.draw(self.status["enemyhand"],self.status["enemydeck"],4)
        for location in self.locationList.values():
            location.startOfTurn()

    def playerTurn(self, hand, deck,energy):
        self.draw(hand,deck,1)
        for location in self.locationList.values():
            location.updateGameState()
        playerpass = False
        turnenergy = energy
        while not playerpass:
            print()
            print("Press 1 to check hand and current energy,"
            " 2 to add an unit to the board,"
            " 3 to pass,"
            " 4 to check board status,"
            " 5 to undo your actions, "
            "6 to move a card, "
            "7 to retreat,"
            " 8 to SNAP!")
            check = True
            while check:
                try:
                    userInput = int(input("What do you want to do? "))
                    check= False
                except:
                    print("InputError")
            match userInput:
                case 1:
                    print("Energy left: ", turnenergy)
                    i=1
                    for unit in hand:
                        print(i,": ",unit.name, "Cost:", unit.cur_cost," Power: ", unit.cur_power, " Description:", unit.description)
                        i+=1
                case 2:
                    print("Energy left:", turnenergy)
                    print("Which unit would you like to add")
                    i=1
                    for unit in hand:
                        print(i,": ",unit.name, "Cost:", unit.cur_cost," Power: ", unit.cur_power, " Description:", unit.description)
                        i+=1
                    
                case 3:
                    playerpass = True
                    return turnenergy
                case 4:
                    self.boardStatus()
                case 5:
                    turnenergy += self.undoActions(self.turnAlly, hand)
                case 6:
                    print("Which card would you like to move?")
                    self.moveSelection(self.turnAlly)
                case 8:
                    if (self.turnAlly and not self.status["allysnapped"]) or (not self.turnAlly and not self.status["enemysnapped"]):
                        print("SNAP!")
                        self.snap(self.status, self.turnAlly)
                    else: print("You already snapped!")

                case _:
                    print("Input error")

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
            if self.status["tempcubes"] == self.status["cubes"]: self.status["tempcubes"] *= 2
            else: self.status["cubes"], self.status["tempcubes"] = 4,4
        elif (not turnAlly and not self.status["enemysnapped"]):
            self.status["enemysnapped"] = True
            if self.status["tempcubes"] == self.status["cubes"]: self.status["tempcubes"] *= 2
            else: self.status["cubes"], self.status["tempcubes"] = 4,4

    def startOfTurn(self):
        self.locationList["location1"].startOfTurn()
        self.locationList["location2"].startOfTurn()
        self.locationList["location3"].startOfTurn()
        self.status["allyenergy"] = self.status["allymaxenergy"] + self.status["tempenergyally"]
        self.status["enemyenergy"] = self.status["enemymaxenergy"] + self.status["tempenergyenemy"]
        self.status["tempenergyally"], self.status["tempenergyenemy"] = 0,0
        winning = self.checkWinner()
        for card in self.locationList["location1"].allies + self.locationList["location2"].allies + self.locationList["location3"].allies + self.locationList["location1"].enemies + self.locationList["location2"].enemies + self.locationList["location3"].enemies:
            card.startOfTurn()
        match winning:
            case "Ally" | "Tie":
                self.status["allypriority"] = True
                print("Allies have priority")
            case "Enemy":
                self.status["allypriority"] = False
                print("Enemies have priority")
        for card in self.status["allyhand"] + self.status["allydeck"] + self.status ["enemyhand"] + self.status["enemydeck"]:
            card.updateCard(self.locationList)
        self.draw(self.status["allyhand"],self.status["allydeck"],1)
        self.draw(self.status["enemyhand"],self.status["enemydeck"],1)
    def announcer(self):
        match self.status["allypriority"]:
            case True:
                print("Revealing ally cards")
            case False:
                print("Revealing enemy cards")

    def endOfTurn(self):
        if not(self.passStatus["retreatally"] or self.passStatus["retreatenemy"]): self.status["cubes"] = self.status["tempcubes"]
        self.announcer()
        self.locationList["location1"].startOfTurnMoves(), self.locationList["location2"].startOfTurnMoves(), self.locationList["location3"].startOfTurnMoves()
        self.locationList["location1"].revealCards(), self.locationList["location2"].revealCards(), self.locationList["location3"].revealCards()
        self.status["allypriority"] = not self.status["allypriority"]
        self.announcer()
        self.locationList["location1"].revealCards(), self.locationList["location2"].revealCards(), self.locationList["location3"].revealCards()
        print("End of turn ", self.status["turncounter"])
        self.locationList["location1"].endOfTurn(), self.locationList["location2"].endOfTurn(), self.locationList["location3"].endOfTurn()
        self.status["turncounter"] +=1
        self.status["allymaxenergy"]+=1
        self.status["enemymaxenergy"]+=1


    def endGame(self):
        self.boardStatus()
        winner = self.checkWinner()
        self.game_end = True
        if self.status['turncounter'] == self.status['maxturns']:
            match winner:
                case "Ally":
                    print("Allies have won ", self.status["cubes"]*2)
                    print("Enemies have lost ", self.status["cubes"]*2)
                    self.game['winner'] = 'player1'
                    self.passStatus['winner'] = 'player1'
                case "Enemy":
                    print("Allies have lost ", self.status["cubes"]*2)
                    print("Enemies have won ", self.status["cubes"]*2)
                    self.game['winner'] = 'player2'
                    self.passStatus['winner'] = 'player2'
                case "Tie":
                    print("Tie!")
                    self.game['winner'] = 'Tie'
                    self.passStatus['winner'] = 'Tie'
        else: 
            match self.passStatus['winner']:
                case "Ally":
                    self.game['winner'] = 'player1'
                    self.passStatus['winner'] = 'player1'
                case "Enemy":
                    self.game['winner'] = 'player2'
                    self.passStatus['winner'] = 'player2'
                case "Tie":
                    self.game['winner'] = 'Tie'
                    self.passStatus['winner'] = 'Tie'
        self.game['end_time'] = datetime.utcfromtimestamp(time.time()).isoformat() + "Z"
        #self.registerGame(self.game)
    
    def registerGame(self,game):
        print("registering!")
        with open("matchlogs/schema/schema-game.json", "r") as schema_file:
            schema = json.load(schema_file)
        try:
            jsonschema.validate(game, schema)
        except jsonschema.exceptions.ValidationError as e:
            print(e)
        directory = "matchlogs"
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, f"games/game-{self.version}-data.json")
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
        else:
            data = []
        
        if any(existing["game_id"] == game["game_id"] for existing in data):
            print("ignored")
            return
        data.append(game)

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)


    def gaming(self):
        while self.status["turncounter"]<=self.status["maxturns"]:
            print("Turn ", self.status["turncounter"],", player turn")
            print("")
            self.startOfTurn(self.status)
            self.boardStatus()
            turnAlly = not turnAlly
            print("Turn ", self.status["turncounter"],", enemy turn")
            turnAlly = not turnAlly
            self.status["enemyenergy"]= self.playerTurn(self.status["enemyhand"], self.status["enemydeck"], self.status["enemyenergy"])
            self.endOfTurn()
        self.endGame()
    
    def turnEnd(self, training):
        self.endOfTurn()
        self.passStatus['turnend'] = True
        if training and self.status["turncounter"] == self.status["maxturns"]: self.endGame()
        elif training: self.startOfTurn()
    
    def retreat(allyOrEnemy):
        pass

    def getHand(self, agent):
        if agent == "player_1":
            return self.status["allyhand"]
        elif agent == "player_2":
            return self.status["enemyhand"]

game = GameState()

game.gameStart()