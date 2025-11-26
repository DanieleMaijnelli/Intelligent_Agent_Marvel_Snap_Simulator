from Locations.Location import Location
import Locations
from flask import url_for, current_app
import os


class Card:
    def __init__(self, cost, power, name, ally, status):
        self.base_cost = cost
        self.base_power = power
        self.name = name
        self.location = 0
        self.ally = ally
        self.cur_power = self.base_power
        self.onreveal_buff = 0
        self.has_ongoing = False
        self.has_ongoing_buff = False
        self.status = status
        self.activate_while_in_hand = False
        self.description = "Does nothing"
        self.can_be_destroyed = True
        self.has_ongoing_late = False
        self.ongoing_buff = 0
        self.ongoing_to_apply = []
        self.has_ongoing_buffpower = False
        self.can_move = 0
        self.onslaught = True
        self.can_be_played = True
        self.moves_number = 0
        self.was_created = False
        self.has_start_of_game = False
        self.activate_in_deck = False
        self.activate_on_destroy = False
        self.cost = self.base_cost
        self.cur_cost = self.cost
        self.cost_ongoing = 0

    def __repr__(self):
        return self.name

    def onReveal(self, locationlist):
        pass

    def playCard(self, location):
        print("Played ", self.name, " in ", location.name)
        self.location = location

    def startOfTurn(self):
        pass

    def endOfTurn(self):
        pass

    def whenDestroyed(self, locationlist):
        print("Destroyed ", self.name)
        return True

    def updateCard(self, locationlist):
        self.canbedestroyed = True
        self.ongoing_buff = 0
        self.cost_ongoing = 0
        for buff in self.ongoing_to_apply:
            buff.ongoing(self)
        self.cur_cost = self.cost + self.cost_ongoing
        if self.cur_cost < 0: self.cur_cost = 0
        self.cur_power = self.base_power + self.onreveal_buff + self.ongoing_buff
        self.ongoing_to_apply = []

    def nextCardBuff(self, card):
        pass

    def ongoing(self, locationlist):
        pass

    def onMove(self):
        pass

    def move(self, newloc):
        if self.ally:
            next = newloc.allies
        else:
            next = newloc.enemies

        if len(next) < 4 and newloc != self.location:
            self.location.removeCard(self)
            self.location = newloc
            self.onMove()
            next.append(self)
            newloc.moveEffects(self)

    def onCardBeingMoved(self, card):
        pass

    def onCardBeingPlayed(self, card):
        pass

    def discard(self):
        print("Discarding ", self.name)
        try:
            if self.ally and self in self.status["allyhand"]:
                self.status["allyhand"].remove(self)
            elif not self.ally and self in self.status["enemyhand"]:
                self.status["enemyhand"].remove(self)
            if self.ally:
                self.status["alliesdiscarded"].append(self)
            else:
                self.status["enemiesdiscarded"].append(self)
            self.whenDiscarded()
        except:
            print("Card not in hand")

    def whenDiscarded(self):
        pass

    def startOfGame(self, locationlist):
        pass

    def render(self):
        image_path = f'assets/{self.name.replace(" ", "").capitalize()}.webp'
        full_path = os.path.join(current_app.static_folder, image_path)
        print(full_path)
        return url_for('static', filename=image_path)
