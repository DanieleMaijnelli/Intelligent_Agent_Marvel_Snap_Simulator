from cards import Card
import random


class Whitetiger(Card):
    def __init__(self, ally, status):
        super().__init__(5, 1, "White Tiger", ally, status)
        self.description = "On Reveal: Add a 8-Power Tiger to another location."

    def onReveal(self, locationlist):
        possible = []
        if self.ally:
            for loc in locationlist.values():
                if loc != self.location and len(loc.allies) < 4:
                    possible.append(loc)
        else:
            for loc in locationlist.values():
                if loc != self.location and len(loc.enemies) < 4:
                    possible.append(loc)
        if len(possible) > 0:
            loc = random.choice(possible)
            if self.ally:
                tiger = Card(5, 8, "Tiger", self.ally, self.status)
                tiger.location = loc
                loc.allies.append(tiger)
            else:
                tiger = Card(5, 8, "Tiger", self.ally, self.status)
                tiger.location = loc
                loc.enemies.append(tiger)
        else:
            print("No possible locations")
