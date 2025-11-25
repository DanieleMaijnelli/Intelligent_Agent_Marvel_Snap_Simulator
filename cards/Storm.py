from cards import Card
from Locations.Location import Location


class Storm(Card):
    def __init__(self, ally, status):
        super().__init__(4, 5, "Storm", ally, status)
        self.description = "On Reveal: Flood this location. Next turn is the last turn cards can be played here"

    class Flooding(Location):
        def __init__(self, number, status, locationlist):
            super().__init__(number, status, locationlist)
            self.name = "Flooding"
            self.description = "This is the last turn cards can be played here"
            self.counter = 1

        def startOfTurn(self):
            super().startOfTurn()
            self.counter -= 1

        class Flooded(Location):
            def __init__(self, number, status, locationlist):
                super().__init__(number, status, locationlist)
                self.name = "Flooded"
                self.description = "Cards can't be played here"
                self.can_play_cards_base = False
                self.can_play_cards_allies = False
                self.can_play_cards_enemies = False

        def endOfTurn(self):
            super().endOfTurn()
            if self.counter == 0:
                self.changeLocation(self.Flooded(self.locationNum, self.status, self.locationlist))

    def onReveal(self, locationlist):
        self.location.changeLocation(self.Flooding(0, self.status, locationlist))
