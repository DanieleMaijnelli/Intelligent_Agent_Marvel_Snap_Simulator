from cards import Card


class Buckybarnes(Card):
    def __init__(self, ally, status):
        super().__init__(2, 1, "Bucky Barnes", ally, status)
        self.description = "When this is destroyed, replace it with the Winter Soldier."

    class WinterSoldier(Card):
        def __init__(self, ally, status):
            super().__init__(2, 7, "Winter Soldier", ally, status)
            self.description = "It's time for me to face my past"

    def whenDestroyed(self, locationlist):
        if self.ally:
            newcard = self.WinterSoldier(self.ally, self.status)
            newcard.location = self.location
            locationlist["location1"].allies.append(newcard)
        else:
            newcard = self.WinterSoldier(self.ally, self.status)
            newcard.location = self.location
            locationlist["location1"].enemies.append(newcard)
