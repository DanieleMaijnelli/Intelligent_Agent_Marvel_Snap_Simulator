from cards import Card


class Sentinel(Card):
    def __init__(self, ally, status):
        super().__init__(2, 3, "Sentinel", ally, status)
        self.description = "On Reveal: Add another Sentinel card to your hand"

    def onReveal(self, locationlist):
        if self.ally:
            self.status["allyhand"].append(Sentinel(self.ally, self.status))
        else:
            self.status["enemyhand"].append(Sentinel(self.ally, self.status))

        print("Added another Sentinel card to your hand")
