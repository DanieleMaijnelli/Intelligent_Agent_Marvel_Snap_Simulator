from cards import Card


class Mistersinister(Card):
    def __init__(self, ally, status):
        super().__init__(2, 2, "Mister Sinister", ally, status)
        self.description = "On Reveal: Add a Sinister Clone here with the same power"

    class SinisterClone(Card):
        def __init__(self, ally, power, status):
            super().__init__(2, power, "Sinister Clone", ally, status)
            self.description = "Clone of Mister Sinister"

    def onReveal(self, locationlist):
        if self.ally and len(self.location.allies) < 4:
            newSinisterClone = self.SinisterClone(self.ally, self.cur_power, self.status)
            self.location.allies.append(newSinisterClone)
            newSinisterClone.location = self.location
        elif not self.ally and len(self.location.enemies) < 4:
            newSinisterClone = self.SinisterClone(self.ally, self.cur_power, self.status)
            self.location.enemies.append(newSinisterClone)
            newSinisterClone.location = self.location
