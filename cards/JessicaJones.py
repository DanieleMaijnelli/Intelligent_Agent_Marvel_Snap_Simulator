from cards import Card


class Jessicajones(Card):
    def __init__(self, ally, status):
        super().__init__(4, 5, "Jessica Jones", ally, status)
        self.description = "On Reveal: If you don't play a card at this location next turn, +5 Power"
        self.turnToCheck = 0
        self.onRevealNum = 0

    def onReveal(self, locationlist):
        self.onRevealNum += 1
        self.locationNumToCheck = self.location.locationNum
        self.turnToCheck = self.status["turncounter"] + 1

    def endOfTurn(self):
        check = True
        if self.status["turncounter"] == self.turnToCheck and self.onRevealNum != 0:
            for cardPlayed in self.status["cardsplayed"]:
                if cardPlayed[1] == self.turnToCheck and cardPlayed[2] == self.locationNumToCheck and cardPlayed[
                    0].ally == self.ally:
                    check = False
                    break
            if check:
                self.onreveal_buff += 5 * self.onRevealNum
