from cards import Card


class Devildinosaur(Card):
    def __init__(self, ally, status):
        super().__init__(5, 4, "Devil Dinosaur", ally, status)
        self.description = "Ongoing: +2 Power for each card in your hand."
        self.has_ongoing = True

    def applyOngoing(self, locationlist):
        self.ongoing_to_apply.append(self)

    def ongoing(self, locationlist):
        if self.ally:
            self.ongoing_buff += 2 * len(self.status["allyhand"])
        else:
            self.ongoing_buff += 2 * len(self.status["enemyhand"])
