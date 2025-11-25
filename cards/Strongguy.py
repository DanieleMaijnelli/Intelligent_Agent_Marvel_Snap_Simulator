from cards import Card


class Strongguy(Card):
    def __init__(self, ally, status):
        super().__init__(3, 3, "Strong Guy", ally, status)
        self.description = "Ongoing: if your hand has 1 or fewer cards, +6 power"
        self.has_ongoing = True

    def ongoing(self, card):
        self.ongoing_buff += 6

    def applyOngoing(self, locationlist):
        if self.ally:
            if len(self.status["allyhand"]) <= 1:
                self.ongoing_to_apply.append(self)
        else:
            if len(self.status["enemyhand"]) <= 1:
                self.ongoing_to_apply.append(self)
