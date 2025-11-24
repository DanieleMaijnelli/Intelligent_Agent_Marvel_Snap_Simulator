from cards import Card


class Colossus(Card):
    def __init__(self, ally, status):
        super().__init__(2, 3, "Colossus", ally, status)
        self.has_ongoing = True
        self.description = "Ongoing: This card can't be destroyed, moved or have it's power reduced"
        self.onreveal_to_check = 0
        self.ongoing_to_check = 0

    def applyOngoing(self, locationlist):
        self.ongoing_to_apply.append(self)

    def ongoing(self, locationlist):
        self.can_be_destroyed = False
        self.onreveal_buff = self.onreveal_to_check = max(self.onreveal_to_check, self.onreveal_buff)
        self.ongoing_buff = self.ongoing_to_check = max(self.ongoing_to_check, self.ongoing_buff)

    def move(self, location):
        if self.has_ongoing:
            print("Colossus can't be moved!")
        else:
            super().move(location)
