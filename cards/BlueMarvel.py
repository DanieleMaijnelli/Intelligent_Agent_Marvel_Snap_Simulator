from cards import Card


class Bluemarvel(Card):
    def __init__(self, ally, status):
        super().__init__(5, 3, "Blue Marvel", ally, status)
        self.description = "Ongoing: Your other cards have +1 Power."
        self.has_ongoing = True

    def applyOngoing(self, locationlist):
        if self.ally:
            for unit in locationlist["location1"].allies + locationlist["location2"].allies + locationlist[
                "location3"].allies:
                if unit != self:
                    unit.ongoing_to_apply.append(self)
        else:
            for unit in locationlist["location1"].enemies + locationlist["location2"].enemies + locationlist[
                "location3"].enemies:
                if unit != self:
                    unit.ongoing_to_apply.append(self)

    def ongoing(self, card):
        card.ongoing_buff += 1
