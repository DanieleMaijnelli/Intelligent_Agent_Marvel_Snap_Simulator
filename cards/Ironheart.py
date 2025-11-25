from cards import Card


class Ironheart(Card):
    def __init__(self, ally, status):
        super().__init__(3, 0, "Ironheart", ally, status)
        self.description = "On Reveal: Give 3 of your other cards +2 Power"

    def onReveal(self, locationlist):
        alreadyApplied = []
        if self.ally:
            for card in locationlist["location1"].allies + locationlist["location2"].allies + locationlist[
                "location3"].allies:
                if card != self and len(alreadyApplied) < 3:
                    alreadyApplied.append(card)
                    card.onreveal_buff += 2
        else:
            for card in locationlist["location1"].enemies + locationlist["location2"].enemies + locationlist[
                "location3"].enemies:
                if card != self and len(alreadyApplied) < 3:
                    alreadyApplied.append(card)
                    card.onreveal_buff += 2
