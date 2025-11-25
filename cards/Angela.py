from cards import Card


class Angela(Card):
    def __init__(self, ally, status):
        super().__init__(2, 3, "Angela", ally, status)
        self.description = "After you play a card here, +1 Power"

    def onCardBeingPlayed(self, card):
        if card.ally == self.ally and card.location == self.location and card != self:
            self.onreveal_buff += 1
