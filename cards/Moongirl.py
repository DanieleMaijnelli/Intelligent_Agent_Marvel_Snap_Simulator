from cards import Card
import copy


class Moongirl(Card):
    def __init__(self, ally, status):
        super().__init__(4, 5, "Moongirl", ally, status)
        self.description = "On Reveal: Duplicate your hand."

    def onReveal(self, locationlist):
        if self.ally:
            hand = self.status["allyhand"]
        else:
            hand = self.status["enemyhand"]

        original_cards = list(hand)

        for card in original_cards:
            if len(hand) >= 7:
                break
            new_card = copy.deepcopy(card)
            new_card.status = self.status
            hand.append(copy.deepcopy(new_card))
