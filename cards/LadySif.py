from cards import Card
import random


class Ladysif(Card):
    def __init__(self, ally, status):
        super().__init__(3, 5, "Lady Sif", ally, status)
        self.description = "On Reveal: Discard the highest cost card in your hand"

    def onReveal(self, locationlist):
        if self.ally:
            handToCheck = self.status["allyhand"]
        else:
            handToCheck = self.status["enemyhand"]

        highestCost = 0
        highestCostCard = []
        for card in handToCheck:
            if card.cur_cost == highestCost:
                highestCostCard.append(card)
            elif card.cur_cost > highestCost:
                highestCost = card.cur_cost
                highestCostCard = [card]

        if len(highestCostCard) > 0:
            random.choice(highestCostCard).discard()
