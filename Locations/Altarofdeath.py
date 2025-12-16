from Locations import Location

class AltarOfDeath(Location):
    def __init__(self, number, status, locationlist):
        super().__init__(number, status, locationlist)
        self.name = "Altar of Death"
        self.description = "After you play a card here, destroy it to get +2 Energy next turn."
    
    def onPlayEffect(self, card):
        if card.location == self:
            #print("Destroying ", card.name, " to get +2 Energy next turn.")
            if card.ally:
                self.status["tempenergyally"] += 2
            else:
                self.status["tempenergyenemy"] += 2
            if card.has_ongoing:
                card.applyOngoing(self.locationlist)
            self.destroyCard(card)