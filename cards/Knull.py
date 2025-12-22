from cards import Card

class Knull(Card):
    def __init__(self, ally, status):
        super().__init__(1, 0, "Knull", ally, status)
        self.has_ongoing = True
        self.description = "Ongoing: Has the combined attack of all destroyed cards"
    
    def ongoing(self,card):
        for unit in self.status["alliesdestroyed"] + self.status["enemiesdestroyed"]:
            self.ongoing_buff += unit.cur_power
    
    def applyOngoing(self, locationlist):
        self.ongoing_to_apply.append(self)
