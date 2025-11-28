from Locations import Location 

class Asgard(Location):
    def __init__(self, number, status, locationlist):
        super().__init__(number, status, locationlist)
        self.name = "Asgard"
        self.description = "After turn 4, whoever is winning here draws 2 cards"
    
    def endOfTurn(self):
        super().endOfTurn()
        if self.status["turncounter"] == 4:
            if self.winning == "Ally":
                for i in range(2):
                    if self.status["allydeck"] != []:
                        self.status["allyhand"].append(self.status["allydeck"].pop())
            elif self.winning == "Enemy":
                for i in range(2):
                    if self.status["enemydeck"] != []:
                        self.status["enemyhand"].append(self.status["enemydeck"].pop())