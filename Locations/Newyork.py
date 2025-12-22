from Locations import Location 

class Newyork(Location):
    def __init__(self, number, status, locationlist):
        super().__init__(number, status, locationlist)
        self.name = "New York"
        self.description = "On turn 5, you can move cards here"
    
    def startOfTurn(self):
        super().startOfTurn()
        if self.status["turncounter"] == 5:
            #print("You can move cards here!!")
            self.location_can_be_moved_to = True