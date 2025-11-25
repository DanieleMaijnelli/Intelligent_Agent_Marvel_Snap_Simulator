from cards import Card


class Squirrelgirl(Card):
    def __init__(self, ally, status):
        super().__init__(1, 2, "Squirrel Girl", ally, status)
        self.description = "On Reveal: Add a 1-Power Squirrel to each other location"

    class Squirrel(Card):
        def __init__(self, ally, status):
            super().__init__(1, 1, "Squirrel", ally, status)
            self.description = "Squeak!"

    def onReveal(self, locationlist):
        if self.ally:
            for location in locationlist.values():
                if location != self.location and len(location.allies) < 4:
                    newSquirrel = self.Squirrel(self.ally, self.status)
                    location.allies.append(newSquirrel)
                    newSquirrel.location = location
        else:
            for location in locationlist.values():
                if location != self.location and len(location.enemies) < 4:
                    newSquirrel = self.Squirrel(self.ally, self.status)
                    location.enemies.append(newSquirrel)
                    newSquirrel.location = location
