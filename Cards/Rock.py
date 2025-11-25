from cards import Card


class Rock(Card):
    def __init__(self, ally, status):
        super().__init__(1, 0, "Rock", ally, status)
        self.description = "Rock!"
