from cards import Card
import random
import os, importlib, inspect
import flask


class Agent13(Card):
    def __init__(self, ally, status):
        super().__init__(1, 2, "Agent 13", ally, status)
        self.description = "On Reveal: Add a random card to your hand."

    def randomCard(self):
        location_folder = os.path.dirname(__file__)
        files = [
            f for f in os.listdir(location_folder)
            if f.endswith(".py") and f != "Card.py" and not f.startswith("__")
        ]

        chosen_file = random.choice(files)
        module_name = chosen_file[:-3]
        full_module = f"{__package__}.{module_name}" if __package__ else module_name

        module = importlib.import_module(full_module)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Card) and obj is not Card:
                random_card = obj
                break
        return random_card(self.ally, self.status)

    def onReveal(self, locationlist):
        if self.ally:
            self.status["allyhand"].append(self.randomCard())
        else:
            self.status["enemyhand"].append(self.randomCard())

    def render(self):
        return flask.url_for('static', filename='assets/Agent13.webp')
