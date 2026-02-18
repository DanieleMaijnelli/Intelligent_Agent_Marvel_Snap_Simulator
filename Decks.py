import cards
import inspect
import random
import Locations

ALL_CARDS = [
    cls for name, cls in inspect.getmembers(cards, inspect.isclass)
    if cls.__module__.startswith("cards") and cls is not cards.Card
]

CLASS_TO_INDEX = {cls: idx for idx, cls in enumerate(ALL_CARDS)}

ALL_LOCATIONS = [
    cls for name, cls in inspect.getmembers(Locations, inspect.isclass)
    if cls.__module__.startswith("Locations") and cls is not Locations.Location
]

LOCATION_CLASS_TO_INDEX = {cls: idx for idx, cls in enumerate(ALL_LOCATIONS)}

deck_1 = [
    "Elektra",
    "Nightcrawler",
    "Squirrelgirl",
    "Mistersinister",
    "Cable",
    "Ironheart",
    "Wolfsbane",
    "Jessicajones",
    "Whitetiger",
    "Bluemarvel",
    "Enchantress",
    "Odin"
]

deck_2 = [
    "Antman",
    "Elektra",
    "Nightcrawler",
    "Squirrelgirl",
    "Blade",
    "Angela",
    "Wolverine",
    "Strongguy",
    "Ladysif",
    "Swordmaster",
    "Kazar",
    "Bluemarvel"
]

deck_3 = [
    "Antman",
    "Nightcrawler",
    "Angela",
    "Armor",
    "Colossus",
    "Mrfantastic",
    "Cosmo",
    "Namor",
    "Ironman",
    "Klaw",
    "Spectrum",
    "Onslaught"
]

deck_4 = [
    "Nova",
    "Agent13",
    "Buckybarnes",
    "Carnage",
    "Sabretooth",
    "Cable",
    "Sentinel",
    "Wolverine",
    "Killmonger",
    "Deathlok",
    "Moongirl",
    "Devildinosaur"
]

deck_5 = [
    "Squirrelgirl",
    "Mistersinister",
    "Cable",
    "Ironheart",
    "Nova",
    "Agent13",
    "Heimdall",
    "Bishop",
    "Forge",
    "Iceman",
    "Okoye",
    "Nakia"
]

deck_6 = [
    "Ironfist",
    "Kraven",
    "Vulture",
    "Doctorstrange",
    "Lizard",
    "Whitequeen",
    "Heimdall",
    "Bishop",
    "Forge",
    "Iceman",
    "Mantis",
    "Okoye"
]


def form_random_deck():
    deck = random.sample(ALL_CARDS, 12)
    return deck


def form_random_training_deck():
    basic_decks = [deck_1, deck_2, deck_3, deck_4]

    idx0 = random.randrange(len(basic_decks))
    chosen_deck = basic_decks[idx0]
    deck_number = idx0 + 1

    deck = [cls for cls in ALL_CARDS if cls.__name__ in chosen_deck]
    return deck, deck_number


def form_hybrid_deck():
    hybrid_deck = deck_5
    deck_number = 5

    deck = [cls for cls in ALL_CARDS if cls.__name__ in hybrid_deck]
    return deck, deck_number


def form_new_deck():
    new_deck = deck_6
    deck_number = 6

    deck = [cls for cls in ALL_CARDS if cls.__name__ in new_deck]
    return deck, deck_number


'''
for i in range(10):
    deck, _ = form_random_training_deck()
    print(deck)
    print(len(deck))

try_deck, _ = form_hybrid_deck()
test_deck, _ = form_new_deck()
print(try_deck)
print(len(try_deck))
print(test_deck)
print(len(test_deck))
'''
