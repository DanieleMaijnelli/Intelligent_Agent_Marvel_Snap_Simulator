import cards
import inspect
import random

ALL_CARDS = [
    cls for name, cls in inspect.getmembers(cards, inspect.isclass)
    if cls.__module__.startswith("cards") and cls is not cards.Card
]

CLASS_TO_INDEX = {cls: idx for idx, cls in enumerate(ALL_CARDS)}


def form_random_deck():
    deck = random.sample(ALL_CARDS, 12)
    return deck


def form_random_basic_deck():
    basic_decks = [[
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
    ], [
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
    ], [
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
    ], [
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
    ]]

    chosen_deck = random.choice(basic_decks)
    deck = [
        cls
        for cls in ALL_CARDS
        if cls.__name__ in chosen_deck
    ]
    return deck


'''for i in range(10):
    deck = form_random_basic_deck()
    print(deck)
    print(len(deck))'''
