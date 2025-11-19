import cards
import inspect
import random

ALL_CARDS = [
    cls for name, cls in inspect.getmembers(cards, inspect.isclass)
    if cls.__module__.startswith("cards") and cls is not cards.Card
]


def form_random_deck():
    deck = random.sample(ALL_CARDS, 12)
    return deck


def form_random_basic_deck():
    basic_decks = [[
        "Elektra",
        "Nightcrawler",
        "Squirrel Girl",
        "Mister Sinister",
        "Cable",
        "Ironheart",
        "Wolfsbane",
        "Jessica Jones",
        "White Tiger",
        "Blue Marvel",
        "Enchantress",
        "Odin"
    ], [
        "Antman",
        "Elektra",
        "Nightcrawler",
        "Squirrel Girl",
        "Blade",
        "Angela",
        "Wolverine",
        "Strong Guy",
        "Lady Sif",
        "Sword Master",
        "Kazar",
        "Blue Marvel"
    ], [
        "Antman",
        "Nightcrawler",
        "Angela",
        "Armor",
        "Colossus",
        "Mr Fantastic",
        "Cosmo",
        "Namor",
        "Ironman",
        "Klaw",
        "Spectrum",
        "Onslaught"
    ], [
        "Nova",
        "Agent 13",
        "Bucky Barnes",
        "Carnage",
        "Sabretooth",
        "Cable",
        "Sentinel",
        "Wolverine",
        "Killmonger",
        "Deathlok",
        "Moon Girl",
        "Devil Dinosaur"
    ]]

    chosen_deck = random.choice(basic_decks)
    deck = [
        cls
        for cls in ALL_CARDS
        if cls.__name__ in chosen_deck
    ]
    return deck
