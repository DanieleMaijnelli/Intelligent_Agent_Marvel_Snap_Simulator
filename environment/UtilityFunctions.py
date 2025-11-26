import numpy
import Decks


def get_observation_array(game_state, action_mask_length):
    status_dictionary = game_state.status
    location_dictionary = game_state.locationList
    feature_list = []
    for location_key in ["location1", "location2", "location3"]:
        location = location_dictionary[location_key]
        feature_list.append(float(location.alliesPower) / 10.0)
        feature_list.append(float(location.enemiesPower) / 10.0)
    feature_list.append(float(status_dictionary["allyenergy"]) / 10.0)
    feature_list.append(float(status_dictionary["turncounter"]) / 7.0)
    feature_list.append(1.0 if status_dictionary["allypriority"] else 0.0)

    features = numpy.array(feature_list, dtype=numpy.float32)
    action_mask = build_player_action_mask(game_state, True, action_mask_length).astype(numpy.float32)
    return numpy.concatenate([features, action_mask], axis=0)


def build_player_action_mask(game_state, is_ally, action_mask_length):
    action_mask = numpy.zeros(action_mask_length, dtype=numpy.int8)
    hand = game_state.status["allyhand"] if is_ally else game_state.status["enemyhand"]
    location_dictionary = game_state.locationList

    for card in hand:
        card_index = Decks.CLASS_TO_INDEX[card.__class__]
        for location_index, location_key in enumerate(["location1", "location2", "location3"]):
            location = location_dictionary[location_key]
            if can_play(game_state, is_ally, card, location):
                action_index = (card_index * 3) + location_index
                action_mask[action_index] = 1
    action_mask[action_mask_length - 1] = 1
    return action_mask


def decode_action_index(action_index, card_pool_list):
    card_index = action_index // 3
    location_index = action_index % 3
    card = card_pool_list[card_index]
    location_number = location_index + 1
    return card, location_number


def can_play(game_state, ally: bool, card, location):
    if ally:
        energy = game_state.status["allyenergy"]
        energy_check = card.cur_cost <= energy
        location_check = (not location.checkIfLocationFull(ally)) and location.can_play_cards_allies
        unit_check = location.canCardBePlayed(card)
        return energy_check and location_check and unit_check
    else:
        energy = game_state.status["enemyenergy"]
        energy_check = card.cur_cost <= energy
        location_check = (not location.checkIfLocationFull(ally)) and location.can_play_cards_enemies
        unit_check = location.canCardBePlayed(card)
        return energy_check and location_check and unit_check


def play_randomly(game_state, is_ally: bool, action_space_length, card_pool_list):
    pass_key = "allypass" if is_ally else "enemypass"
    hand_key = "allyhand" if is_ally else "enemyhand"

    while not game_state.status[pass_key]:
        action_mask = build_player_action_mask(game_state, is_ally, action_space_length)
        valid_actions = numpy.flatnonzero(action_mask)
        random_valid_action = numpy.random.choice(valid_actions)

        if random_valid_action == (action_space_length - 1):
            game_state.status[pass_key] = True
            break

        card_type, location_number = decode_action_index(random_valid_action, card_pool_list)

        for card_index, card in enumerate(game_state.status[hand_key]):
            if isinstance(card, card_type):
                if not game_state.addUnit(card_index, False, location_number):
                    continue
                break
