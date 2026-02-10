import numpy
import Decks
from sentence_transformers import SentenceTransformer

mpnet_model = SentenceTransformer("all-mpnet-base-v2")


def embed_text(text):
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)
    embedding_vector = mpnet_model.encode(text, normalize_embeddings=True)
    return numpy.array(embedding_vector, dtype=numpy.float32)


CARD_EMBEDDING_CACHE = {}
LOCATION_EMBEDDING_CACHE = {}
EMPTY_EMBEDDING = list(embed_text(""))
EMBEDDING_LENGTH = len(EMPTY_EMBEDDING)
EMPTY_HAND_SLOT = [0.0, 0.0, 0.0] + [0.0] * EMBEDDING_LENGTH
EMPTY_BOARD_SLOT = [0.0, 0.0] + [0.0] * EMBEDDING_LENGTH


def get_card_embedding(card_object):
    card_class = card_object.__class__
    if card_class in CARD_EMBEDDING_CACHE:
        return CARD_EMBEDDING_CACHE[card_class]
    card_text = card_object.description
    embedding_vector = embed_text(card_text)
    embedding_list = list(embedding_vector)
    CARD_EMBEDDING_CACHE[card_class] = embedding_list
    return embedding_list


def get_location_embedding(location_object):
    cache_key = location_object.name
    if cache_key in LOCATION_EMBEDDING_CACHE:
        return LOCATION_EMBEDDING_CACHE[cache_key]
    if location_object.description != "":
        location_text = location_object.description
    else:
        location_text = location_object.name
    embedding_vector = embed_text(location_text)
    embedding_list = list(embedding_vector)
    LOCATION_EMBEDDING_CACHE[cache_key] = embedding_list
    return embedding_list


def build_basic_observation(game_state, is_ally):
    status_dictionary = game_state.status
    feature_list = [float(status_dictionary["turncounter"]) / 7.0]
    if is_ally:
        feature_list.append(float(status_dictionary["allyenergy"]) / 10.0)
        feature_list.append(1.0 if status_dictionary["allypriority"] else 0.0)
    else:
        feature_list.append(float(status_dictionary["enemyenergy"]) / 10.0)
        feature_list.append(1.0 if not status_dictionary["allypriority"] else 0.0)
    return feature_list


def build_hand_observation(game_state, is_ally, action_flag):
    hand = game_state.status["allyhand"] if is_ally else game_state.status["enemyhand"]
    maximum_hand_size = 7
    hand_feature_list = []
    hand_index = 0
    while hand_index < maximum_hand_size:
        if hand_index < len(hand):
            card_object = hand[hand_index]
            if action_flag == hand_index:
                played_card_flag = 1
            else:
                played_card_flag = 0
            current_power_value = float(card_object.cur_power)
            current_cost_value = float(card_object.cur_cost)
            embedding_vector = get_card_embedding(card_object)
            hand_card_slot = [played_card_flag, current_power_value, current_cost_value]
            hand_card_slot.extend(embedding_vector)
        else:
            hand_card_slot = EMPTY_HAND_SLOT
        hand_feature_list.extend(hand_card_slot)
        hand_index += 1
    return hand_feature_list


def build_locations_observation(game_state, is_ally, action_flag):
    locations_feature_list = []
    location_dictionary = game_state.locationList
    location_index = 0
    for location_key in ["location1", "location2", "location3"]:
        location = location_dictionary[location_key]
        if location_index == action_flag:
            chosen_location_flag = 1
        else:
            chosen_location_flag = 0
        locations_feature_list.append(chosen_location_flag)
        if is_ally:
            locations_feature_list.append(float(location.alliesPower) / 10.0)
            locations_feature_list.append(float(location.enemiesPower) / 10.0)
        else:
            locations_feature_list.append(float(location.enemiesPower) / 10.0)
            locations_feature_list.append(float(location.alliesPower) / 10.0)
        embedding_vector = get_location_embedding(location)
        for value in embedding_vector:
            locations_feature_list.append(float(value))
        location_index += 1
    return locations_feature_list


def build_played_cards_observation(game_state, is_ally):
    played_cards_feature_list = []
    location_dictionary = game_state.locationList
    for location_key in ["location1", "location2", "location3"]:
        location = location_dictionary[location_key]
        if is_ally:
            ally_list = location.allies
            enemy_list = location.enemies
        else:
            ally_list = location.enemies
            enemy_list = location.allies

        for card_list in [ally_list, enemy_list]:
            slot_index = 0
            while slot_index < 4:
                if slot_index < len(card_list):
                    card_object = card_list[slot_index]
                    current_power_value = float(card_object.cur_power)
                    current_cost_value = float(card_object.cur_cost)
                    embedding_vector = get_card_embedding(card_object)
                    card_slot = [current_power_value, current_cost_value]
                    card_slot.extend(embedding_vector)
                else:
                    card_slot = list(EMPTY_BOARD_SLOT)
                played_cards_feature_list.extend(card_slot)
                slot_index += 1

    return played_cards_feature_list


def build_observation_with_chosen_action(game_state, is_ally, action_tuple):
    feature_list = []
    basic_observation = build_basic_observation(game_state, is_ally)
    locations_observation = build_locations_observation(game_state, is_ally, action_tuple[1])
    hand_observation = build_hand_observation(game_state, is_ally, action_tuple[0])
    played_cards_observation = build_played_cards_observation(game_state, is_ally)
    feature_list.extend(basic_observation)
    feature_list.extend(locations_observation)
    feature_list.extend(hand_observation)
    feature_list.extend(played_cards_observation)
    return numpy.array(feature_list, dtype=numpy.float32)


def build_basic_observation_snap_agent(game_state, is_ally):
    status_dictionary = game_state.status
    feature_list = [float(status_dictionary["turncounter"]) / 7.0, float(status_dictionary["cubes"]) / 8.0,
                    float(status_dictionary["tempcubes"]) / 8.0]

    if is_ally:
        feature_list.append(float(status_dictionary["allyenergy"]) / 10.0)
        feature_list.append(1.0 if status_dictionary["allypriority"] else 0.0)
        feature_list.append(1.0 if status_dictionary["allysnapped"] else 0.0)
        feature_list.append(1.0 if status_dictionary["enemysnapped"] else 0.0)
    else:
        feature_list.append(float(status_dictionary["enemyenergy"]) / 10.0)
        feature_list.append(1.0 if not status_dictionary["allypriority"] else 0.0)
        feature_list.append(1.0 if status_dictionary["enemysnapped"] else 0.0)
        feature_list.append(1.0 if status_dictionary["allysnapped"] else 0.0)

    return feature_list


def build_observation_snap_agent_with_chosen_action(game_state, is_ally, action):
    feature_list = []
    one_hot = [0.0, 0.0, 0.0]
    one_hot[action.value] = 1.0
    basic_observation = build_basic_observation_snap_agent(game_state, is_ally)
    locations_observation = build_locations_observation(game_state, is_ally, -1)
    hand_observation = build_hand_observation(game_state, is_ally, -1)
    played_cards_observation = build_played_cards_observation(game_state, is_ally)
    feature_list.extend(basic_observation)
    feature_list.extend(locations_observation)
    feature_list.extend(hand_observation)
    feature_list.extend(played_cards_observation)
    feature_list.extend(one_hot)
    return numpy.array(feature_list, dtype=numpy.float32)


def can_play(game_state, ally: bool, card, location):
    if ally:
        energy = game_state.status["allyenergy"]
        energy_check = card.cur_cost <= energy
        location_check = (not location.checkIfLocationFull(True)) and location.can_play_cards_allies
        unit_check = location.canCardBePlayed(card)
        return energy_check and location_check and unit_check
    else:
        energy = game_state.status["enemyenergy"]
        energy_check = card.cur_cost <= energy
        location_check = (not location.checkIfLocationFull(False)) and location.can_play_cards_enemies
        unit_check = location.canCardBePlayed(card)
        return energy_check and location_check and unit_check


def get_legal_actions(game_state, is_ally):
    status_dictionary = game_state.status
    location_dictionary = game_state.locationList
    if is_ally:
        hand = status_dictionary["allyhand"]
    else:
        hand = status_dictionary["enemyhand"]

    legal_actions_list = [(-1, -1)]

    hand_index = 0
    while hand_index < len(hand):
        card = hand[hand_index]
        location_index = 0
        while location_index < 3:
            location_key = "location" + str(location_index + 1)
            location = location_dictionary[location_key]
            if can_play(game_state, is_ally, card, location):
                legal_actions_list.append((hand_index, location_index))
            location_index += 1
        hand_index += 1

    return legal_actions_list
