import random
import numpy
import Decks




def get_observation_array_snap_agent(game_state, number_of_cards):
    status_dictionary = game_state.status
    feature_list = build_basic_observation(game_state)
    feature_list.append(float(status_dictionary["cubes"]) / 8.0)
    feature_list.append(float(status_dictionary["tempcubes"]) / 8.0)
    feature_list.append(1.0 if status_dictionary["allysnapped"] else 0.0)

    features = numpy.array(feature_list, dtype=numpy.float32)
    owned_cards_mask = build_owned_cards_vector(game_state, True, number_of_cards).astype(numpy.float32)
    return numpy.concatenate([features, owned_cards_mask], axis=0)




def build_locations_mask(game_state, locations_mask_length):
    locations_mask = numpy.zeros(locations_mask_length, dtype=numpy.int8)
    location_dictionary = game_state.locationList
    for location_number, location_key in enumerate(["location1", "location2", "location3"]):
        location = location_dictionary[location_key]
        location_class = location.__class__
        location_index = Decks.LOCATION_CLASS_TO_INDEX.get(location_class)
        if location_index is not None:
            locations_mask[location_index * 3 + location_number] = 1

    return locations_mask


def snap_randomly(game_state, is_ally: bool, snap_probability):
    if random.random() < snap_probability:
        if is_ally:
            game_state.snap(True)
        else:
            game_state.snap(False)
