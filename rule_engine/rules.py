def check_zone_authorisation(person, zones):
    """
    Return a list of zone-authorisation violations.
    Empty list means authorised or unrestricted.
    """
    current_zone = person.current_zone

    if current_zone is None:
        return []

    zone_data = zones.get(current_zone, {})
    authorised_ids = zone_data.get("authorised_track_ids")

    if not authorised_ids:
        return []

    if person.track_id in authorised_ids:
        return []

    return [{
        "track_id": person.track_id,
        "type": "UNAUTHORISED_ZONE",
        "zone": current_zone,
    }]


def update_gowning_sequence(person, item):
    """
    Record PPE items in the order they are first observed.
    """
    if item not in person.gowning_sequence:
        person.gowning_sequence.append(item)


def check_gowning_sequence(person, required_order):
    """
    Check whether observed gowning order follows the SOP.
    """
    observed = person.gowning_sequence

    for index, item in enumerate(observed):
        if index >= len(required_order):
            break

        if item != required_order[index]:
            return [{
                "track_id": person.track_id,
                "type": "GOWNING_SEQUENCE_VIOLATION",
                "item": item,
                "expected": required_order[index],
                "observed": observed.copy(),
            }]

    return []
