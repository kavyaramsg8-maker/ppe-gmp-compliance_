def point_in_polygon(point, polygon):
    """
    Return True if point (x, y) lies inside polygon.

    Uses the ray-casting algorithm.
    """
    x, y = point
    inside = False

    j = len(polygon) - 1

    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        intersects = (
            (yi > y) != (yj > y)
            and
            x < (xj - xi) * (y - yi) / (yj - yi) + xi
        )

        if intersects:
            inside = not inside

        j = i

    return inside
import yaml


def load_zones(path="configs/zones.yaml"):
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    # Allow either:
    # zones:
    #   production: ...
    #
    # or directly:
    # production: ...
    return data.get("zones", data)


def update_person_zone(person, x, y, zones):
    new_zone = None

    for zone_name, zone_data in zones.items():
        polygon = zone_data.get("polygon", [])

        if polygon and point_in_polygon((x, y), polygon):
            new_zone = zone_name
            break

    if new_zone != person.current_zone:
        person.previous_zone = person.current_zone
        person.current_zone = new_zone

        if new_zone is not None:
            person.zone_history.append(new_zone)

    return new_zone
