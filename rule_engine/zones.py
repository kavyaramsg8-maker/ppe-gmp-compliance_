import yaml

def load_zones(filepath="configs/zones.yaml"):
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)
    return data.get("zones", {})

def get_zone(camera_id, x, y, zones):
    """
    Determines the zone for a person based on camera assignment first, 
    and falls back to polygon coordinates if no camera is assigned.
    """
    # 1. CHECK IF THE CAMERA IS HARD-ASSIGNED TO A ZONE
    for zone_name, zone_data in zones.items():
        if zone_data.get("camera_id") == camera_id:
            return zone_name

    # 2. FALLBACK: If no camera assignment exists, use the polygon logic
    for zone_name, zone_data in zones.items():
        polygon = zone_data.get("polygon", [])
        if not polygon:
            continue
            
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]

        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        if min_x <= x <= max_x and min_y <= y <= max_y:
            return zone_name

    return None

def update_person_zone(person, camera_id, x, y, zones):
    """
    Updates the person's zone. Note we added 'camera_id' to the arguments.
    """
    new_zone = get_zone(camera_id, x, y, zones)

    # Only update if the zone has changed
    if new_zone != person.current_zone:
        person.current_zone = new_zone

        if new_zone is not None:
            person.zone_history.append(new_zone)

    return new_zone