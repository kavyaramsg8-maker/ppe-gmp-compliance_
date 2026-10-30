class PersonState:
    def __init__(self, track_id):
        self.track_id = track_id

        # Boolean PPE memory
        self.ppe = {}

        # Exact PPE state from ML adapter
        self.current_ppe_state = {}

        # Zone tracking
        self.current_zone = None
        self.previous_zone = None
        self.zone_history = []

        # Gowning order tracking
        self.gowning_sequence = []

        # Optional tracking metadata
        self.last_seen_frame = None
