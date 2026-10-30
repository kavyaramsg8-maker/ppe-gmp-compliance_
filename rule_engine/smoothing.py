from collections import defaultdict, deque


class ViolationManager:
    def __init__(self, k=3, n=5, cooldown_frames=30):
        self.k = k
        self.n = n
        self.cooldown_frames = cooldown_frames

        self.history = defaultdict(lambda: deque(maxlen=n))
        self.cooldowns = {}
        self.frame_counter = 0

    def process(
        self,
        track_id,
        rule_name,
        condition,
        violation_data,
        frame=None,
    ):
        self.frame_counter += 1

        key = (track_id, rule_name)

        self.history[key].append(bool(condition))

        true_count = sum(self.history[key])

        if true_count < self.k:
            return None

        last_alert_frame = self.cooldowns.get(key)

        if (
            last_alert_frame is not None
            and self.frame_counter - last_alert_frame < self.cooldown_frames
        ):
            return None

        self.cooldowns[key] = self.frame_counter

        return violation_data.copy()

    def clear_cooldown(self, track_id, rule_name):
        key = (track_id, rule_name)
        self.cooldowns.pop(key, None)

    def update(self):
        pass
