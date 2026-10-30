from rule_engine.smoothing import ViolationManager


def test_k_out_of_n_triggers_once():
    manager = ViolationManager(k=3, n=5, cooldown_frames=10)

    results = []

    for condition in [False, True, True, False, True]:
        event = manager.process(
            track_id=17,
            rule_name="MISSING_GLOVES",
            condition=condition,
            violation_data={
                "track_id": 17,
                "type": "MISSING_PPE",
                "item": "gloves",
                "zone": "production",
            },
            frame=None,
        )
        results.append(event)

    assert results[-1] is not None

