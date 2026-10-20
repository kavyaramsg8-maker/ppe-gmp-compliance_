from rule_engine.zones import point_in_polygon


def test_point_inside_square():
    square = [
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100],
    ]

    assert point_in_polygon((50, 50), square) is True


def test_point_outside_square():
    square = [
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100],
    ]

    assert point_in_polygon((150, 50), square) is False

