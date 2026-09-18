from dataset_builder import rolling_snapshot_numbers


def test_rolling_snapshot_numbers_move_forward():
    assert rolling_snapshot_numbers(6, 3, 2, 1) == ([1, 2, 3], [4, 5], [6])
    assert rolling_snapshot_numbers(7, 3, 2, 1) == ([2, 3, 4], [5, 6], [7])
    assert rolling_snapshot_numbers(8, 3, 2, 1) == ([3, 4, 5], [6, 7], [8])
