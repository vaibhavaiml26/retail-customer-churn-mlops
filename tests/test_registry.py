import pandas as pd
from sklearn.linear_model import LogisticRegression
import config
import model_registry


def test_registry_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REGISTRY_ROOT", tmp_path / "registry")
    X = pd.DataFrame({"x": [0.0, 1.0, 0.2, 0.8]})
    y = pd.Series([0, 1, 0, 1])
    model = LogisticRegression().fit(X, y)
    version = model_registry.save_model(
        model,
        "test_model",
        metrics={"threshold": 0.5, "roc_auc": 1.0},
        params={},
        training_snapshots=[1],
        validation_snapshots=[2],
        test_snapshots=[3],
        feature_columns=["x"],
        latest_snapshot_used=3,
    )
    model_registry.set_current_version("test_model", version)
    loaded, meta = model_registry.load_champion("test_model")
    assert meta["version"] == version
    assert meta["latest_snapshot_used"] == 3
    assert loaded.predict(X).tolist() == model.predict(X).tolist()


def test_previous_pointer_tracks_replaced_champion(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REGISTRY_ROOT", tmp_path / "registry")
    X = pd.DataFrame({"x": [0.0, 1.0, 0.2, 0.8]})
    y = pd.Series([0, 1, 0, 1])

    first_model = LogisticRegression().fit(X, y)
    first = model_registry.save_model(
        first_model,
        "test_model",
        metrics={"threshold": 0.5, "roc_auc": 1.0},
        params={},
        training_snapshots=[1],
        validation_snapshots=[2],
        test_snapshots=[3],
        feature_columns=["x"],
        latest_snapshot_used=3,
    )
    model_registry.set_current_version("test_model", first)

    second_model = LogisticRegression(C=0.5).fit(X, y)
    second = model_registry.save_model(
        second_model,
        "test_model",
        metrics={"threshold": 0.4, "roc_auc": 1.0},
        params={"C": 0.5},
        training_snapshots=[2],
        validation_snapshots=[3],
        test_snapshots=[4],
        feature_columns=["x"],
        latest_snapshot_used=4,
    )
    model_registry.set_current_version("test_model", second)

    assert model_registry.get_current_version("test_model") == second
    assert model_registry.get_previous_version("test_model") == first
    _, previous_meta = model_registry.load_previous_champion("test_model")
    assert previous_meta["version"] == first
