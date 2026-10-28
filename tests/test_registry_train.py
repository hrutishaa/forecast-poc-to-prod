from forecast import registry
from forecast.train import train


def test_train_saves_versioned_artifact_and_reloads(tmp_path):
    r = train("energy_demand", root=tmp_path)
    assert r["version"].startswith("gbm_lags-")
    assert (tmp_path / "energy_demand" / "LATEST").read_text() == r["version"]
    model, meta = registry.load_latest("energy_demand", root=tmp_path)
    assert meta["version"] == r["version"] and "baseline" in meta["metrics"]
    assert len(model.predict(3)) == 3
