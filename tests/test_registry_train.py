from forecast import registry
from forecast.train import train


def test_train_saves_versioned_artifact_and_reloads(tmp_path):
    r = train("energy_demand", root=tmp_path)
    assert r["model_name"] == r["selection"]["winner"] and r["version"].startswith(r["model_name"] + "-")
    assert (tmp_path / "energy_demand" / "LATEST").read_text() == r["version"]
    model, meta = registry.load_latest("energy_demand", root=tmp_path)
    assert meta["version"] == r["version"] and "baseline" in meta["metrics"]
    assert meta["metrics"]["selection"]["leaderboard"][0]["model"] == r["model_name"]
    assert len(model.predict(3)) == 3
    assert meta["intervals"]["level"] == 0.8 and len(meta["intervals"]["rel_q"]) == 24


def test_train_with_fixed_model_skips_selection(tmp_path):
    r = train("retail_sales", root=tmp_path, model="ets_simple")
    assert r["version"].startswith("ets_simple-") and "selection" not in r
