import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.data.preprocessing import FEATURE_SETS, NOTEBOOK_FEATURES, build_features, chronological_split
from src.forecasting import evaluation, predict
from src.forecasting.evaluation import NAIVE, PERSISTENCE, VARIANT_NAMES


def synthetic(n=400, machines=("m_1", "m_2")):
    rng = np.random.default_rng(0)
    frames = []
    for m in machines:
        cpu = np.clip(40 + np.cumsum(rng.normal(0, 3, n)), 5, 95).round()
        mem = np.clip(80 + np.cumsum(rng.normal(0, 0.5, n)), 50, 99).round()
        frames.append(pd.DataFrame({"machine_id": m, "time_stamp": np.arange(n) * 10,
                                    "cpu_util_percent": cpu, "mem_util_percent": mem}))
    return build_features(pd.concat(frames, ignore_index=True))


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Both variants trained on synthetic data and saved the way train.py saves them (models/<variant>/)."""
    df = synthetic()
    train, test = chronological_split(df, 0.8)
    d = tmp_path_factory.mktemp("models")
    preds = {}
    for variant, feats in FEATURE_SETS.items():
        cpu = RandomForestRegressor(n_estimators=10, random_state=0).fit(train[feats], train["cpu_target"])
        mem = RandomForestRegressor(n_estimators=10, random_state=0).fit(train[feats], train["mem_target"])
        (d / variant).mkdir()
        joblib.dump(cpu, d / variant / "cpu_random_forest.joblib")
        joblib.dump(mem, d / variant / "memory_random_forest.joblib")
        preds[variant] = (cpu.predict(test[feats]), mem.predict(test[feats]))
    return test, preds, d


# ---------- comparison ----------
def test_compare_has_both_resources_and_all_models(trained):
    test, preds, _ = trained
    comp = evaluation.compare(test, preds)
    assert set(comp["Resource"]) == {"CPU", "Memory"}
    assert set(comp["Model"]) == {VARIANT_NAMES["current_value"], VARIANT_NAMES["notebook"], NAIVE, PERSISTENCE}
    assert len(comp) == 8
    assert (comp["MAE"] >= 0).all() and (comp["RMSE"] >= comp["MAE"] - 1e-9).all()


def test_baseline_is_last_value():
    t = pd.DataFrame({"cpu_target": [10.0, 20.0], "cpu_lag_1": [10.0, 20.0], "cpu_util_percent": [1.0, 2.0],
                      "mem_target": [50.0, 60.0], "mem_lag_1": [50.0, 60.0], "mem_util_percent": [1.0, 2.0]})
    comp = evaluation.compare(t, {"notebook": ([0, 0], [0, 0])})
    assert (comp[comp["Model"] == NAIVE]["MAE"] == 0).all()


def test_current_value_baseline_uses_current_observation_not_lag():
    t = pd.DataFrame({"cpu_target": [10.0, 20.0], "cpu_lag_1": [0.0, 0.0], "cpu_util_percent": [10.0, 20.0],
                      "mem_target": [50.0, 60.0], "mem_lag_1": [0.0, 0.0], "mem_util_percent": [50.0, 60.0]})
    comp = evaluation.compare(t, {"notebook": ([0, 0], [0, 0])}).set_index(["Resource", "Model"])
    assert comp.loc[("CPU", PERSISTENCE), "MAE"] == 0
    assert comp.loc[("CPU", NAIVE), "MAE"] == 15.0


def test_per_machine_has_one_row_per_machine_and_all_models(trained):
    test, preds, _ = trained
    pm = evaluation.per_machine(test, preds)
    assert len(pm) == test["machine_id"].nunique()
    for col in ("CPU_RF_current_value_MAE", "CPU_RF_notebook_MAE", "CPU_Naive_MAE", "CPU_Current_MAE",
                "Memory_RF_current_value_MAE", "Memory_RF_notebook_MAE", "Memory_Naive_MAE", "Memory_Current_MAE"):
        assert col in pm.columns and (pm[col] >= 0).all()


def test_verdict_is_computed_from_the_numbers_and_cannot_contradict_them():
    rows = []
    for res, rf_mae, base_mae in [("CPU", 2.0, 3.0), ("Memory", 0.9, 0.5)]:
        rows += [{"Resource": res, "Model": VARIANT_NAMES["current_value"], "MAE": rf_mae, "RMSE": rf_mae * 1.5},
                 {"Resource": res, "Model": VARIANT_NAMES["notebook"], "MAE": 9.0, "RMSE": 9.0},
                 {"Resource": res, "Model": NAIVE, "MAE": 9.0, "RMSE": 9.0},
                 {"Resource": res, "Model": PERSISTENCE, "MAE": base_mae, "RMSE": base_mae * 1.5}]
    cpu, mem = evaluation.verdict(pd.DataFrame(rows))
    assert "lowest MAE is 'Random Forest + current value'" in cpu and "lower MAE (-33.3%)" in cpu
    assert "lowest MAE is 'Persistence (current value)'" in mem and "higher MAE (+80.0%)" in mem


# ---------- prediction function ----------
@pytest.mark.parametrize("variant", sorted(FEATURE_SETS))
def test_predict_next_returns_valid_percentages(trained, variant):
    *_, d = trained
    cpu, mem = predict.predict_next([38, 40, 42, 45], [80, 80, 81, 81], 10, model_dir=d, variant=variant)
    assert 0 <= cpu <= 100 and 0 <= mem <= 100


def test_current_value_variant_uses_current_observation_and_notebook_variant_does_not(trained):
    *_, d = trained
    base = ([38, 40, 42, 45], [80, 80, 81, 81])
    changed = ([38, 40, 42, 90], [80, 80, 81, 99])          # only the CURRENT value differs
    a = predict.predict_next(*base, 10, model_dir=d, variant="notebook")
    b = predict.predict_next(*changed, 10, model_dir=d, variant="notebook")
    assert a == b                                              # notebook model ignores the current value
    c = predict.predict_next(*base, 10, model_dir=d, variant="current_value")
    e = predict.predict_next(*changed, 10, model_dir=d, variant="current_value")
    assert c != e                                              # the new model reacts to it


def test_predict_next_rejects_bad_input(trained):
    *_, d = trained
    with pytest.raises(ValueError):                            # 3 values: lags only, no current value
        predict.predict_next([40, 42, 45], [80, 81, 82], 10, model_dir=d, variant="notebook")
    with pytest.raises(ValueError):
        predict.predict_next([40, float("nan"), 42, 45], [80, 81, 82, 83], 10, model_dir=d, variant="current_value")
    with pytest.raises(ValueError):
        predict.predict_next([1, 2, 3, 4], [1, 2, 3, 4], 10, model_dir=d, variant="nonsense")


def test_active_variant_comes_from_config_and_is_validated():
    assert predict.active_variant({"forecast": {"active_variant": "notebook"}}) == "notebook"
    assert predict.active_variant({}) == "current_value"
    with pytest.raises(ValueError):
        predict.active_variant({"forecast": {"active_variant": "magic"}})


# ---------- feature design (documented, so it cannot change silently) ----------
def test_feature_sets_differ_only_by_the_current_values():
    assert FEATURE_SETS["notebook"] == NOTEBOOK_FEATURES and len(NOTEBOOK_FEATURES) == 7
    assert FEATURE_SETS["current_value"] == NOTEBOOK_FEATURES + ["cpu_util_percent", "mem_util_percent"]


def test_notebook_features_exclude_current_value_but_new_features_include_it():
    raw = pd.DataFrame({"machine_id": "m_1", "time_stamp": [i * 10 for i in range(8)],
                        "cpu_util_percent": [float(v) for v in (10, 20, 30, 40, 50, 60, 70, 80)],
                        "mem_util_percent": [float(v) for v in (1, 2, 3, 4, 5, 6, 7, 8)]})
    row = build_features(raw).iloc[0]
    assert row["cpu_util_percent"] == 40.0 and row["cpu_lag_1"] == 30.0 and row["cpu_target"] == 50.0
    assert 40.0 not in [row[f] for f in FEATURE_SETS["notebook"]]
    assert 40.0 in [row[f] for f in FEATURE_SETS["current_value"]] and 4.0 in [row[f] for f in FEATURE_SETS["current_value"]]


def test_verdict_handles_a_perfect_baseline_without_dividing_by_zero():
    rows = []
    for res in ("CPU", "Memory"):
        rows += [{"Resource": res, "Model": VARIANT_NAMES["current_value"], "MAE": 1.0, "RMSE": 1.0},
                 {"Resource": res, "Model": VARIANT_NAMES["notebook"], "MAE": 2.0, "RMSE": 2.0},
                 {"Resource": res, "Model": NAIVE, "MAE": 2.0, "RMSE": 2.0},
                 {"Resource": res, "Model": PERSISTENCE, "MAE": 0.0, "RMSE": 0.0}]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        text = evaluation.verdict(pd.DataFrame(rows))
    assert all("no error on this data" in t for t in text)
