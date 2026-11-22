import pandas as pd
from src.data.preprocessing import clean, build_features, chronological_split, FEATURES


def make(n=30, machine="m_1", step=10, start=0):
    return pd.DataFrame({
        "machine_id": machine,
        "time_stamp": [start + i * step for i in range(n)],
        "cpu_util_percent": [float(i % 50) for i in range(n)],
        "mem_util_percent": [float(50 + i % 10) for i in range(n)],
    })


def test_clean_removes_duplicates_and_invalid():
    df = pd.concat([make(5), make(5).iloc[[0]]], ignore_index=True)
    df.loc[1, "cpu_util_percent"] = 150.0
    out = clean(df)
    assert not out.duplicated(["machine_id", "time_stamp"]).any()
    assert out["cpu_util_percent"].max() <= 100
    assert len(out) == 4


def test_features_are_correct_lags_and_targets():
    out = build_features(make(10))
    first = out.iloc[0]  # raw row index 3
    assert first["cpu_lag_1"] == 2.0 and first["cpu_lag_3"] == 0.0
    assert first["cpu_target"] == 4.0
    assert set(FEATURES) <= set(out.columns)
    assert (out["input_gap"] == 10).all()


def test_features_do_not_cross_machines():
    df = pd.concat([make(6, "m_1"), make(6, "m_2")], ignore_index=True)
    out = build_features(df)
    assert len(out) == 2 * 2  # each machine loses 3 head rows and 1 tail row


def test_large_gaps_filtered():
    df = make(10)
    df.loc[5:, "time_stamp"] += 500
    out = build_features(df, max_gap=60)
    assert (out["input_gap"] <= 60).all() and (out["target_gap"] <= 60).all()


def test_chronological_split_has_no_leakage():
    out = build_features(make(100))
    train, test = chronological_split(out, 0.8)
    assert train["time_stamp"].max() < test["time_stamp"].min()
    assert abs(len(train) / len(out) - 0.8) < 0.02
