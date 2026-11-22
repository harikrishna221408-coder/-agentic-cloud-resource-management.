"""Metrics, baseline comparison and plots for the forecasting models."""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Display names. The two Random Forests differ ONLY in their input features.
VARIANT_NAMES = {"current_value": "Random Forest + current value",
                 "notebook": "Random Forest (notebook features)"}
NAIVE = "Naive Persistence"
PERSISTENCE = "Persistence (current value)"


def mae_rmse(actual, predicted):
    return (float(mean_absolute_error(actual, predicted)),
            float(np.sqrt(mean_squared_error(actual, predicted))))


BASELINE_NOTE = (
    "Naive Persistence = the original notebook's baseline: it repeats cpu_lag_1 / mem_lag_1, which is the observation "
    "BEFORE the current one, so it predicts two steps ahead from stale data. Persistence (current value) is the standard "
    "one-step baseline: it repeats the latest observation. 'Random Forest (notebook features)' uses the original 7 "
    "features, which exclude the current value; 'Random Forest + current value' adds the current CPU and memory.")


def compare(test_df, predictions):
    """MAE / RMSE of every model for CPU and memory.

    predictions: {variant: (cpu_pred, mem_pred)}, e.g. {"current_value": (...), "notebook": (...)}.
    Returns a tidy DataFrame (Resource, Model, MAE, RMSE): the Random Forests first, then the two baselines.
    """
    rows = []
    for res, actual, idx, naive, current in [
        ("CPU", test_df["cpu_target"], 0, test_df["cpu_lag_1"], test_df["cpu_util_percent"]),
        ("Memory", test_df["mem_target"], 1, test_df["mem_lag_1"], test_df["mem_util_percent"]),
    ]:
        models = [(VARIANT_NAMES[v], preds[idx]) for v, preds in predictions.items()]
        models += [(NAIVE, naive), (PERSISTENCE, current)]
        for model, pred in models:
            mae, rmse = mae_rmse(actual, pred)
            rows.append({"Resource": res, "Model": model, "MAE": mae, "RMSE": rmse})
    return pd.DataFrame(rows)


def per_machine(test_df, predictions):
    """MAE per machine for each model. Columns: <CPU|Memory>_RF_<variant>_MAE, _Naive_MAE, _Current_MAE."""
    d = test_df.reset_index(drop=True).copy()
    for v, (cpu, mem) in predictions.items():
        d[f"cpu_rf_{v}"], d[f"mem_rf_{v}"] = np.asarray(cpu), np.asarray(mem)
    rows = []
    for m, g in d.groupby("machine_id"):
        row = {"machine_id": m}
        for res, key, tgt, naive, cur in [("CPU", "cpu", "cpu_target", "cpu_lag_1", "cpu_util_percent"),
                                          ("Memory", "mem", "mem_target", "mem_lag_1", "mem_util_percent")]:
            for v in predictions:
                row[f"{res}_RF_{v}_MAE"] = mae_rmse(g[tgt], g[f"{key}_rf_{v}"])[0]
            row[f"{res}_Naive_MAE"] = mae_rmse(g[tgt], g[naive])[0]
            row[f"{res}_Current_MAE"] = mae_rmse(g[tgt], g[cur])[0]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_actual_vs_pred(test_df, predictions, machine_id, out_dir, n=300):
    d = test_df.reset_index(drop=True).copy()
    for v, (cpu, mem) in predictions.items():
        d[f"cpu_rf_{v}"], d[f"mem_rf_{v}"] = np.asarray(cpu), np.asarray(mem)
    d = d[d["machine_id"] == machine_id].head(n)
    paths = []
    for res, key, tgt, cur in [("CPU", "cpu", "cpu_target", "cpu_util_percent"),
                               ("Memory", "mem", "mem_target", "mem_util_percent")]:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(d["time_stamp"], d[tgt], label="Actual (next value)", lw=1.8, color="#222")
        for v in predictions:
            ax.plot(d["time_stamp"], d[f"{key}_rf_{v}"], label=VARIANT_NAMES[v], lw=1.1)
        ax.plot(d["time_stamp"], d[cur], label=PERSISTENCE, lw=1, ls="--", alpha=0.8)
        ax.set(xlabel="Timestamp (s)", ylabel=f"{res} utilisation (%)", title=f"{res} forecast vs actual - {machine_id}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = out_dir / f"{res.lower()}_forecast_{machine_id}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def verdict(comp):
    """Plain-language findings computed from the comparison table, so text can never contradict the numbers.

    Returns one sentence per resource: the model with the lowest MAE, and how the current-value Random Forest
    compares with the standard one-step baseline (repeat the current value) on MAE and RMSE.
    """
    rf_name = VARIANT_NAMES["current_value"]
    out = []
    for res in ("CPU", "Memory"):
        d = comp[comp["Resource"] == res].set_index("Model")
        best = d["MAE"].idxmin()
        rf, base = d.loc[rf_name], d.loc[PERSISTENCE]
        if base["MAE"] == 0 or base["RMSE"] == 0:       # a perfectly constant series: a % comparison is meaningless
            out.append(f"{res}: lowest MAE is '{best}' ({d.loc[best, 'MAE']:.3f}). {PERSISTENCE} makes no error on "
                       f"this data, so a percentage comparison is not possible.")
            continue
        d_mae = 100 * (rf["MAE"] - base["MAE"]) / base["MAE"]
        d_rmse = 100 * (rf["RMSE"] - base["RMSE"]) / base["RMSE"]
        out.append(f"{res}: lowest MAE is '{best}' ({d.loc[best, 'MAE']:.3f}). Against {PERSISTENCE}, {rf_name} has "
                   f"{'lower' if d_mae < 0 else 'higher'} MAE ({d_mae:+.1f}%) and "
                   f"{'lower' if d_rmse < 0 else 'higher'} RMSE ({d_rmse:+.1f}%).")
    return out
