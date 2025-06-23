
import optuna
import numpy as np

import forecasting
from state import load_state, save_state


def objective(trial, series: np.ndarray):
    lam = trial.suggest_float("lam", 0.5, 0.99)
    p = trial.suggest_int("p", 1, 4)
    q = trial.suggest_int("q", 1, 3)
    garch = trial.suggest_categorical("garch", [True, False])
    return forecasting.walk_forward_mae(series, lam=lam, p_ar=p, q_res=q, garch=garch)


def main(argv=None):
    mem = load_state()
    prices = np.array(mem.get("prices", []), dtype=float)
    if prices.size <= 30:
        return 1
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, prices), n_trials=50, timeout=300)
    best_params = study.best_params
    best_mae = study.best_value
    mem["tuned_params"] = best_params
    mem["mae"] = best_mae
    save_state(mem)
    return 0


def cli(argv=None):
    raise SystemExit(main(argv))


if __name__ == "__main__":
    cli()
