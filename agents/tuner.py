import json
from pathlib import Path

import optuna
import numpy as np

import forecasting

TUNE_PATH = Path('tuning.json')
MEM_PATH = Path('memory.json')


def load_mem():
    if MEM_PATH.exists():
        try:
            with MEM_PATH.open() as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_mem(mem):
    with MEM_PATH.open('w') as f:
        json.dump(mem, f)


def objective(trial, series: np.ndarray):
    lam = trial.suggest_float('lam', 0.5, 0.99)
    p = trial.suggest_int('p', 1, 4)
    q = trial.suggest_int('q', 1, 3)
    garch = trial.suggest_categorical('garch', [True, False])
    return forecasting.walk_forward_mae(series, lam=lam, p_ar=p, q_res=q, garch=garch)


def main(argv=None):
    mem = load_mem()
    prices = np.array(mem.get('prices', []), dtype=float)
    if prices.size <= 30:
        return 1
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda t: objective(t, prices), n_trials=50, timeout=300)
    best_params = study.best_params
    best_mae = study.best_value
    with TUNE_PATH.open('w') as f:
        json.dump({'params': best_params, 'mae': best_mae}, f)
    prev_mae = mem.get('mae', float('inf'))
    if best_mae <= prev_mae * 0.98:
        mem['tuned_params'] = best_params
        mem['mae'] = best_mae
        save_mem(mem)
        return 0
    save_mem(mem)
    return 1


def cli(argv=None):
    raise SystemExit(main(argv))


if __name__ == '__main__':
    cli()
