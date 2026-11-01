"""Keras regression experiment for the Deep Learning homework.

Run:  python experiment.py
Outputs are written below ./results.  The script performs one-at-a-time
hyperparameter comparisons, each for 300 epochs, as requested in the brief.
"""
from __future__ import annotations

import csv
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score

SEED = 18  # School number ending 018; the brief requests the final two digits.
EPOCHS = 300
OUT = Path("results")
OUT.mkdir(exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)


@dataclass(frozen=True)
class Config:
    activation: str = "relu"
    learning_rate: float = 0.1
    optimizer: str = "sgd"
    initializer: str = "normal"
    hidden_layers: int = 1
    neurons: int = 2


BASELINE = Config()


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


def activation_layer(name: str):
    if name == "leaky_relu":
        return tf.keras.layers.LeakyReLU(negative_slope=0.3)
    if name == "elu":
        return tf.keras.layers.ELU(alpha=1.0)
    if name == "selu":
        return tf.keras.layers.Activation("selu")
    return tf.keras.layers.Activation(name)


def make_initializer(name: str):
    # The global seed is reset before every model.  Do not give every layer the
    # same local seed: that can make same-shaped layers start identically.
    if name == "normal":
        return tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
    if name == "uniform":
        return tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05)
    raise ValueError(f"Unknown initializer: {name}")


def make_optimizer(name: str, lr: float):
    optimizers = {
        "sgd": tf.keras.optimizers.SGD,
        "rmsprop": tf.keras.optimizers.RMSprop,
        "adam": tf.keras.optimizers.Adam,
    }
    return optimizers[name](learning_rate=lr)


def build_model(cfg: Config) -> tf.keras.Model:
    initializer = make_initializer(cfg.initializer)
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(1,))])
    for _ in range(cfg.hidden_layers):
        model.add(tf.keras.layers.Dense(cfg.neurons, kernel_initializer=initializer,
                                        bias_initializer="zeros"))
        model.add(activation_layer(cfg.activation))
    model.add(tf.keras.layers.Dense(1, activation="linear", kernel_initializer=initializer,
                                    bias_initializer="zeros"))
    model.compile(optimizer=make_optimizer(cfg.optimizer, cfg.learning_rate), loss="mse")
    return model


class R2History(tf.keras.callbacks.Callback):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        super().__init__()
        self.x, self.y, self.values = x, y, []

    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model.predict(self.x, verbose=0).reshape(-1)
        self.values.append(r2_score(self.y.reshape(-1), prediction))


def data():
    # The supplied target contains a variable typo. p is the declared input.
    p_train = np.linspace(-2.0, 2.0, 81, dtype=np.float32).reshape(-1, 1)
    p_eval = np.linspace(-2.0, 2.0, 401, dtype=np.float32).reshape(-1, 1)
    target = lambda p: 1.0 + np.sin(np.pi * p / 4.0)
    return p_train, target(p_train).astype(np.float32), p_eval, target(p_eval).astype(np.float32)


def run_experiment(group: str, option: str, cfg: Config) -> dict:
    set_seed()
    x_train, y_train, x_eval, y_eval = data()
    model = build_model(cfg)
    r2_callback = R2History(x_eval, y_eval)
    started = time.perf_counter()
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=len(x_train),
                        verbose=0, callbacks=[r2_callback])
    seconds = time.perf_counter() - started
    prediction = model.predict(x_eval, verbose=0).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_eval.reshape(-1), prediction)))
    r2 = float(r2_score(y_eval.reshape(-1), prediction))
    slug = f"{group}_{option}".replace(".", "_").replace(" ", "_")
    pd.DataFrame({"epoch": range(1, EPOCHS + 1), "mse": history.history["loss"],
                  "r2": r2_callback.values}).to_csv(OUT / f"history_{slug}.csv", index=False)
    record = {"group": group, "option": option, **asdict(cfg), "rmse": rmse,
              "r2": r2, "time_seconds": seconds, "history_file": f"history_{slug}.csv"}
    return record


def chart_best(record: dict) -> None:
    history = pd.read_csv(OUT / record["history_file"])
    _, _, x_eval, y_eval = data()
    cfg = Config(**{k: record[k] for k in asdict(BASELINE)})
    # Retrain only to draw the selected configuration's final prediction.
    set_seed(); model = build_model(cfg)
    x_train, y_train, _, _ = data()
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=len(x_train), verbose=0)
    pred = model.predict(x_eval, verbose=0).reshape(-1)
    label = f"{record['group']} - {record['option']}"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(history["epoch"], history["mse"], color="#1f77b4")
    axes[0].set_yscale("log"); axes[0].set(title="Epoch - MSE (log)", xlabel="Epoch", ylabel="MSE")
    axes[1].plot(history["epoch"], history["r2"], color="#2ca02c")
    axes[1].set(title="Epoch - R2", xlabel="Epoch", ylabel="R2")
    axes[2].plot(x_eval.reshape(-1), y_eval.reshape(-1), label="Target", linewidth=2)
    axes[2].plot(x_eval.reshape(-1), pred, label="Output", linestyle="--")
    axes[2].set(title="Target - Output", xlabel="p", ylabel="y"); axes[2].legend()
    fig.suptitle(label); fig.tight_layout()
    fig.savefig(OUT / "figures" / f"{record['group']}_best.png", dpi=180)
    plt.close(fig)


def cases():
    # Every group changes exactly one parameter; all others use BASELINE values.
    yield "activation", [(x, Config(**{**asdict(BASELINE), "activation": x}))
                         for x in ["linear", "sigmoid", "tanh", "relu", "leaky_relu", "elu", "selu"]]
    yield "learning_rate", [(str(x), Config(**{**asdict(BASELINE), "learning_rate": x}))
                            for x in [0.1, 0.01, 0.001]]
    yield "optimizer", [(x, Config(**{**asdict(BASELINE), "optimizer": x}))
                         for x in ["sgd", "rmsprop", "adam"]]
    yield "initializer", [(x, Config(**{**asdict(BASELINE), "initializer": x}))
                           for x in ["normal", "uniform"]]
    yield "hidden_layers", [(str(x), Config(**{**asdict(BASELINE), "hidden_layers": x}))
                             for x in [1, 2, 5, 10, 20]]
    yield "neurons", [(str(x), Config(**{**asdict(BASELINE), "neurons": x}))
                      for x in [2, 5, 10, 20, 50]]

def main() -> None:
    records = []
    for group, options in cases():
        print(f"[{group}] Grubu çalıştırılıyor...")
        group_records = []
        for option, cfg in options:
            print(f"  -> Seçenek: {option} eğitiliyor...")
            rec = run_experiment(group, option, cfg)
            group_records.append(rec)
        records.extend(group_records)
        best = min(group_records, key=lambda r: r["rmse"])
        chart_best(best)
    pd.DataFrame(records).to_csv(OUT / "summary.csv", index=False)
    print("Tamamlandı. Sonuçlar: results/summary.csv ve results/figures/")

if __name__ == "__main__":
    main()
