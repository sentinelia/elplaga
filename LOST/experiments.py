"""Experiment runner for the LOST project (MountainCarContinuous-v0).

Suites
------
- ``discretization``: Q-Learning with fixed hyperparameters over several
  state-grid and action-set resolutions (assignment task 1).
- ``hyperparams``: Q-Learning hyperparameter exploration around a base
  configuration on the chosen discretization (assignment task 3).
- ``dynaq``: Dyna-Q with several planning-step budgets vs. plain Q-Learning
  under a reduced episode budget (assignment task 4).
- ``final``: long training run of the best configuration; saves the final
  models shipped with the submission.

Usage: ``python experiments.py <suite>``

Each run writes:
- ``results/<suite>_summary.csv``: one row per run (config, eval metrics, time).
- ``results/history_<run_id>.csv``: per-episode training curve.
- ``models/<run_id>.pkl``: the trained Q-table (final suite and best runs).
"""

import csv
import os
import sys

import gymnasium as gym
import numpy as np

from discretization import Discretizer
from dyna_q_agent import DynaQAgent
from q_learning_agent import QLearningAgent

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

ENV_ID = "MountainCarContinuous-v0"
EVAL_EPISODES = 100
EVAL_SEED = 12345

# Base configuration used as the anchor of every sweep.
BASE = {
    "position_bins": 20,
    "velocity_bins": 20,
    "num_actions": 5,
    "episodes": 3000,
    "epsilon": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.999,
    "gamma": 0.99,
    "alpha": 0.1,
    "planning_steps": None,  # None -> plain Q-Learning
    "seed": 0,
}


def run_one(run_id, config, save_model=False):
    """Train and evaluate a single configuration; returns a summary row."""
    cfg = {**BASE, **config}
    disc = Discretizer(cfg["position_bins"], cfg["velocity_bins"], cfg["num_actions"])
    if cfg["planning_steps"] is None:
        agent = QLearningAgent(disc, seed=cfg["seed"])
    else:
        agent = DynaQAgent(disc, planning_steps=cfg["planning_steps"], seed=cfg["seed"])

    env = gym.make(ENV_ID)
    history = agent.train_agent(
        env,
        episodes=cfg["episodes"],
        epsilon=cfg["epsilon"],
        epsilon_min=cfg["epsilon_min"],
        epsilon_decay=cfg["epsilon_decay"],
        gamma=cfg["gamma"],
        alpha=cfg["alpha"],
    )

    # Deterministic evaluation: greedy policy, fixed set of start states.
    eval_env = gym.make(ENV_ID)
    eval_env.reset(seed=EVAL_SEED)
    results = agent.test_agent(eval_env, episodes=EVAL_EPISODES)
    env.close()
    eval_env.close()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"history_{run_id}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "steps", "success", "epsilon"])
        for i in range(len(history["reward"])):
            writer.writerow(
                [
                    i,
                    f"{history['reward'][i]:.4f}",
                    history["steps"][i],
                    int(history["success"][i]),
                    f"{history['epsilon'][i]:.5f}",
                ]
            )

    if save_model:
        os.makedirs(MODELS_DIR, exist_ok=True)
        agent.save(os.path.join(MODELS_DIR, f"{run_id}.pkl"))

    row = {
        "run_id": run_id,
        **{k: cfg[k] for k in BASE},
        "eval_mean_reward": float(np.mean(results["reward"])),
        "eval_std_reward": float(np.std(results["reward"])),
        "eval_success_rate": float(np.mean(results["success"])),
        "eval_mean_steps": float(np.mean(results["steps"])),
        "train_success_rate_last500": float(np.mean(history["success"][-500:])),
        "train_seconds": history["train_seconds"],
    }
    print(
        f"[{run_id}] eval_reward={row['eval_mean_reward']:.1f} "
        f"success={row['eval_success_rate']:.2f} "
        f"train_time={row['train_seconds']:.0f}s",
        flush=True,
    )
    return row


def write_summary(suite, rows):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{suite}_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


SEEDS = [0, 1]


def suite_discretization():
    rows = []
    # State-grid sweep at 5 actions, then action sweep on the 20x20 grid.
    grids = [(10, 10), (20, 20), (50, 50), (100, 100)]
    for px, vx in grids:
        for seed in SEEDS:
            rid = f"disc_grid{px}x{vx}_a5_s{seed}"
            rows.append(run_one(rid, {"position_bins": px, "velocity_bins": vx, "seed": seed}))
    for num_actions in [2, 3, 11]:
        for seed in SEEDS:
            rid = f"disc_grid20x20_a{num_actions}_s{seed}"
            rows.append(run_one(rid, {"num_actions": num_actions, "seed": seed}))
    write_summary("discretization", rows)


def suite_hyperparams():
    rows = []
    sweeps = {
        "alpha": [0.05, 0.2, 0.5],
        "gamma": [0.9, 0.999, 1.0],
        "epsilon_decay": [0.997, 0.9995],
        "epsilon_min": [0.0, 0.2],
    }
    # Base config itself (reused as the anchor point of every sweep).
    for seed in SEEDS:
        rows.append(run_one(f"hp_base_s{seed}", {"seed": seed}))
    for param, values in sweeps.items():
        for value in values:
            for seed in SEEDS:
                rid = f"hp_{param}{value}_s{seed}"
                rows.append(run_one(rid, {param: value, "seed": seed}))
    write_summary("hyperparams", rows)


def suite_hyperparams_a4():
    """Same coordinate-wise sweeps, on the chosen 4-action discretization.

    On the 5-action grid every configuration collapses into the do-nothing
    local optimum (see hyperparams_summary.csv), so that sweep cannot
    discriminate between hyperparameters. Repeating it on the 20x20 / 4-action
    discretization, where the task is solvable, makes the differences visible.
    """
    rows = []
    sweeps = {
        "alpha": [0.05, 0.2, 0.5],
        "gamma": [0.9, 0.999, 1.0],
        "epsilon_decay": [0.997, 0.9995],
        "epsilon_min": [0.0, 0.2],
    }
    for seed in SEEDS:
        rows.append(run_one(f"hp4_base_s{seed}", {"num_actions": 4, "seed": seed}))
    for param, values in sweeps.items():
        for value in values:
            for seed in SEEDS:
                rid = f"hp4_{param}{value}_s{seed}"
                rows.append(run_one(rid, {"num_actions": 4, param: value, "seed": seed}))
    write_summary("hyperparams_a4", rows)


def suite_dynaq():
    rows = []
    # Reduced episode budget: the point is sample efficiency.
    episodes = 600
    for planning_steps in [None, 0, 5, 20, 50]:
        label = "qlearning" if planning_steps is None else f"n{planning_steps}"
        for seed in SEEDS:
            rid = f"dynaq_{label}_ep{episodes}_s{seed}"
            rows.append(
                run_one(
                    rid,
                    {"planning_steps": planning_steps, "episodes": episodes, "seed": seed},
                    save_model=(planning_steps == 20 and seed == 0),
                )
            )
    write_summary("dynaq", rows)


def suite_final():
    rows = []
    rows.append(
        run_one(
            "final_qlearning",
            {"episodes": 5000, "seed": 0},
            save_model=True,
        )
    )
    rows.append(
        run_one(
            "final_dynaq_n20",
            {"planning_steps": 20, "episodes": 1500, "seed": 0},
            save_model=True,
        )
    )
    write_summary("final", rows)


SUITES = {
    "discretization": suite_discretization,
    "hyperparams": suite_hyperparams,
    "hyperparams_a4": suite_hyperparams_a4,
    "dynaq": suite_dynaq,
    "final": suite_final,
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in SUITES:
        print(f"usage: python experiments.py [{'|'.join(SUITES)}]")
        sys.exit(1)
    SUITES[sys.argv[1]]()
