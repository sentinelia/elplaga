"""Generate the report figures from the CSVs written by experiments.py.

Usage: ``python make_plots.py`` (after running the experiment suites).
Writes PNG figures into ``results/``.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from q_learning_agent import QLearningAgent

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
MODELS = os.path.join(HERE, "models")

# Categorical palette (fixed slot order), light mode.
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"
SURFACE = "#fcfcfb"

SEQ_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "text.color": TEXT_PRIMARY,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def read_summary(suite):
    with open(os.path.join(RESULTS, f"{suite}_summary.csv")) as f:
        return list(csv.DictReader(f))


def read_history(run_id):
    with open(os.path.join(RESULTS, f"history_{run_id}.csv")) as f:
        rows = list(csv.DictReader(f))
    return {
        "reward": np.array([float(r["reward"]) for r in rows]),
        "success": np.array([int(r["success"]) for r in rows]),
    }


def smooth(x, w=100):
    if len(x) < w:
        w = max(1, len(x) // 10)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def mean_curve(run_ids, key="reward"):
    curves = [read_history(r)[key].astype(float) for r in run_ids]
    n = min(len(c) for c in curves)
    return np.mean([c[:n] for c in curves], axis=0)


def curves_figure(groups, title, path, ylabel="Return (moving avg over 100 episodes)"):
    """groups: list of (label, [run_ids])."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, (label, run_ids) in enumerate(groups):
        y = smooth(mean_curve(run_ids))
        ax.plot(np.arange(len(y)), y, color=SERIES[i % len(SERIES)], label=label)
        ax.annotate(
            label,
            xy=(len(y) - 1, y[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            color=SERIES[i % len(SERIES)],
            fontsize=9,
            va="center",
        )
    ax.set_xlabel("Training episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.margins(x=0.12)
    fig.savefig(os.path.join(RESULTS, path))
    plt.close(fig)


def grouped(rows, key_fn):
    """Group summary rows by key_fn -> {key: [rows]} preserving order."""
    out = {}
    for r in rows:
        out.setdefault(key_fn(r), []).append(r)
    return out


def bars_figure(ax, labels, values, errs, color, title, highlight=None):
    y = np.arange(len(labels))
    colors = [color if l != highlight else SERIES[2] for l in labels]
    ax.barh(y, values, xerr=errs, height=0.55, color=colors, error_kw={"ecolor": TEXT_SECONDARY, "lw": 1})
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_title(title)
    for yi, v in zip(y, values):
        ax.annotate(f"{v:.0f}", xy=(v, yi), xytext=(4 if v >= 0 else -4, 0),
                    textcoords="offset points", va="center",
                    ha="left" if v >= 0 else "right", fontsize=9, color=TEXT_PRIMARY)


def fig_discretization():
    # Grid-resolution sweep on the solvable 4-action set (the 5-action sweep
    # is uniformly stuck at 0 and cannot discriminate resolutions).
    rows4 = read_summary("discretization_a4")
    by_grid = grouped(rows4, lambda r: f"{r['position_bins']}x{r['velocity_bins']}")
    curves_figure(
        [(f"{k} grid", [r["run_id"] for r in v]) for k, v in by_grid.items()],
        "State-grid resolution (4 actions, mean of 2 seeds)",
        "fig1_discretization_grids.png",
    )
    # Action-count sweep at 20x20: odd counts (with a 0-throttle action) vs
    # even counts (without). The 4-action runs come from the a4 grid sweep.
    rows = read_summary("discretization")
    act_rows = [r for r in rows if f"{r['position_bins']}x{r['velocity_bins']}" == "20x20"]
    act_rows += [r for r in rows4 if f"{r['position_bins']}x{r['velocity_bins']}" == "20x20"]
    by_act = grouped(act_rows, lambda r: int(r["num_actions"]))
    curves_figure(
        [(f"{k} actions", [r["run_id"] for r in v]) for k, v in sorted(by_act.items())],
        "Action-set size (20x20 grid, mean of 2 seeds)",
        "fig2_discretization_actions.png",
    )


def fig_hyperparams():
    # The sweeps on the chosen (4-action) discretization; the 5-action sweep
    # is shown as a table in the notebook/report instead (nearly all zeros).
    rows = read_summary("hyperparams_a4")
    base = [r for r in rows if r["run_id"].startswith("hp4_base")]
    base_reward = np.mean([float(r["eval_mean_reward"]) for r in base])

    sweeps = {
        "alpha": ("Learning rate alpha", "0.1"),
        "gamma": ("Discount gamma", "0.99"),
        "epsilon_decay": ("Epsilon decay", "0.999"),
        "epsilon_min": ("Epsilon floor", "0.05"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, (param, (title, base_val)) in zip(axes.flat, sweeps.items()):
        swept = [r for r in rows if r["run_id"].startswith(f"hp4_{param}")]
        by_val = grouped(swept, lambda r: r[param])
        labels, values, errs = [f"{base_val}*"], [base_reward], [
            np.std([float(r["eval_mean_reward"]) for r in base])
        ]
        for val, v in sorted(by_val.items(), key=lambda kv: float(kv[0])):
            labels.append(val)
            values.append(np.mean([float(r["eval_mean_reward"]) for r in v]))
            errs.append(np.std([float(r["eval_mean_reward"]) for r in v]))
        bars_figure(ax, labels, values, errs, SERIES[0], title)
        ax.set_xlabel("Eval return (100 greedy episodes)")
    fig.suptitle("Hyperparameter sweeps around the base configuration (* = base value)", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig3_hyperparams.png"))
    plt.close(fig)


def fig_dynaq():
    rows = read_summary("dynaq")

    def label(r):
        if r["planning_steps"] in ("", "None"):
            return "Q-Learning"
        return f"Dyna-Q n={r['planning_steps']}"

    by = grouped(rows, label)
    curves_figure(
        [(k, [r["run_id"] for r in v]) for k, v in by.items()],
        "Dyna-Q planning budget vs. Q-Learning (600 episodes, mean of 2 seeds)",
        "fig4_dynaq_curves.png",
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    labels = list(by.keys())
    rewards = [np.mean([float(r["eval_mean_reward"]) for r in v]) for v in by.values()]
    rerrs = [np.std([float(r["eval_mean_reward"]) for r in v]) for v in by.values()]
    times = [np.mean([float(r["train_seconds"]) for r in v]) for v in by.values()]
    bars_figure(ax1, labels, rewards, rerrs, SERIES[0], "Eval return after 600 episodes")
    ax1.set_xlabel("Eval return (100 greedy episodes)")
    bars_figure(ax2, labels, times, [0] * len(times), SERIES[1], "Training wall-clock time")
    ax2.set_xlabel("Seconds")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig5_dynaq_tradeoff.png"))
    plt.close(fig)


def fig_final():
    rows = read_summary("final")
    fig, axes = plt.subplots(1, len(rows), figsize=(3.6 * len(rows), 3.8))
    for ax, r, color in zip(np.atleast_1d(axes), rows, SERIES):
        h = read_history(r["run_id"])
        y = smooth(h["reward"])
        ax.plot(np.arange(len(y)), y, color=color)
        ax.set_title(
            f"{r['run_id']}\neval {float(r['eval_mean_reward']):.1f} "
            f"({float(r['eval_success_rate']) * 100:.0f}% success)",
            fontsize=9,
        )
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Return (moving avg)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig6_final_learning_curves.png"))
    plt.close(fig)


def fig_policy():
    agent = QLearningAgent.load(os.path.join(MODELS, "final_qlearning.pkl"))
    d = agent.discretizer
    visited = agent.Q.any(axis=2)
    value = agent.Q.max(axis=2)
    policy = agent.Q.argmax(axis=2).astype(float)
    throttle = np.array([d.actions[int(i)] for i in policy.ravel()]).reshape(policy.shape)
    value_m = np.ma.masked_where(~visited, value)
    throttle_m = np.ma.masked_where(~visited, throttle)

    seq = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    seq.set_bad(color="#f0efec")
    div = LinearSegmentedColormap.from_list("div_blue_red", ["#2a78d6", "#f0efec", "#e34948"])
    div.set_bad(color="#ffffff")

    extent = [-1.2, 0.6, -0.07, 0.07]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    im1 = ax1.imshow(value_m.T, origin="lower", aspect="auto", extent=extent, cmap=seq)
    ax1.set_title("State value  max_a Q(s, a)")
    fig.colorbar(im1, ax=ax1, shrink=0.85)
    im2 = ax2.imshow(throttle_m.T, origin="lower", aspect="auto", extent=extent, cmap=div, vmin=-1, vmax=1)
    ax2.set_title("Greedy throttle (blue = push left, red = push right)")
    fig.colorbar(im2, ax=ax2, shrink=0.85)
    for ax in (ax1, ax2):
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.grid(False)
        ax.axvline(0.45, color=TEXT_SECONDARY, lw=1, ls="--")
        ax.annotate("goal", xy=(0.45, 0.06), fontsize=8, color=TEXT_SECONDARY, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig7_policy_value.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_discretization()
    fig_hyperparams()
    fig_dynaq()
    fig_final()
    fig_policy()
    print("figures written to results/")
