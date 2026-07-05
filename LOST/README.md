# Project LOST — Learning-based Orientation and Steering for Traversal

Reinforcement-learning agent for the Gymnasium `MountainCarContinuous-v0`
environment: uniform-grid discretization of the continuous observation and
action spaces, tabular **Q-Learning**, a hyperparameter study, and **Dyna-Q**
(Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed.,
Sections 8.1–8.2) as the research component.

See `REPORT.md` for the full write-up (approach, experiments, results and
conclusions). All figures referenced by the report live in `results/`.

## Layout

| Path | Contents |
|---|---|
| `discretization.py` | Uniform-grid discretizer for states and actions |
| `q_learning_agent.py` | Tabular Q-Learning agent (train / test / save / load) |
| `dyna_q_agent.py` | Dyna-Q agent (Q-Learning + learned model + planning) |
| `experiments.py` | Experiment suites: `discretization`, `hyperparams`, `dynaq`, `final` |
| `make_plots.py` | Regenerates every figure in `results/` from the CSVs |
| `continuous_mountain_car.ipynb` | Notebook walking through the whole project |
| `models/*.pkl` | Trained Q-tables (computed models) |
| `results/*.csv` | Per-run summaries and per-episode training histories |
| `results/*.png` | Report figures |

## Setup

With Poetry (the course environment):

```bash
poetry install
poetry run jupyter notebook   # or: poetry run python experiments.py final
```

Or with plain pip: `pip install numpy gymnasium matplotlib`.

## Reproducing the experiments

```bash
python experiments.py discretization   # task 1: state/action grid study
python experiments.py hyperparams      # task 3: hyperparameter exploration
python experiments.py dynaq            # task 4: Dyna-Q vs Q-Learning
python experiments.py final            # final long runs; saves models/*.pkl
python make_plots.py                   # regenerate results/*.png
```

Runs are seeded; each suite writes a `results/<suite>_summary.csv` with the
evaluation of every configuration (100 greedy episodes on a fixed set of
start states).

## Using a trained model

```python
import gymnasium as gym
from q_learning_agent import QLearningAgent

agent = QLearningAgent.load("models/final_qlearning.pkl")
env = gym.make("MountainCarContinuous-v0", render_mode="human")
print(agent.test_agent(env, episodes=5))
```
