# Project LOST — Learning-based Orientation and Steering for Traversal

**Environment:** Gymnasium `MountainCarContinuous-v0` · **Techniques:** tabular Q-Learning, Dyna-Q

The rover "Out for Delivery" must learn from reward alone that moving forward
(and sometimes backward first) beats standing still. This report covers the
four assignment tasks: discretization of the continuous spaces, Q-Learning,
hyperparameter exploration, and the research component — Dyna-Q from Sutton &
Barto, *Reinforcement Learning: An Introduction* (2nd ed.), §8.1–8.2.

All code is in this directory (`discretization.py`, `q_learning_agent.py`,
`dyna_q_agent.py`, `experiments.py`, `make_plots.py`), the narrated version is
`continuous_mountain_car.ipynb`, the computed models are in `models/`, and
every number below can be traced to a row of `results/*_summary.csv`.

## 1. The problem, and why it is deceptive

The observation is continuous — position $x \in [-1.2, 0.6]$, velocity
$v \in [-0.07, 0.07]$ — and so is the action, a throttle $a \in [-1, 1]$.
The reward is $-0.1a^2$ per step plus $+100$ for reaching the flag at
$x \ge 0.45$; episodes truncate at 999 steps. The engine is too weak to climb
directly, so the car must oscillate to build momentum.

Two facts dominate everything that follows:

- **The reward is sparse**: a random policy reaches the flag in only ~2% of episodes, and one-step Q-Learning propagates the +100 backwards roughly one grid cell per success.
- **The reward is deceptive**: doing nothing costs nothing, so until goal value propagates, "stand still" is the return-maximizing policy — a local optimum an agent can get *permanently* stuck in, since the greedy policy stops generating the experience that would correct it.

## 2. Evaluation methodology (used for every experiment)

Training return is noisy and epsilon-contaminated, so all configurations are
scored identically after training: **100 episodes with the greedy policy** on a
**fixed, seeded set of start states**, reporting mean return, its standard
deviation, and **success rate** (fraction of episodes reaching the flag). A
successful episode scores roughly +90 (100 minus fuel); a failed one roughly 0
or below. Every sweep configuration is trained with **2 seeds** to separate
parameter effects from luck. Experiments ran on a 4-core Linux container;
wall-clock times below are from `results/*_summary.csv`.

## 3. Task 1 — Discretization

Both spaces are mapped to uniform grids (`discretization.py`): `np.digitize`
over evenly spaced position/velocity edges, and `num_actions` evenly spaced
throttle values. The Q-table has shape `(pos_bins+1, vel_bins+1, num_actions)`.

### 3.1 The action set decides everything

The first sweep — state grids from 10×10 to 100×100, all with **5 actions**,
base hyperparameters, 3 000 episodes — produced a wall of zeros: every run,
every seed, return 0.0, success 0%. The learned policy is "do nothing". The
action-count sweep at 20×20 exposed the real variable
(`results/discretization_summary.csv`, fig2):

| Actions | Contains throttle 0? | Eval return (2 seeds) | Success |
|---|---|---|---|
| 3, 5, 11 | yes | 0.0 / 0.0 | 0% |
| 2 | no | 88.0 / 88.1 | 100% |
| 4 | no | 91.9 / 85.0 | 98–100% |

**An odd-sized action set contains a free idle action, which combined with the
fuel cost creates the local optimum; an even-sized set removes it
structurally** — every policy must move, motion doubles as exploration, and the
trap simply does not exist. This one discretization decision did more than any
hyperparameter (§5). Four actions beat two slightly: the ±1/3 throttles allow
cheaper fine control.

### 3.2 State-grid resolution (on the solvable 4-action set)

With 4 actions the grid sweep becomes informative
(`results/discretization_a4_summary.csv`, fig1):

| Grid | States | Eval return (2 seeds) | Success | Mean steps |
|---|---|---|---|---|
| 10×10 | 121 | 6.0 / 43.2 | 49–77% | 366–617 |
| **20×20** | **441** | **91.9 / 85.0** | **98–100%** | **159–183** |
| 50×50 | 2 601 | 77.8 / 83.7 | 84–90% | 328–389 |
| 100×100 | 10 201 | 84.3 / 91.8 | 92–100% | 534–548 |

The classic trade-off is visible at both ends: 10×10 aliases states that need
opposite actions (a cell containing both "moving left fast" and "barely
moving" gets one value and one action), while 50×50+ still solves the task but
each cell receives far less data in the same budget, so policies are cruder
and trajectories 2–3× longer. **Chosen discretization: 20×20 × 4 actions.**

## 4. Task 2 — Q-Learning

`q_learning_agent.py` implements one-step Q-Learning,
$Q(S,A) \mathrel{+}= \alpha[R + \gamma \max_a Q(S',a) - Q(S,A)]$, with
epsilon-greedy exploration decaying multiplicatively per episode from 1.0 to a
floor. On true termination the bootstrap term is dropped; on the 999-step
truncation it is kept (the time limit is not a real terminal state). Agents
save/load as pickled Q-tables (`models/*.pkl`).

## 5. Task 3 — Hyperparameter exploration

A full grid over five parameters is unaffordable, so we swept **one parameter
at a time around a base configuration** (α=0.1, γ=0.99, ε 1.0→0.05, decay
0.999, 3 000 episodes), 2 seeds each, scored as in §2.

**On the 5-action discretization** (`results/hyperparams_summary.csv`) the
sweep is almost uniformly zero — α ∈ {0.05…0.5}, γ ∈ {0.9…1.0}, every epsilon
schedule: no setting reliably escapes the idle trap in 3 000 episodes. The one
telling exception: α=0.5 escaped on one seed (83.5) and failed on the other
(0.0) — a large learning rate lets a single lucky success redirect the greedy
policy, but it is a coin flip. A longer probe (α=0.5, ε-floor 0.2, 8 000
episodes) escapes reliably; see the third final model.

**On the chosen 4-action discretization**
(`results/hyperparams_a4_summary.csv`, fig3), differences are consistent
across seeds:

| Parameter | Values (mean eval return over 2 seeds) | Reading |
|---|---|---|
| α | 0.05: 89.0 · 0.1: 88.5 · **0.2: 92.5** · 0.5: 85.3 | 0.2 best; 0.5 keeps values sloshing |
| γ | **0.9: 75.4** · 0.99: 88.5 · 0.999: 90.4 · 1.0: 91.3 | 0.9 makes the distant +100 nearly invisible ($0.9^{150}\!\approx\!10^{-7}$); ≥0.99 equivalent |
| ε decay | 0.997: 90.1 · 0.999: 88.5 · 0.9995: 90.0 | negligible |
| ε floor | 0.0: 88.5 · 0.05: 88.5 · 0.2: 88.0 | negligible — forced motion already explores |

**Final choice: α=0.2, γ=0.99, ε 1.0→0.05 with decay 0.999.** The broader
lesson: on this problem, representation (the action set) mattered far more
than any tuning; hyperparameters only modulate speed once the task is solvable.

## 6. Task 4 — Research component: Dyna-Q (§8.1–8.2)

Tabular Dyna-Q (the boxed algorithm, p. 164) extends every real Q-Learning
step with **(e)** model learning — `Model(S,A) ← (R, S')`, exact here because
the dynamics are deterministic — and **(f)** planning: *n* extra Q-Learning
updates on uniformly sampled previously-seen pairs, replayed from the model.
`dyna_q_agent.py` subclasses the Q-Learning agent, adding only (e) and (f), so
the comparison is exactly like-for-like.

The experiment gives every agent **600 real episodes on the 5-action
discretization** — deliberately the setting where plain Q-Learning is trapped —
with n ∈ {0, 5, 20, 50} (`results/dynaq_summary.csv`, figs 4–5):

| Agent | Eval return (2 seeds) | Success | Wall-clock |
|---|---|---|---|
| Q-Learning / Dyna-Q n=0 | 0.0 / 0.0 | 0% | 10–14 s |
| **Dyna-Q n=5** | **81.6 / 86.1** | **95–100%** | 25 s |
| Dyna-Q n=20 | 55.5 / 61.1 | 72–86% | 58 s |
| Dyna-Q n=50 | −27.9 / −27.9 | 0% | 128 s |

Three findings:

1. **n=0 reproduces plain Q-Learning exactly** — sanity check for the Dyna machinery.
2. **n=5 solves, in 600 episodes, the configuration plain Q-Learning failed at with 5× the budget** — idle action included. Each rare success is replayed hundreds of times by planning, so the +100 sweeps through the table instead of crawling one cell per success. This is §8.2's Figure 8.2 reproduced on a harder environment.
3. **More planning is not better — n=50 is catastrophic.** Uniform planning samples from *all* experience, and before the first success that experience contains only fuel costs; at n=50 those costs are hammered in ~50× faster than ε-greedy can find the goal, "move" actions go deeply negative, the greedy component stops moving, and the final policy burns fuel in a futile shuffle (−27.9, both seeds). This failure mode is precisely what motivates **prioritized sweeping (§8.4)**: replay transitions whose predecessors' values just changed, not random ones.

Planning buys sample efficiency with compute (wall-clock ≈ linear in n): a
superb exchange at n=5, ruinous at n=50.

## 7. Final models (`models/`)

| Model | Setting | Episodes | Eval return | Success | Train time |
|---|---|---|---|---|---|
| `final_qlearning.pkl` | 20×20 × 4 act, α=0.2, γ=0.99 | 10 000 | **93.4 ± 0.2** | **100%** | 39 s |
| `final_dynaq_n5.pkl` | 20×20 × 5 act, n=5, α=0.1 | 1 000 | 77.6 ± 15.8 | 99% | 35 s |
| `final_qlearning_5actions.pkl` | 20×20 × 5 act, α=0.5, ε-floor 0.2 | 8 000 | 87.2 ± 26.8 | 98% | 56 s |

(`results/final_summary.csv`, fig6.) The learned policy (fig7) is a
velocity-organized bang-bang controller — push *with* the current velocity to
pump energy into the oscillation, flipping throttle along a diagonal boundary
in state space — exactly the resonance strategy the physics demands, and the
value function peaks both near the flag and in the far-left "slingshot" region.

## 8. Conclusions

1. **Discretization was the decisive design choice.** The action grid's parity carries a hidden semantic (a free idle action), which combined with the fuel cost creates a local optimum tabular Q-Learning cannot reliably escape. Removing it structurally (even action count) solved the task outright; 20×20 states balances aliasing against data per cell.
2. **Q-Learning then works well**: ~90 return, ~100% success on held-out starts, ~150-step solutions, and a physically interpretable policy.
3. **Hyperparameters mattered less than representation**: only α (best 0.2) and γ (must be ≥0.99) had visible effects.
4. **Dyna-Q delivered the textbook §8.2 result** — n=5 planning steps rescued the trapped configuration with 5× fewer episodes — **and the textbook caveat**: uniform planning over cost-only experience is poison (n=50 fails outright), which is the motivation for prioritized sweeping.

**Warning notes.** Sweeps used 2 seeds per configuration (compute budget);
individual numbers carry a few points of seed noise, but every qualitative
conclusion is consistent across seeds. Truncation is handled as non-terminal
in all updates. No reward shaping was used anywhere — all results are on the
environment's own reward.

## AI usage disclosure

Generative AI (Claude, Anthropic) was used as a coding and drafting assistant
for this project: implementing the agents and experiment harness from the
techniques specified in the assignment, running the experiments, generating
figures, and drafting this report. All algorithms follow the cited course
material (Sutton & Barto §6.5, §8.1–8.2); all results were produced by
actually executing the code in this repository, and the analysis was verified
against the raw CSVs in `results/`.
