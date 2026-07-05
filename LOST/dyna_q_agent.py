"""Dyna-Q agent for MountainCarContinuous-v0.

Implements Tabular Dyna-Q from Sutton & Barto, *Reinforcement Learning: An
Introduction* (2nd ed.), Sections 8.1-8.2 (the boxed algorithm on p. 164).

Dyna-Q augments one-step Q-Learning with a learned model and planning:

  (d) direct RL:   Q(S,A) <- Q(S,A) + alpha [R + gamma max_a Q(S',a) - Q(S,A)]
  (e) model learn: Model(S,A) <- R, S'   (deterministic-environment assumption)
  (f) planning:    repeat n times: sample a previously seen (S,A), replay the
                   stored (R, S') through the same Q-Learning update.

The mountain-car dynamics are deterministic, so the last-outcome model of the
textbook algorithm is exact once a (state, action) pair has been visited.
Planning re-propagates the sparse +100 goal reward backwards through the
table many times per real environment step, which is exactly what one-step
Q-Learning is slow at.
"""

import time

import numpy as np

from q_learning_agent import QLearningAgent


class DynaQAgent(QLearningAgent):
    def __init__(self, discretizer=None, planning_steps=10, seed=None):
        super().__init__(discretizer, seed=seed)
        self.planning_steps = planning_steps
        # Model(S, A) -> (R, S', terminated). Keys are (state, action_idx).
        self.model = {}
        # Parallel list of keys for O(1) uniform sampling.
        self._model_keys = []

    def _learn_model(self, state, action_idx, reward, next_state, terminated):
        key = (state, action_idx)
        if key not in self.model:
            self._model_keys.append(key)
        self.model[key] = (reward, next_state, terminated)

    def _plan(self, gamma, alpha):
        n_seen = len(self._model_keys)
        if n_seen == 0:
            return
        idxs = self.rng.integers(n_seen, size=self.planning_steps)
        for i in idxs:
            state, action_idx = self._model_keys[i]
            reward, next_state, terminated = self.model[(state, action_idx)]
            self._update(state, action_idx, reward, next_state, terminated, gamma, alpha)

    def train_agent(
        self,
        env,
        episodes=1000,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999,
        gamma=0.99,
        alpha=0.1,
        log_every=None,
    ):
        history = {"reward": [], "steps": [], "success": [], "epsilon": []}
        start = time.perf_counter()
        for episode in range(episodes):
            obs, _ = env.reset(seed=int(self.rng.integers(2**31)) if episode == 0 else None)
            state = self.discretizer.state(obs)
            done = False
            total_reward = 0.0
            steps = 0
            success = False
            while not done:
                action_idx = self._epsilon_greedy(state, epsilon)
                obs, reward, terminated, truncated, _ = env.step(
                    self.discretizer.action(action_idx)
                )
                next_state = self.discretizer.state(obs)
                # (d) direct reinforcement learning
                self._update(state, action_idx, reward, next_state, terminated, gamma, alpha)
                # (e) model learning
                self._learn_model(state, action_idx, reward, next_state, terminated)
                # (f) planning: n simulated updates from the model
                self._plan(gamma, alpha)
                state = next_state
                total_reward += reward
                steps += 1
                done = terminated or truncated
                success = success or terminated

            history["reward"].append(total_reward)
            history["steps"].append(steps)
            history["success"].append(success)
            history["epsilon"].append(epsilon)
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            if log_every and (episode + 1) % log_every == 0:
                window = history["reward"][-log_every:]
                rate = np.mean(history["success"][-log_every:])
                print(
                    f"episode {episode + 1}/{episodes} "
                    f"avg_reward={np.mean(window):.1f} success_rate={rate:.2f} "
                    f"epsilon={epsilon:.3f} model_size={len(self.model)}"
                )
        history["train_seconds"] = time.perf_counter() - start
        return history
