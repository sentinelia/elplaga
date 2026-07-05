"""Tabular Q-Learning agent for MountainCarContinuous-v0.

Implements one-step Q-Learning (Watkins, 1989; Sutton & Barto, Section 6.5):

    Q(S, A) <- Q(S, A) + alpha * [R + gamma * max_a Q(S', a) - Q(S, A)]

on top of the uniform-grid discretization defined in ``discretization.py``.
Exploration is epsilon-greedy with an optional per-episode multiplicative
decay so training starts almost fully random (the goal reward can only be
discovered by chance) and gradually shifts to exploiting the learned values.
"""

import pickle
import time

import numpy as np

from discretization import Discretizer


class QLearningAgent:
    def __init__(self, discretizer=None, seed=None):
        self.discretizer = discretizer or Discretizer()
        self.Q = np.zeros(self.discretizer.q_table_shape)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Policies
    # ------------------------------------------------------------------
    def next_action(self, obs):
        """Greedy action (as a continuous env action) for a raw observation."""
        state = self.discretizer.state(obs)
        return self.discretizer.action(int(np.argmax(self.Q[state])))

    def _epsilon_greedy(self, state, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.discretizer.num_actions))
        return int(np.argmax(self.Q[state]))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def _update(self, state, action_idx, reward, next_state, terminated, gamma, alpha):
        # The bootstrap term is dropped on true termination (goal reached):
        # the value of a terminal state is 0 by definition.
        target = reward
        if not terminated:
            target += gamma * np.max(self.Q[next_state])
        self.Q[state][action_idx] += alpha * (target - self.Q[state][action_idx])

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
        """Train with epsilon-greedy Q-Learning.

        Returns a history dict with per-episode return, steps, success flag
        and the epsilon used, plus total wall-clock time.
        """
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
                self._update(state, action_idx, reward, next_state, terminated, gamma, alpha)
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
                    f"epsilon={epsilon:.3f}"
                )
        history["train_seconds"] = time.perf_counter() - start
        return history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def test_agent(self, env, episodes=10, render=False):
        """Run the greedy policy; returns per-episode rewards, steps, successes."""
        results = {"reward": [], "steps": [], "success": []}
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            success = False
            while not done:
                obs, reward, terminated, truncated, _ = env.step(self.next_action(obs))
                total_reward += reward
                steps += 1
                done = terminated or truncated
                success = success or terminated
                if render:
                    env.render()
            results["reward"].append(total_reward)
            results["steps"].append(steps)
            results["success"].append(success)
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "Q": self.Q,
                    "position_bins": self.discretizer.position_bins,
                    "velocity_bins": self.discretizer.velocity_bins,
                    "num_actions": self.discretizer.num_actions,
                },
                f,
            )

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(
            Discretizer(
                position_bins=data["position_bins"],
                velocity_bins=data["velocity_bins"],
                num_actions=data["num_actions"],
            )
        )
        agent.Q = data["Q"]
        return agent
