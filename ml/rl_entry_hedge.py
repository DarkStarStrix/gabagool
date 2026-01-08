#!/usr/bin/env python3
"""
Reinforcement learning baseline for entry/hedge timing.

Inputs per timestep:
- Polymarket orderbook top-of-book (bid/ask)
- Chainlink price
- Binance price
- Binance CVD

Outputs:
- Action: HOLD, ENTER, HEDGE

This is intentionally dependency-light (stdlib only) so it can run in
minimal environments. Swap the discretizer and reward function to match
your desired objective and risk profile.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
import csv
import random
import statistics
from typing import Deque, Dict, List, Optional, Sequence, Tuple

class Action(IntEnum):
    HOLD = 0
    ENTER = 1
    HEDGE = 2


@dataclass(frozen=True)
class MarketSnapshot:
    timestamp: float
    polymarket_bid: float
    polymarket_ask: float
    chainlink_price: float
    binance_price: float
    binance_cvd: float

    @property
    def mid(self) -> float:
        return (self.polymarket_bid + self.polymarket_ask) / 2

    @property
    def spread(self) -> float:
        return max(self.polymarket_ask - self.polymarket_bid, 0.0)


@dataclass(frozen=True)
class FeatureVector:
    spread_bps: float
    price_diff_bps: float
    cvd_delta: float
    pm_vs_chainlink_bps: float


@dataclass
class PositionState:
    in_position: bool = False
    entry_price: Optional[float] = None


class FeatureBuilder:
    def build(self, current: MarketSnapshot, previous: Optional[MarketSnapshot]) -> FeatureVector:
        spread_bps = _bps(current.spread, current.mid)
        price_diff_bps = _bps(current.binance_price - current.chainlink_price, current.chainlink_price)
        pm_vs_chainlink_bps = _bps(current.mid - current.chainlink_price, current.chainlink_price)
        cvd_delta = 0.0
        if previous is not None:
            cvd_delta = current.binance_cvd - previous.binance_cvd
        return FeatureVector(
            spread_bps=spread_bps,
            price_diff_bps=price_diff_bps,
            cvd_delta=cvd_delta,
            pm_vs_chainlink_bps=pm_vs_chainlink_bps,
        )


class StateDiscretizer:
    """Simple bucketizer for converting features into discrete state keys."""

    def __init__(self, spread_bins: Sequence[float], price_diff_bins: Sequence[float], cvd_bins: Sequence[float], pm_bins: Sequence[float]):
        self.spread_bins = list(spread_bins)
        self.price_diff_bins = list(price_diff_bins)
        self.cvd_bins = list(cvd_bins)
        self.pm_bins = list(pm_bins)

    def discretize(self, features: FeatureVector, in_position: bool) -> Tuple[int, int, int, int, int]:
        return (
            _bucket(features.spread_bps, self.spread_bins),
            _bucket(features.price_diff_bps, self.price_diff_bins),
            _bucket(features.cvd_delta, self.cvd_bins),
            _bucket(features.pm_vs_chainlink_bps, self.pm_bins),
            int(in_position),
        )


class EntryHedgeEnv:
    """Toy environment for deciding when to enter and hedge."""

    def __init__(self, snapshots: Sequence[MarketSnapshot], feature_builder: FeatureBuilder, discretizer: StateDiscretizer):
        if len(snapshots) < 2:
            raise ValueError("Need at least 2 snapshots to train")
        self.snapshots = snapshots
        self.feature_builder = feature_builder
        self.discretizer = discretizer
        self.reset()

    def reset(self) -> Tuple[int, int, int, int, int]:
        self.index = 0
        self.position = PositionState()
        self.prev_snapshot: Optional[MarketSnapshot] = None
        return self._current_state()

    def step(self, action: Action) -> Tuple[Tuple[int, int, int, int, int], float, bool]:
        current = self.snapshots[self.index]
        reward = 0.0

        if action == Action.ENTER and not self.position.in_position:
            self.position.in_position = True
            self.position.entry_price = current.mid
        elif action == Action.HEDGE and self.position.in_position:
            reward += self._close_position(current)
        elif action == Action.HOLD and self.position.in_position:
            reward += self._mark_to_market(current)

        self.prev_snapshot = current
        self.index += 1
        done = self.index >= len(self.snapshots) - 1

        if done and self.position.in_position:
            reward += self._close_position(current)

        return self._current_state(), reward, done

    def _current_state(self) -> Tuple[int, int, int, int, int]:
        current = self.snapshots[self.index]
        features = self.feature_builder.build(current, self.prev_snapshot)
        return self.discretizer.discretize(features, self.position.in_position)

    def _close_position(self, snapshot: MarketSnapshot) -> float:
        if not self.position.in_position or self.position.entry_price is None:
            return 0.0
        reward = snapshot.mid - self.position.entry_price
        self.position = PositionState()
        return reward

    def _mark_to_market(self, snapshot: MarketSnapshot) -> float:
        if not self.position.in_position or self.position.entry_price is None:
            return 0.0
        return snapshot.mid - self.position.entry_price


class QLearningAgent:
    def __init__(self, actions: Sequence[Action], alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1):
        self.actions = list(actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: Dict[Tuple[int, int, int, int, int], List[float]] = {}

    def select_action(self, state: Tuple[int, int, int, int, int]) -> Action:
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = self._get_q_values(state)
        best_index = max(range(len(self.actions)), key=lambda i: q_values[i])
        return self.actions[best_index]

    def update(self, state: Tuple[int, int, int, int, int], action: Action, reward: float, next_state: Tuple[int, int, int, int, int]) -> float:
        q_values = self._get_q_values(state)
        next_q_values = self._get_q_values(next_state)
        action_index = self.actions.index(action)

        td_target = reward + self.gamma * max(next_q_values)
        td_error = td_target - q_values[action_index]
        q_values[action_index] += self.alpha * td_error
        return td_error

    def _get_q_values(self, state: Tuple[int, int, int, int, int]) -> List[float]:
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in self.actions]
        return self.q_table[state]


def train_agent(snapshots: Sequence[MarketSnapshot], episodes: int = 20, log_every_steps: int = 100) -> QLearningAgent:
    feature_builder = FeatureBuilder()
    discretizer = StateDiscretizer(
        spread_bins=[0.5, 1.0, 2.0, 5.0],
        price_diff_bins=[-50, -10, 0, 10, 50],
        cvd_bins=[-1000, -250, 0, 250, 1000],
        pm_bins=[-50, -10, 0, 10, 50],
    )
    env = EntryHedgeEnv(snapshots, feature_builder, discretizer)
    agent = QLearningAgent(actions=list(Action))

    loss_window: Deque[float] = deque(maxlen=1000)

    for episode in range(episodes):
        state = env.reset()
        done = False
        step = 0
        episode_losses: List[float] = []
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            td_error = agent.update(state, action, reward, next_state)
            loss = float(td_error ** 2)
            episode_losses.append(loss)
            loss_window.append(loss)
            if log_every_steps and step % log_every_steps == 0:
                avg_loss = float(statistics.fmean(loss_window)) if loss_window else 0.0
                print(f"[episode {episode + 1}] step {step} loss={loss:.6f} avg_loss={avg_loss:.6f}")
            state = next_state
            step += 1

        episode_avg_loss = float(statistics.fmean(episode_losses)) if episode_losses else 0.0
        print(f"[episode {episode + 1}] avg_loss={episode_avg_loss:.6f} steps={step}")

    return agent


def load_snapshots_from_csv(path: str) -> List[MarketSnapshot]:
    snapshots: List[MarketSnapshot] = []
    with open(path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        required = {
            "timestamp",
            "polymarket_bid",
            "polymarket_ask",
            "chainlink_price",
            "binance_price",
            "binance_cvd",
        }
        if not required.issubset(reader.fieldnames or []):
            missing = required.difference(reader.fieldnames or [])
            raise ValueError(f"CSV missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            snapshots.append(
                MarketSnapshot(
                    timestamp=float(row["timestamp"]),
                    polymarket_bid=float(row["polymarket_bid"]),
                    polymarket_ask=float(row["polymarket_ask"]),
                    chainlink_price=float(row["chainlink_price"]),
                    binance_price=float(row["binance_price"]),
                    binance_cvd=float(row["binance_cvd"]),
                )
            )
    return snapshots


def _bucket(value: float, bins: Sequence[float]) -> int:
    for idx, edge in enumerate(bins):
        if value <= edge:
            return idx
    return len(bins)


def _bps(delta: float, reference: float) -> float:
    if reference == 0:
        return 0.0
    return (delta / reference) * 10000


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train a Q-learning agent for entry/hedge timing.")
    parser.add_argument("csv", help="Path to CSV with market data")
    parser.add_argument("--episodes", type=int, default=20, help="Training episodes")
    parser.add_argument("--log-every", type=int, default=100, help="Log loss every N steps")
    args = parser.parse_args()

    snapshots = load_snapshots_from_csv(args.csv)
    agent = train_agent(snapshots, episodes=args.episodes, log_every_steps=args.log_every)

    print(f"Trained Q-table states: {len(agent.q_table)}")


if __name__ == "__main__":
    main()
