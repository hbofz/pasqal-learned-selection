"""Real market graphs and portfolios: membership as of a date, formation returns, the conflict graph, holding returns.

No network: `docs/superpowers/stage6/fetch6.py` writes the snapshot this reads. A return exists only between consecutive
trading days that both have a price, so a missing day never spreads a move over two days. Nothing here reads a price on
or after a rebalance date to build a graph, and nothing reads one before it to score a holding.
"""
import csv, hashlib
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

FORMATION_DAYS, MIN_RETURNS = 250, 240
THETAS = (0.3, 0.4, 0.5, 0.6, 0.7)
EPISODES = (("2008-09-01", "2009-06-30"), ("2020-02-15", "2020-06-30"), ("2022-01-01", "2022-12-31"))
FORMATION_CRISIS_DAYS = 60
# Yahoo's continuous history here is another company's, with no late start or gap to catch it: ticker -> dropped before.
TAKEOVERS = {"T": "2006-01-01"}     # AT&T Corp until SBC took the ticker in late 2005


def ticker(raw: str) -> str:
    return raw.replace(".", "-")


class Market:
    def __init__(self, close: pd.DataFrame, membership: list[tuple[str, list[str]]]):
        self.close = close.sort_index()
        self.log_returns = np.log(self.close).diff()
        self.simple_returns = self.close / self.close.shift(1) - 1
        self.member_dates = pd.DatetimeIndex([d for d, _ in membership])
        self.members = [m for _, m in membership]

    @classmethod
    def load(cls, folder="cache/market") -> "Market":
        folder = Path(folder)
        close = pd.read_csv(folder / "close.csv.gz", index_col=0, parse_dates=True)
        csv.field_size_limit(10 ** 7)
        with (folder / "membership.csv").open() as f:
            rows = [(r["date"], [ticker(t) for t in r["tickers"].split(",")]) for r in csv.DictReader(f)]
        return cls(close, rows)

    def members_at(self, date) -> list[str]:
        """The list in force on `date`: the latest one dated on or before it."""
        i = self.member_dates.searchsorted(pd.Timestamp(date), side="right") - 1
        if i < 0:
            raise ValueError(f"no membership list on or before {date}")
        return self.members[i]

    def rebalance_dates(self, first="2005-01-01", last="2026-10-31") -> pd.DatetimeIndex:
        """The first trading day of each January, April, July and October in the range."""
        days = self.close.index[(self.close.index >= first) & (self.close.index <= last)]
        s = pd.Series(days, index=days)
        firsts = s.groupby([days.year, days.month]).first()
        return pd.DatetimeIndex([d for d in firsts if d.month in (1, 4, 7, 10)])

    def _position(self, date) -> int:
        i = self.close.index.searchsorted(pd.Timestamp(date))
        if i >= len(self.close.index) or self.close.index[i] != pd.Timestamp(date):
            raise ValueError(f"{date} is not a trading day")
        return i

    def formation(self, date) -> pd.DataFrame:
        """Log returns of the FORMATION_DAYS trading days ending the day before `date`."""
        i = self._position(date)
        return self.log_returns.iloc[i - FORMATION_DAYS:i]

    def universe(self, date) -> list[str]:
        """Members on `date` with enough formation returns, in membership order, without the takeover cases."""
        w, day = self.formation(date), pd.Timestamp(date)
        keep = [t for t in self.members_at(date) if t in w.columns and w[t].notna().sum() >= MIN_RETURNS]
        return [t for t in keep if t not in TAKEOVERS or day >= pd.Timestamp(TAKEOVERS[t])]

    def correlations(self, date) -> tuple[list[str], np.ndarray]:
        names = self.universe(date)
        return names, self.formation(date)[names].corr().to_numpy()

    def graph(self, date, theta: float) -> tuple[nx.Graph, list[str]]:
        """Node i is names[i]; an edge when the correlation exceeds theta, positive only. An undefined one is no edge."""
        names, C = self.correlations(date)
        G = nx.Graph()
        G.add_nodes_from(range(len(names)))
        G.add_edges_from(np.argwhere(np.triu(C > theta, k=1)).tolist())
        nx.set_node_attributes(G, 1, "weight")
        return G, names

    def holding(self, date, next_date) -> pd.DataFrame:
        """Simple returns from `date`'s close through `next_date`'s close."""
        return self.simple_returns.iloc[self._position(date) + 1:self._position(next_date) + 1]

    def episode_formation(self, date) -> int | None:
        """The episode holding at least FORMATION_CRISIS_DAYS of this formation window; None is calm."""
        i = self._position(date)
        days = self.close.index[i - FORMATION_DAYS:i]
        return next((e for e, (a, b) in enumerate(EPISODES) if ((days >= a) & (days <= b)).sum() >= FORMATION_CRISIS_DAYS), None)

    def episode_holding(self, date, next_date) -> int | None:
        """The episode this holding window overlaps; None is calm."""
        days = self.close.index[self._position(date) + 1:self._position(next_date) + 1]
        return next((e for e, (a, b) in enumerate(EPISODES) if ((days >= a) & (days <= b)).any()), None)

    def audit(self) -> dict:
        """The ticker check: members never priced, and those priced only after they first joined."""
        joined = {}
        for d, m in zip(self.member_dates, self.members):
            for t in m:
                joined.setdefault(t, d)
        start = self.close.apply(pd.Series.first_valid_index)
        start = {t: d for t, d in start.items() if d is not None and not pd.isna(d)}
        window = self.close.index[0]
        late = sorted(t for t, j in joined.items() if t in start and start[t] > max(j, window))
        return dict(tickers=len(joined), never_priced=sorted(t for t in joined if t not in start), late_start=late)


def block_ci(values, block=4, reps=2000, seed=0) -> tuple[float, float, float]:
    """Mean and 95% percentile interval, drawing moving blocks of consecutive values, for values in date order."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    b = min(block, len(v))
    starts = rng.integers(0, len(v) - b + 1, (reps, int(np.ceil(len(v) / b))))
    draws = np.array([np.concatenate([v[s:s + b] for s in row])[:len(v)].mean() for row in starts])
    return float(v.mean()), *np.percentile(draws, [2.5, 97.5]).tolist()


def episode_ci(values, episodes, reps=2000, seed=0) -> tuple[float, float, float]:
    """Mean and 95% percentile interval, drawing dates within each episode."""
    v, e = np.asarray(values, float), np.asarray(episodes)
    rng = np.random.default_rng(seed)
    draws = np.zeros((reps, len(v)))
    at = 0
    for x in np.unique(e):
        part = v[e == x]
        draws[:, at:at + len(part)] = part[rng.integers(0, len(part), (reps, len(part)))]
        at += len(part)
    return float(v.mean()), *np.percentile(draws.mean(1), [2.5, 97.5]).tolist()


def word(ci, episode_means=None, below="gain", above="worse") -> str:
    """Below or above 0 by the interval; in a crisis pool, every episode must also share the sign."""
    side = -1 if ci[2] < 0 else 1 if ci[1] > 0 else 0
    if side == 0 or (episode_means is not None and any(np.sign(m) != side for m in episode_means)):
        return "not resolved"
    return below if side < 0 else above


def portfolio_returns(returns: pd.DataFrame, names) -> pd.Series:
    """Equal weight, reset daily; a stock with no return on a day is left out of that day's mean."""
    return returns[list(names)].mean(axis=1, skipna=True)


def volatility(daily: pd.Series) -> float:
    return float(daily.std(ddof=1) * np.sqrt(252))


def max_drawdown(daily: pd.Series) -> float:
    wealth = np.concatenate([[1.0], (1 + daily.fillna(0).to_numpy()).cumprod()])
    return float((wealth / np.maximum.accumulate(wealth) - 1).min())


def total_return(daily: pd.Series) -> float:
    return float((1 + daily.fillna(0)).prod() - 1)


def random_sets(names: list[str], size: int, count: int, key: str) -> list[list[str]]:
    """`count` uniform subsets of `size`, reproducible from `key` alone."""
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))
    return [[names[i] for i in rng.choice(len(names), size, replace=False)] for _ in range(count)]
