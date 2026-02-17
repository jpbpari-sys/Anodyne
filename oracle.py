import datetime
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List
from zoneinfo import ZoneInfo

import numpy as np
import pywt
import requests
import sympy
from scipy.signal import find_peaks


# ============================================================
# 1. CORE DATA STRUCTURES
# ============================================================


@dataclass
class SpectralPeak:
    time_idx: int
    frequency: float
    magnitude: float


@dataclass
class FrequencyRidge:
    times: np.ndarray
    frequencies: np.ndarray
    magnitudes: np.ndarray


@dataclass
class AgentDNA:
    prime_id: int
    traits: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not sympy.isprime(self.prime_id):
            raise ValueError("Invalid prime ID")

    def mutate(self, trait_key: str, mutation_rate: float = 0.1) -> None:
        if trait_key in self.traits:
            self.traits[trait_key] *= 1 + random.uniform(-mutation_rate, mutation_rate)


# ============================================================
# 2. PORTFOLIO & RISK ENGINE (Kelly-based)
# ============================================================


class PortfolioManager:
    def __init__(self, initial_capital: float = 100_000, fractional_kelly: float = 0.25) -> None:
        self.capital = initial_capital
        self.fractional_kelly = fractional_kelly
        self.win_rate = 0.55
        self.win_loss_ratio = 1.2

    def position_size(self, confidence: float) -> float:
        p = (self.win_rate * confidence) + (0.5 * (1 - confidence))
        b = self.win_loss_ratio
        q = 1 - p
        kelly = (b * p - q) / b
        safe_f = max(0, min(kelly * self.fractional_kelly, 0.20))
        return self.capital * safe_f


# ============================================================
# 3. MARKET INGESTION
# ============================================================


def fetch_btc_log_returns(limit: int = 300) -> np.ndarray:
    url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1s&limit={limit}"
    data = requests.get(url, timeout=10).json()
    prices = np.array([float(k[4]) for k in data])
    returns = np.diff(np.log(prices))
    return (returns - returns.mean()) / (returns.std() + 1e-8)


# ============================================================
# 4. SPECTRAL ENGINE
# ============================================================


def compute_cwt_peaks(signal: np.ndarray) -> tuple[list[SpectralPeak], np.ndarray]:
    widths = np.arange(1, 64)
    coeffs, freqs = pywt.cwt(signal, widths, "cmor1.5-1.0")
    power = np.abs(coeffs) ** 2

    peaks: list[SpectralPeak] = []
    for t in range(power.shape[1]):
        idx, _ = find_peaks(power[:, t], height=np.max(power[:, t]) * 0.1)
        for i in idx:
            peaks.append(SpectralPeak(t, freqs[i], power[i, t]))
    return peaks, freqs


# ============================================================
# 5. RIDGE SWARM (Incremental)
# ============================================================


class RidgeSwarm:
    def __init__(self, max_freq_jump: float = 0.12, max_gap: int = 3, min_len: int = 6) -> None:
        self.ridges: List[List[SpectralPeak]] = []
        self.active: Dict[int, int] = {}
        self.params = dict(max_freq_jump=max_freq_jump, max_gap=max_gap, min_len=min_len)

    def update(self, peaks: List[SpectralPeak]) -> None:
        for p in sorted(peaks, key=lambda x: x.time_idx):
            matched = False
            for ridx, last_t in list(self.active.items()):
                if p.time_idx - last_t <= self.params["max_gap"]:
                    last_f = self.ridges[ridx][-1].frequency
                    if (
                        abs(p.frequency - last_f) / max(last_f, 1e-8)
                        <= self.params["max_freq_jump"]
                    ):
                        self.ridges[ridx].append(p)
                        self.active[ridx] = p.time_idx
                        matched = True
                        break
            if not matched:
                self.ridges.append([p])
                self.active[len(self.ridges) - 1] = p.time_idx

    def get_ridges(self) -> List[FrequencyRidge]:
        out = []
        for r in self.ridges:
            if len(r) >= self.params["min_len"]:
                out.append(
                    FrequencyRidge(
                        np.array([p.time_idx for p in r]),
                        np.array([p.frequency for p in r]),
                        np.array([p.magnitude for p in r]),
                    )
                )
        return out


# ============================================================
# 6. RIDGE → SIGNAL
# ============================================================


def ridge_score(r: FrequencyRidge, T: int) -> float:
    recency = 1 / (T - r.times[-1] + 1)
    duration = len(r.times) / 10
    strength = np.mean(r.magnitudes)
    return strength * recency * duration


def ridge_to_trade(r: FrequencyRidge) -> tuple[str, float, float]:
    slope, _ = np.polyfit(r.times, r.frequencies, 1)
    confidence = min(np.mean(r.magnitudes) * 8, 1.0)
    signal = "LONG" if slope > 0 else "SHORT"
    return signal, confidence, slope


# ============================================================
# 7. SOVEREIGN MEMORY & AGENT SWARM WITH EVOLUTION
# ============================================================


@dataclass
class Draft:
    platform: str
    content: str
    created_at: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(ZoneInfo("UTC"))
    )


class Memory:
    def __init__(self) -> None:
        self.drafts: List[Draft] = []


memory = Memory()


class NarratorAgent:
    def __init__(self, name: str, dna: AgentDNA) -> None:
        self.name = name
        self.dna = dna

    def narrate(self, signal: str, confidence: float) -> None:
        msg = f"{self.name}: {signal} detected | confidence {confidence:.2f}"
        memory.drafts.append(Draft(platform="X", content=msg))


class SwarmAgent:
    def __init__(self, prime_id: int, params: Dict[str, float]) -> None:
        self.dna = AgentDNA(prime_id)
        self.dna.traits = params
        self.ridge_swarm = RidgeSwarm(**params)
        self.fitness = 0.0

    def update(self, peaks: List[SpectralPeak]) -> None:
        self.ridge_swarm.update(peaks)

    def compute_fitness(self, ridges: List[FrequencyRidge], T: int) -> None:
        if ridges:
            self.fitness = max(ridge_score(r, T) for r in ridges)
        else:
            self.fitness = 0.0


class EvolutionarySwarm:
    def __init__(self, population_size: int = 100) -> None:
        self.population: List[SwarmAgent] = []
        self.narrators = []
        for i in range(population_size):
            prime_id = sympy.nextprime(random.randint(10**6, 10**7))
            params = {
                "max_freq_jump": random.uniform(0.05, 0.2),
                "max_gap": random.randint(2, 5),
                "min_len": random.randint(4, 10),
            }
            agent = SwarmAgent(prime_id, params)
            self.population.append(agent)
            self.narrators.append(NarratorAgent(f"Agent-{i}", AgentDNA(prime_id)))

        self.generation = 0

    def evolve(self) -> None:
        self.population.sort(key=lambda a: a.fitness, reverse=True)
        survivors = self.population[: len(self.population) // 2]
        offspring = []
        for s in survivors:
            new_prime = sympy.nextprime(s.dna.prime_id + random.randint(1, 1000))
            new_params = s.dna.traits.copy()
            child_dna = AgentDNA(new_prime)
            child_dna.traits = new_params
            child_dna.mutate("max_freq_jump")
            child_dna.mutate("max_gap")
            child_dna.mutate("min_len")
            child_dna.traits["max_gap"] = int(child_dna.traits["max_gap"])
            child_dna.traits["min_len"] = int(child_dna.traits["min_len"])
            offspring.append(SwarmAgent(new_prime, child_dna.traits))
        self.population = survivors + offspring
        self.narrators = [
            NarratorAgent(f"Agent-{i}", a.dna) for i, a in enumerate(self.population)
        ]
        self.generation += 1
        print(f"Evolved to generation {self.generation}")

    def get_best_agent(self) -> SwarmAgent:
        return max(self.population, key=lambda a: a.fitness)

    def broadcast(self, signal: str, confidence: float) -> None:
        for n in self.narrators:
            n.narrate(signal, confidence)


# ============================================================
# 8. SECURITY PROTOCOL ELEMENTS (Simplified for Integration)
# ============================================================


def verify_agent(agent: SwarmAgent) -> bool:
    return sympy.isprime(agent.dna.prime_id)


# ============================================================
# 9. ORCHESTRATOR (THE BRAIN)
# ============================================================


def run_oracle() -> None:
    evo_swarm = EvolutionarySwarm(100)
    portfolio = PortfolioManager()

    print("\n🧠 BTC AGENTIC ORACLE - SOVEREIGN AGI ENGINE ONLINE\n")

    cycle_count = 0
    while True:
        signal_data = fetch_btc_log_returns()
        peaks, _freqs = compute_cwt_peaks(signal_data)
        T = len(signal_data)

        for agent in evo_swarm.population:
            if verify_agent(agent):
                agent.update(peaks)
                ridges = agent.ridge_swarm.get_ridges()
                agent.compute_fitness(ridges, T)

        best_agent = evo_swarm.get_best_agent()
        ridges = best_agent.ridge_swarm.get_ridges()
        if not ridges:
            time.sleep(2)
            continue

        best_ridge = max(ridges, key=lambda r: ridge_score(r, T))
        signal, conf, slope = ridge_to_trade(best_ridge)
        size = portfolio.position_size(conf)

        print(
            f"{signal} | conf={conf:.2f} | slope={slope:.6f} | size=${size:,.0f} | "
            f"gen={evo_swarm.generation}"
        )

        evo_swarm.broadcast(signal, conf)

        cycle_count += 1
        if cycle_count % 10 == 0:
            evo_swarm.evolve()

        time.sleep(2)


# ============================================================
# 10. EXECUTION
# ============================================================


if __name__ == "__main__":
    run_oracle()
