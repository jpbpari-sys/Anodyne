import asyncio
import datetime
import json
import random
import threading
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pywt
import requests
import sympy
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from scipy.signal import find_peaks
from zoneinfo import ZoneInfo

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


# ============================================================
# 2. PORTFOLIO (FRACTIONAL KELLY)
# ============================================================


class PortfolioManager:
    def __init__(self, capital: float = 100_000, frac_kelly: float = 0.25) -> None:
        self.capital = capital
        self.win_rate = 0.55
        self.win_loss = 1.2
        self.frac_kelly = frac_kelly

    def size(self, confidence: float) -> float:
        p = (self.win_rate * confidence) + 0.5 * (1 - confidence)
        b = self.win_loss
        q = 1 - p
        k = (b * p - q) / b
        f = max(0, min(k * self.frac_kelly, 0.2))
        return self.capital * f


# ============================================================
# 3. MARKET INGESTION
# ============================================================


def fetch_returns(limit: int = 300) -> tuple[np.ndarray, float]:
    url = (
        "https://api.binance.com/api/v3/klines"
        f"?symbol=BTCUSDT&interval=1s&limit={limit}"
    )
    data = requests.get(url, timeout=5).json()
    prices = np.array([float(k[4]) for k in data])
    returns = np.diff(np.log(prices))
    standardized = (returns - returns.mean()) / (returns.std() + 1e-8)
    return standardized, prices[-1]


# ============================================================
# 4. SPECTRAL ENGINE
# ============================================================


def compute_peaks(signal: np.ndarray) -> list[SpectralPeak]:
    widths = np.arange(1, 64)
    coeffs, freqs = pywt.cwt(signal, widths, "cmor1.5-1.0")
    power = np.abs(coeffs) ** 2
    peaks: list[SpectralPeak] = []
    for t in range(power.shape[1]):
        idx, _ = find_peaks(power[:, t], height=np.max(power[:, t]) * 0.1)
        for i in idx:
            peaks.append(SpectralPeak(t, freqs[i], power[i, t]))
    return peaks


# ============================================================
# 5. RIDGE SWARM
# ============================================================


class RidgeSwarm:
    def __init__(self, jf: float = 0.12, gap: int = 3, minlen: int = 6) -> None:
        self.ridges: list[list[SpectralPeak]] = []
        self.active: dict[int, int] = {}
        self.jf = jf
        self.gap = gap
        self.minlen = minlen

    def update(self, peaks: list[SpectralPeak]) -> None:
        for peak in sorted(peaks, key=lambda x: x.time_idx):
            matched = False
            for idx, last_time in list(self.active.items()):
                if peak.time_idx - last_time <= self.gap:
                    last_freq = self.ridges[idx][-1].frequency
                    if abs(peak.frequency - last_freq) / max(last_freq, 1e-8) <= self.jf:
                        self.ridges[idx].append(peak)
                        self.active[idx] = peak.time_idx
                        matched = True
                        break
            if not matched:
                self.ridges.append([peak])
                self.active[len(self.ridges) - 1] = peak.time_idx

    def get(self) -> list[FrequencyRidge]:
        output = []
        for ridge in self.ridges:
            if len(ridge) >= self.minlen:
                output.append(
                    FrequencyRidge(
                        np.array([p.time_idx for p in ridge]),
                        np.array([p.frequency for p in ridge]),
                        np.array([p.magnitude for p in ridge]),
                    )
                )
        return output


# ============================================================
# 6. EVOLUTIONARY AGENTS
# ============================================================


@dataclass
class Agent:
    prime: int
    params: Dict[str, float]
    swarm: RidgeSwarm
    fitness: float = 0.0


def ridge_score(ridge: FrequencyRidge, horizon: int) -> float:
    return np.mean(ridge.magnitudes) * (len(ridge.times) / 10) / (
        horizon - ridge.times[-1] + 1
    )


class Evolution:
    def __init__(self, n: int = 40) -> None:
        self.pop: list[Agent] = []
        for _ in range(n):
            prime = sympy.nextprime(random.randint(1_000_000, 9_000_000))
            params = {
                "jf": random.uniform(0.05, 0.2),
                "gap": random.randint(2, 5),
                "minlen": random.randint(4, 9),
            }
            self.pop.append(Agent(prime, params, RidgeSwarm(**params)))
        self.gen = 0

    def evolve(self) -> None:
        self.pop.sort(key=lambda a: a.fitness, reverse=True)
        keep = self.pop[: len(self.pop) // 2]
        children = []
        for agent in keep:
            prime = sympy.nextprime(agent.prime + random.randint(1, 1000))
            params = {
                k: v * (1 + random.uniform(-0.1, 0.1)) for k, v in agent.params.items()
            }
            params["gap"] = int(params["gap"])
            params["minlen"] = int(params["minlen"])
            children.append(Agent(prime, params, RidgeSwarm(**params)))
        self.pop = keep + children
        self.gen += 1


# ============================================================
# 7. SHARED STATE (WEB PORTAL)
# ============================================================


STATE = {
    "price": 0,
    "signal": "HOLD",
    "confidence": 0,
    "size": 0,
    "generation": 0,
    "fitness": 0,
}
AGENTS: list[dict] = []
EVENTS: list[str] = []


# ============================================================
# 8. ORACLE LOOP (THREAD)
# ============================================================


def oracle_loop() -> None:
    evo = Evolution()
    pf = PortfolioManager()
    while True:
        try:
            signal, price = fetch_returns()
            peaks = compute_peaks(signal)
            horizon = len(signal)
            for agent in evo.pop:
                agent.swarm.update(peaks)
                ridges = agent.swarm.get()
                agent.fitness = max(
                    [ridge_score(ridge, horizon) for ridge in ridges], default=0
                )

            best = max(evo.pop, key=lambda a: a.fitness)
            ridges = best.swarm.get()
            if ridges:
                ridge = max(ridges, key=lambda x: ridge_score(x, horizon))
                slope, _ = np.polyfit(ridge.times, ridge.frequencies, 1)
                confidence = min(np.mean(ridge.magnitudes) * 8, 1)
                signal = "LONG" if slope > 0 else "SHORT"
                size = pf.size(confidence)

                STATE.update(
                    {
                        "price": price,
                        "signal": signal,
                        "confidence": confidence,
                        "size": size,
                        "generation": evo.gen,
                        "fitness": best.fitness,
                    }
                )
                EVENTS.append(f"{signal} | conf={confidence:.2f} | size=${size:,.0f}")
            AGENTS.clear()
            for i, agent in enumerate(
                sorted(evo.pop, key=lambda x: x.fitness, reverse=True)[:10]
            ):
                AGENTS.append({"id": i, "fitness": agent.fitness, "params": agent.params})

            if len(EVENTS) % 10 == 0:
                evo.evolve()
            time.sleep(2)
        except Exception as exc:
            EVENTS.append(f"ERR {exc}")
            time.sleep(2)


# ============================================================
# 9. WEB PORTAL
# ============================================================


app = FastAPI()


@app.get("/")
def dash() -> HTMLResponse:
    return HTMLResponse(
        """
<html><body style="background:#0e0e11;color:#eaeaea;font-family:monospace">
<h1>🧠 BTC AGENTIC ORACLE</h1>
<pre id=state></pre>
<pre id=agents></pre>
<pre id=events></pre>
<script>
const ws=new WebSocket("ws://localhost:8000/ws");
ws.onmessage=e=>{
let d=JSON.parse(e.data);
state.textContent=JSON.stringify(d.state,null,2);
agents.textContent=JSON.stringify(d.agents,null,2);
events.textContent=d.events.join("\\n");
}
</script>
</body></html>
"""
    )


@app.websocket("/ws")
async def ws(socket: WebSocket) -> None:
    await socket.accept()
    while True:
        await socket.send_text(
            json.dumps({"state": STATE, "agents": AGENTS, "events": EVENTS[-30:]})
        )
        await asyncio.sleep(1)


# ============================================================
# 10. BOOT
# ============================================================


if __name__ == "__main__":
    threading.Thread(target=oracle_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
