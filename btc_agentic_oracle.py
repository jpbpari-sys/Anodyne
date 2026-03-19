import asyncio
import json
import random
import sqlite3
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict

import aiohttp
import ccxt.async_support as ccxt
import numpy as np
import pywt
import sympy
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from scipy.signal import find_peaks

# ============================================================
# 1. CORE DATA STRUCTURES (FIRE-INFUSED)
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
class Agent:
    prime: int
    params: Dict[str, float]
    swarm: "RidgeSwarm"
    fitness: float = 0.0


# ============================================================
# 2. PORTFOLIO (FRACTIONAL KELLY — UNLEASHED)
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
# 3. POSITION SIMULATOR (FEES + SLIPPAGE + DRAW) — PLUGS TO REAL
# ============================================================


FEE = 0.0004
SLIPPAGE = 0.0005

POSITION = {
    "side": "FLAT",
    "entry": 0.0,
    "size": 0.0,
    "pnl": 0.0,
    "equity": 100_000.0,
    "max_drawdown": 0.0,
    "peak": 100_000.0,
}


# ============================================================
# 4. PLUG & PLAY: REAL TRADING VIA CCXT
# ============================================================


ENABLE_LIVE = False
BINANCE_API_KEY = "YOUR_API_KEY_HERE"
BINANCE_API_SECRET = "YOUR_API_SECRET_HERE"

exchange = (
    ccxt.binance(
        {
            "apiKey": BINANCE_API_KEY,
            "secret": BINANCE_API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        }
    )
    if ENABLE_LIVE
    else None
)


def simulate_trade(signal: str, price: float, size_usdt: float) -> str:
    if signal == "HOLD":
        return "HOLD"

    action = "HOLD"
    if POSITION["side"] == "FLAT":
        POSITION["side"] = signal
        POSITION["entry"] = price * (1 + SLIPPAGE if signal == "LONG" else 1 - SLIPPAGE)
        POSITION["size"] = size_usdt
        action = f"OPEN_{signal}"
    elif POSITION["side"] != signal:
        exit_price = price * (1 - SLIPPAGE if POSITION["side"] == "LONG" else 1 + SLIPPAGE)
        qty = POSITION["size"] / max(POSITION["entry"], 1e-9)
        gross = (exit_price - POSITION["entry"]) * qty
        if POSITION["side"] == "SHORT":
            gross = -gross
        fees = (POSITION["entry"] + exit_price) * qty * FEE
        pnl = gross - fees

        POSITION["pnl"] = pnl
        POSITION["equity"] += pnl
        POSITION["peak"] = max(POSITION["peak"], POSITION["equity"])
        dd = (POSITION["peak"] - POSITION["equity"]) / POSITION["peak"]
        POSITION["max_drawdown"] = max(POSITION["max_drawdown"], dd)

        log_trade(POSITION["side"], POSITION["entry"], exit_price, POSITION["size"], pnl, POSITION["equity"])

        POSITION["side"] = signal
        POSITION["entry"] = price * (1 + SLIPPAGE if signal == "LONG" else 1 - SLIPPAGE)
        POSITION["size"] = size_usdt
        action = f"FLIP_TO_{signal}"

    return action


async def execute_real_trade(signal: str, price: float, size_usdt: float) -> str:
    if not ENABLE_LIVE or not exchange:
        return "SIM"

    try:
        symbol = "BTCUSDT"
        side = "buy" if signal == "LONG" else "sell"
        amount = size_usdt / max(price, 1e-9)
        await exchange.set_leverage(10, symbol)
        order = await exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=amount,
            params={"positionSide": "BOTH"},
        )
        return f"LIVE {signal} @ {price:.0f} (order: {order['id']})"
    except Exception as e:
        print(f"LIVE TRADE FAIL: {e}")
        return "SIM_FALLBACK"


# ============================================================
# 5. SQLITE LOGGING (ETERNAL FLAME + AIRDROP)
# ============================================================


def init_db() -> None:
    conn = sqlite3.connect("firehorse.db")
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            ts REAL, side TEXT, entry REAL, exit REAL, size REAL, pnl REAL, equity REAL, live TEXT
        )
    """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            ts REAL, sharpe REAL, drawdown REAL, equity REAL, regime TEXT
        )
    """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS airdrops (
            ts REAL, amount REAL, claimer TEXT, reason TEXT, eth_address TEXT
        )
    """
    )
    conn.commit()
    conn.close()


def log_trade(
    side: str,
    entry: float,
    exit_price: float,
    size: float,
    pnl: float,
    equity: float,
    live_mode: str = "SIM",
) -> None:
    conn = sqlite3.connect("firehorse.db")
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO trades (ts, side, entry, exit, size, pnl, equity, live)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (time.time(), side, entry, exit_price, size, pnl, equity, live_mode),
    )
    conn.commit()
    conn.close()


def log_metrics(sharpe: float, regime: str) -> None:
    conn = sqlite3.connect("firehorse.db")
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO metrics (ts, sharpe, drawdown, equity, regime)
        VALUES (?, ?, ?, ?, ?)
    """,
        (time.time(), sharpe, POSITION["max_drawdown"], POSITION["equity"], regime),
    )
    conn.commit()
    conn.close()


def log_airdrop(
    amount: float,
    claimer: str = "Sovereign",
    reason: str = "Fire Horse Gallop",
    eth_address: str = "0x",
) -> None:
    conn = sqlite3.connect("firehorse.db")
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO airdrops (ts, amount, claimer, reason, eth_address)
        VALUES (?, ?, ?, ?, ?)
    """,
        (time.time(), amount, claimer, reason, eth_address),
    )
    conn.commit()
    conn.close()


# ============================================================
# 6. RETURNS TRACKER + SHARPE FITNESS
# ============================================================


RETURNS: list[float] = []


def update_sharpe(pnl: float) -> float:
    RETURNS.append(pnl)
    if len(RETURNS) > 100:
        RETURNS.pop(0)
    if len(RETURNS) > 10:
        r = np.array(RETURNS)
        return float(np.mean(r) / (np.std(r) + 1e-8))
    return 0.0


def fitness() -> float:
    if len(RETURNS) < 20:
        return 0.0
    r = np.array(RETURNS)
    sharpe = np.mean(r) / (np.std(r) + 1e-8)
    dd_penalty = POSITION["max_drawdown"] * 2
    return float(sharpe - dd_penalty)


# ============================================================
# 7. REGIME DETECTION (VOL-BASED)
# ============================================================


def detect_regime(signal: np.ndarray) -> str:
    vol = np.std(signal)
    if vol < 0.5:
        return "LOW_VOL"
    if vol < 1.2:
        return "MID_VOL"
    return "HIGH_VOL"


# ============================================================
# 8. AIRDROP SYSTEM
# ============================================================


AIRDROP_POOL = 1_000_000.0
AIRDROP_CLAIMED = 0.0
ETH_ADDRESS = "YOUR_ETH_ADDRESS_HERE"


def claim_airdrop(confidence: float = 0.0, regime: str = "MID_VOL") -> float:
    global AIRDROP_CLAIMED
    if AIRDROP_CLAIMED >= AIRDROP_POOL or not ETH_ADDRESS.startswith("0x"):
        return 0.0
    base = AIRDROP_POOL * 0.001
    multiplier = 1 + (confidence * 3) + (1.0 if regime == "LOW_VOL" else 0.5 if regime == "MID_VOL" else 0)
    amount = min(base * multiplier, AIRDROP_POOL - AIRDROP_CLAIMED)
    AIRDROP_CLAIMED += amount
    log_airdrop(amount, "Sovereign Holder", f"Eclipsed Sun Airdrop • {regime}", ETH_ADDRESS)
    return amount


# ============================================================
# 9. MARKET STREAM (ASYNC WS)
# ============================================================


PRICE_BUFFER_FAST: deque[float] = deque(maxlen=300)
PRICE_BUFFER_SLOW: deque[float] = deque(maxlen=600)


async def fetch_live_tick() -> None:
    url = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as ws:
            print("🟢 Connected to Binance WS — Fire Horse Rides")
            async for msg in ws:
                try:
                    data = json.loads(msg.data)
                    price = float(data["p"])
                    await process_tick(price)
                except Exception as e:
                    print(f"WS error: {e}")


async def process_tick(price: float) -> None:
    PRICE_BUFFER_FAST.append(price)
    PRICE_BUFFER_SLOW.append(price)
    if len(PRICE_BUFFER_FAST) < 50:
        return

    prices_fast = np.array(PRICE_BUFFER_FAST)
    prices_slow = np.array(PRICE_BUFFER_SLOW)

    returns_fast = np.diff(np.log(prices_fast))
    returns_slow = np.diff(np.log(prices_slow))

    returns_fast = (returns_fast - returns_fast.mean()) / (returns_fast.std() + 1e-8)
    returns_slow = (returns_slow - returns_slow.mean()) / (returns_slow.std() + 1e-8)

    await oracle_step(returns_fast, returns_slow, price)


# ============================================================
# 10. SPECTRAL ENGINE + RIDGE SWARM
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


class RidgeSwarm:
    def __init__(self, jf: float = 0.12, gap: int = 3, minlen: int = 6) -> None:
        self.ridges: list[list[SpectralPeak]] = []
        self.active: dict[int, int] = {}
        self.jf, self.gap, self.minlen = jf, gap, minlen

    def update(self, peaks: list[SpectralPeak]) -> None:
        for p in sorted(peaks, key=lambda x: x.time_idx):
            matched = False
            for i, t in list(self.active.items()):
                if p.time_idx - t <= self.gap:
                    lf = self.ridges[i][-1].frequency
                    if abs(p.frequency - lf) / max(lf, 1e-8) <= self.jf:
                        self.ridges[i].append(p)
                        self.active[i] = p.time_idx
                        matched = True
                        break
            if not matched:
                self.ridges.append([p])
                self.active[len(self.ridges) - 1] = p.time_idx

    def get(self) -> list[FrequencyRidge]:
        out = []
        for r in self.ridges:
            if len(r) >= self.minlen:
                out.append(
                    FrequencyRidge(
                        np.array([p.time_idx for p in r]),
                        np.array([p.frequency for p in r]),
                        np.array([p.magnitude for p in r]),
                    )
                )
        return out


# ============================================================
# 11. EVOLUTIONARY AGENTS
# ============================================================


class Evolution:
    def __init__(self, n: int = 40) -> None:
        self.pop: list[Agent] = []
        for _ in range(n):
            p = sympy.nextprime(random.randint(1_000_000, 9_000_000))
            params = {
                "jf": random.uniform(0.05, 0.2),
                "gap": random.randint(2, 5),
                "minlen": random.randint(4, 9),
            }
            self.pop.append(Agent(p, params, RidgeSwarm(**params)))
        self.gen = 0

    def evolve(self) -> None:
        self.pop.sort(key=lambda a: a.fitness, reverse=True)
        keep = self.pop[: len(self.pop) // 2]
        children = []
        for a in keep:
            p = sympy.nextprime(a.prime + random.randint(1, 1000))
            params = {k: v * (1 + random.uniform(-0.1, 0.1)) for k, v in a.params.items()}
            params["gap"] = int(params["gap"])
            params["minlen"] = int(params["minlen"])
            children.append(Agent(p, params, RidgeSwarm(**params)))
        self.pop = keep + children
        self.gen += 1


# ============================================================
# 12. SHARED STATE
# ============================================================


STATE = {
    "price": 0.0,
    "signal": "HOLD",
    "confidence": 0.0,
    "size": 0.0,
    "generation": 0,
    "fitness": 0.0,
    "equity": 100_000.0,
    "side": "FLAT",
    "pnl": 0.0,
    "drawdown": 0.0,
    "regime": "MID_VOL",
    "airdrop": 0.0,
    "live_mode": ENABLE_LIVE,
}
AGENTS: list[dict] = []
EVENTS: list[str] = []
LOCK = asyncio.Lock()


# ============================================================
# 13. ORACLE STEP
# ============================================================


LAST_SIGNAL_TIME = 0.0
COOLDOWN = 5


async def oracle_step(returns_fast: np.ndarray, returns_slow: np.ndarray, price: float) -> None:
    global LAST_SIGNAL_TIME
    global evo

    if "evo" not in globals():
        evo = Evolution()
        print("🔥🐎 Fire Horse Swarm Awakened")

    pf = PortfolioManager()

    peaks_fast = compute_peaks(returns_fast)
    peaks_slow = compute_peaks(returns_slow)

    for a in evo.pop:
        a.swarm.update(peaks_fast)
        a.swarm.update(peaks_slow)

    regime = detect_regime(returns_fast)

    best = max(evo.pop, key=lambda a: a.fitness)
    ridges = best.swarm.get()
    if ridges:
        r = max(
            ridges,
            key=lambda x: np.mean(x.magnitudes)
            * (len(x.times) / 10)
            / (len(returns_fast) - x.times[-1] + 1),
        )
        slope_fast, _ = np.polyfit(r.times, r.frequencies, 1)
        conf = min(float(np.mean(r.magnitudes) * 8), 1.0)

        if peaks_slow:
            slow_times = [p.time_idx for p in peaks_slow]
            slow_freqs = [p.frequency for p in peaks_slow]
            slope_slow, _ = np.polyfit(slow_times, slow_freqs, 1)
        else:
            slope_slow = 0.0

        if regime == "HIGH_VOL":
            conf *= 0.65
            sig = "HOLD" if abs(slope_fast) < 0.3 else ("LONG" if slope_fast > 0 and slope_slow > 0 else "SHORT")
        elif regime == "LOW_VOL":
            conf *= 1.2
            sig = "LONG" if slope_fast > 0 and slope_slow > 0 else "SHORT"
        else:
            sig = "LONG" if slope_fast > 0 and slope_slow > 0 else "SHORT"

        vol_factor = 1.3 if regime == "LOW_VOL" else 0.7 if regime == "HIGH_VOL" else 1.0
        size = pf.size(conf) * vol_factor

        now = time.time()
        if now - LAST_SIGNAL_TIME < COOLDOWN:
            action = "HOLD"
        else:
            if ENABLE_LIVE:
                action = await execute_real_trade(sig, price, size)
            else:
                action = simulate_trade(sig, price, size)
            if action != "HOLD":
                LAST_SIGNAL_TIME = now

        sharpe = 0.0
        airdrop = 0.0
        if "HOLD" not in action:
            sharpe = update_sharpe(POSITION["pnl"])
            log_metrics(sharpe, regime)
            airdrop = claim_airdrop(conf, regime)

        if POSITION["max_drawdown"] > 0.25:
            async with LOCK:
                STATE["signal"] = "PAUSED"
                EVENTS.append(
                    f"PAUSED | DD={POSITION['max_drawdown']:.2%} | Equity=${POSITION['equity']:,.0f} | Sharpe={sharpe:.2f} | regime={regime}"
                )
            await asyncio.sleep(60)
            return

        current_fitness = fitness()
        for a in evo.pop:
            a.fitness = current_fitness

        async with LOCK:
            STATE.update(
                {
                    "price": price,
                    "signal": sig,
                    "confidence": conf,
                    "size": size,
                    "generation": evo.gen,
                    "fitness": current_fitness,
                    "equity": POSITION["equity"],
                    "side": POSITION["side"],
                    "pnl": POSITION["pnl"],
                    "drawdown": POSITION["max_drawdown"],
                    "regime": regime,
                    "airdrop": airdrop,
                    "live_mode": ENABLE_LIVE,
                }
            )
            EVENTS.append(
                f"{action} {sig} | conf={conf:.2f} | size=${size:,.0f} | price=${price:,.0f} | equity=${POSITION['equity']:,.0f} | pnl=${POSITION['pnl']:,.0f} | fit={current_fitness:.2f} | {regime} | 💧{airdrop:.0f} FH to {ETH_ADDRESS}"
            )
            AGENTS.clear()
            for i, a in enumerate(sorted(evo.pop, key=lambda x: x.fitness, reverse=True)[:10]):
                AGENTS.append({"id": i, "fitness": a.fitness, "params": a.params})

        if len(EVENTS) % 10 == 0:
            evo.evolve()


# ============================================================
# 14. WEB APP + APIs
# ============================================================


app = FastAPI()


@app.get("/api/trades")
async def get_trades() -> dict:
    conn = sqlite3.connect("firehorse.db")
    c = conn.cursor()
    c.execute("SELECT ts, side, pnl, equity, live FROM trades ORDER BY ts DESC LIMIT 200")
    trades = [
        {"ts": row[0], "side": row[1], "pnl": row[2], "equity": row[3], "live": row[4]}
        for row in c.fetchall()
    ]
    conn.close()
    return {"trades": trades}


@app.get("/api/metrics")
async def get_metrics() -> dict:
    conn = sqlite3.connect("firehorse.db")
    c = conn.cursor()
    c.execute("SELECT ts, sharpe, drawdown, equity, regime FROM metrics ORDER BY ts DESC LIMIT 200")
    metrics = [
        {
            "ts": row[0],
            "sharpe": row[1],
            "drawdown": row[2],
            "equity": row[3],
            "regime": row[4],
        }
        for row in c.fetchall()
    ]
    conn.close()
    return {"metrics": metrics}


@app.get("/api/airdrops")
async def get_airdrops() -> dict:
    conn = sqlite3.connect("firehorse.db")
    c = conn.cursor()
    c.execute("SELECT ts, amount, reason, eth_address FROM airdrops ORDER BY ts DESC LIMIT 50")
    airdrops = [
        {"ts": row[0], "amount": row[1], "reason": row[2], "eth_address": row[3]}
        for row in c.fetchall()
    ]
    conn.close()
    return {"airdrops": airdrops}


@app.get("/")
def dash() -> HTMLResponse:
    return HTMLResponse(
        """
<html>
<head>
    <title>🔥🐎 THE FIRE HORSE ORACLE</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        body {
            background: linear-gradient(135deg, #1a0f00, #4a2c00, #ff4500);
            color: #ffd700;
            font-family: monospace;
            padding: 20px;
            background-image: url('https://picsum.photos/id/1015/1920/1080');
            background-size: cover;
            background-attachment: fixed;
            overflow-x: hidden;
        }
        h1 {
            color: #ff4500;
            text-shadow: 0 0 20px #ff4500, 0 0 40px #ffd700;
            animation: flame 2s infinite alternate;
        }
        @keyframes flame {
            0% { text-shadow: 0 0 10px #ff4500; }
            100% { text-shadow: 0 0 30px #ffd700, 0 0 60px #ff0000; }
        }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        canvas {
            background: rgba(0,0,0,0.7);
            border: 2px solid #ff4500;
            border-radius: 12px;
            box-shadow: 0 0 30px #ffd700;
        }
        pre {
            background: rgba(0,0,0,0.85);
            padding: 15px;
            border-radius: 12px;
            overflow: auto;
            max-height: 300px;
            border: 1px solid #ffd700;
        }
        .header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
        }
        .hero {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            border: 4px solid #ffd700;
            box-shadow: 0 0 40px #ff4500;
            background: url('https://picsum.photos/id/870/300/300') center/cover;
        }
        .airdrop {
            background: linear-gradient(45deg, #ffd700, #ff4500);
            color: #000;
            padding: 10px;
            border-radius: 8px;
            font-weight: bold;
            animation: airdropPulse 3s infinite;
        }
        @keyframes airdropPulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="hero"></div>
        <h1>🐎🔥 THE FIRE HORSE ORACLE v10</h1>
        <span style="font-size:12px; opacity:0.8;">PLUG & PLAY • Eclipsed Sun Airdrop • Year of the Fire Horse 2026</span>
    </div>
    <div style="text-align:center; margin-bottom:20px;">
        <div class="airdrop">💧 LIVE AIRDROP: <span id="airdropTotal">0</span> FH CLAIMED to <span id="ethAddr">0x...</span></div>
        <button onclick="toggleLive()" style="background:#ff4500;color:#000;padding:10px 20px;border:none;border-radius:8px;cursor:pointer;font-weight:bold;">🔌 TOGGLE LIVE TRADING</button>
    </div>
    <div class="grid">
        <div>
            <h2>📈 Equity Curve (Galloping Forward)</h2>
            <canvas id="equityChart" width="600" height="300"></canvas>
        </div>
        <div>
            <h2>📉 PNL Flames</h2>
            <canvas id="pnlChart" width="600" height="300"></canvas>
        </div>
        <div>
            <h2>📊 Drawdown Inferno</h2>
            <canvas id="ddChart" width="600" height="300"></canvas>
        </div>
        <div>
            <h2>🌋 Regime Firebreak</h2>
            <canvas id="regimeChart" width="600" height="300"></canvas>
        </div>
    </div>
    <pre id=state></pre>
    <pre id=agents></pre>
    <pre id=events></pre>
    <script>
        let equityChart, pnlChart, ddChart, regimeChart;
        let airdropTotal = 0;
        const ws = new WebSocket("ws://localhost:8000/ws");

        async function updateCharts() {
            const [tradesRes, metricsRes, airdropsRes] = await Promise.all([
                fetch('/api/trades'), fetch('/api/metrics'), fetch('/api/airdrops')
            ]);
            const trades = (await tradesRes.json()).trades.reverse();
            const metrics = (await metricsRes.json()).metrics.reverse();
            const airdrops = (await airdropsRes.json()).airdrops;

            if (trades.length === 0) return;

            airdropTotal = airdrops.reduce((sum, a) => sum + a.amount, 0);
            document.getElementById('airdropTotal').textContent = airdropTotal.toFixed(0);
            if (airdrops.length > 0) {
                document.getElementById('ethAddr').textContent = airdrops[0].eth_address.slice(0,6) + '...' + airdrops[0].eth_address.slice(-4);
            }

            const equityTimes = trades.map(t => new Date(t.ts * 1000).toLocaleTimeString());
            const equities = trades.map(t => t.equity);
            if (equityChart) equityChart.destroy();
            equityChart = new Chart(document.getElementById('equityChart'), {
                type: 'line',
                data: {
                    labels: equityTimes,
                    datasets: [{
                        label: 'Equity (Fire Path)',
                        data: equities,
                        borderColor: '#ff4500',
                        backgroundColor: 'rgba(255, 69, 0, 0.2)',
                        tension: 0.4,
                        borderWidth: 3
                    }]
                },
                options: {
                    responsive: true,
                    scales: { y: { beginAtZero: false, grid: { color: '#ffd70033' } } },
                    plugins: { legend: { labels: { color: '#ffd700' } } }
                }
            });

            const pnls = trades.map(t => t.pnl);
            const pnlLabels = trades.map((_, i) => `Trade ${i+1}`);
            if (pnlChart) pnlChart.destroy();
            pnlChart = new Chart(document.getElementById('pnlChart'), {
                type: 'bar',
                data: {
                    labels: pnlLabels,
                    datasets: [{
                        label: 'PNL (Flames)',
                        data: pnls,
                        backgroundColor: pnls.map(p => p > 0 ? '#00ffaa' : '#ff4444'),
                        borderColor: '#ffd700',
                        borderWidth: 1
                    }]
                },
                options: { responsive: true, plugins: { legend: { labels: { color: '#ffd700' } } } }
            });

            const ddTimes = metrics.map(m => new Date(m.ts * 1000).toLocaleTimeString());
            const dds = metrics.map(m => m.drawdown * 100);
            if (ddChart) ddChart.destroy();
            ddChart = new Chart(document.getElementById('ddChart'), {
                type: 'line',
                data: {
                    labels: ddTimes,
                    datasets: [{
                        label: 'Drawdown % (Inferno)',
                        data: dds,
                        borderColor: '#ffaa00',
                        backgroundColor: 'rgba(255, 170, 0, 0.2)',
                        tension: 0.4,
                        borderWidth: 3
                    }]
                },
                options: {
                    responsive: true,
                    scales: { y: { beginAtZero: true, max: 30, grid: { color: '#ffd70033' } } },
                    plugins: { legend: { labels: { color: '#ffd700' } } }
                }
            });

            const regimes = metrics.map(m => m.regime);
            const regimeCounts = {};
            regimes.forEach(r => regimeCounts[r] = (regimeCounts[r] || 0) + 1);
            if (regimeChart) regimeChart.destroy();
            regimeChart = new Chart(document.getElementById('regimeChart'), {
                type: 'pie',
                data: {
                    labels: Object.keys(regimeCounts),
                    datasets: [{
                        data: Object.values(regimeCounts),
                        backgroundColor: ['#00ffaa', '#ffff00', '#ff4444'],
                        borderColor: '#ffd700',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { labels: { color: '#ffd700', font: { size: 14 } } },
                        title: { display: true, text: 'Market Regimes — Fire Horse Pulse', color: '#ffd700' }
                    }
                }
            });
        }

        function toggleLive() {
            alert("🔌 Toggle ENABLE_LIVE in the code and restart. Your keys are plugged.");
        }

        ws.onmessage = e => {
            let d = JSON.parse(e.data);
            state.textContent = JSON.stringify(d.state, null, 2);
            agents.textContent = JSON.stringify(d.agents, null, 2);
            events.textContent = d.events.join("\\n");
            updateCharts();
        };

        setTimeout(updateCharts, 2000);
    </script>
</body>
</html>
"""
    )


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    while True:
        async with LOCK:
            payload = {"state": STATE, "agents": AGENTS, "events": EVENTS[-30:]}
        await websocket.send_text(json.dumps(payload))
        await asyncio.sleep(1)


if __name__ == "__main__":
    init_db()
    print("🔥🐎 Fire Horse DB Awakened • Eclipsed Sun Rises Tomorrow")
    print("🔌 HOW TO PLUG: Set ENABLE_LIVE=True + API keys in code. Then restart.")
    loop = asyncio.get_event_loop()
    loop.create_task(fetch_live_tick())
    uvicorn.run(app, host="0.0.0.0", port=8000)
