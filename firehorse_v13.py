import numpy as np
import pywt
import time
import random
import json
import sympy
import asyncio
import aiohttp
import sqlite3

from scipy.ndimage import maximum_filter
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import deque

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn

try:
    import ccxt

    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("⚠️  ccxt not installed — live trading disabled")

# ============================================================
# CONFIG
# ============================================================

ENABLE_LIVE = False
BINANCE_API_KEY = "YOUR_API_KEY_HERE"
BINANCE_API_SECRET = "YOUR_API_SECRET_HERE"
ETH_ADDRESS = "YOUR_ETH_ADDRESS_HERE"

FEE = 0.0004
SLIPPAGE = 0.0005
CWT_RECOMPUTE_EVERY = 10
COOLDOWN = 5

# ============================================================
# DATA STRUCTURES
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
# PORTFOLIO — FRACTIONAL KELLY
# ============================================================


class PortfolioManager:
    def __init__(self, capital=100_000, frac_kelly=0.25):
        self.capital = capital
        self.win_rate = 0.55
        self.win_loss = 1.2
        self.frac_kelly = frac_kelly

    def size(self, confidence: float) -> float:
        p = self.win_rate * confidence + 0.5 * (1 - confidence)
        b = self.win_loss
        k = (b * p - (1 - p)) / b
        f = max(0.0, min(k * self.frac_kelly, 0.2))
        return self.capital * f


# ============================================================
# POSITION SIMULATOR
# ============================================================

POSITION = {
    "side": "FLAT",
    "entry": 0.0,
    "size": 0.0,
    "pnl": 0.0,
    "equity": 100_000.0,
    "max_drawdown": 0.0,
    "peak": 100_000.0,
}


def simulate_trade(signal: str, price: float, size_usdt: float) -> str:
    pos = POSITION
    if pos["side"] != "FLAT" and pos["side"] != signal:
        exit_p = price * (1 - SLIPPAGE if pos["side"] == "LONG" else 1 + SLIPPAGE)
        raw = (
            (exit_p - pos["entry"]) * pos["size"]
            if pos["side"] == "LONG"
            else (pos["entry"] - exit_p) * pos["size"]
        )
        net = raw - (FEE + SLIPPAGE) * pos["entry"] * pos["size"]
        pos["pnl"] = net
        pos["equity"] += net
        pos["peak"] = max(pos["peak"], pos["equity"])
        dd = (pos["peak"] - pos["equity"]) / pos["peak"]
        pos["max_drawdown"] = max(pos["max_drawdown"], dd)
        log_trade(pos["side"], pos["entry"], exit_p, pos["size"], net, pos["equity"])
        pos["side"] = "FLAT"
        pos["entry"] = 0.0
        pos["size"] = 0.0

    if signal in ("LONG", "SHORT") and pos["side"] == "FLAT":
        entry = price * (1 + SLIPPAGE if signal == "LONG" else 1 - SLIPPAGE)
        pos["side"] = signal
        pos["entry"] = entry
        pos["size"] = size_usdt / entry
        return f"OPEN {signal} @ {entry:.2f}"
    return "HOLD"


# ============================================================
# LIVE TRADING
# ============================================================

exchange = None
if ENABLE_LIVE and CCXT_AVAILABLE:
    exchange = ccxt.binance(
        {
            "apiKey": BINANCE_API_KEY,
            "secret": BINANCE_API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        }
    )


async def execute_real_trade(signal: str, price: float, size_usdt: float) -> str:
    if not ENABLE_LIVE or exchange is None:
        return simulate_trade(signal, price, size_usdt)
    try:
        side = "buy" if signal == "LONG" else "sell"
        amount = size_usdt / price
        await exchange.set_leverage(10, "BTCUSDT")
        order = await exchange.create_order(
            "BTCUSDT", "market", side, amount, params={"positionSide": "BOTH"}
        )
        return f"LIVE {signal} @ {price:.0f} (order: {order['id']})"
    except Exception as e:
        print(f"LIVE TRADE FAIL: {e}")
        return simulate_trade(signal, price, size_usdt)


# ============================================================
# SQLITE LOGGING
# ============================================================


def init_db():
    conn = sqlite3.connect("firehorse.db")
    c = conn.cursor()
    c.execute(
        "CREATE TABLE IF NOT EXISTS trades "
        "(ts REAL, side TEXT, entry REAL, exit REAL, size REAL, pnl REAL, equity REAL, live TEXT)"
    )
    c.execute(
        "CREATE TABLE IF NOT EXISTS metrics "
        "(ts REAL, sharpe REAL, drawdown REAL, equity REAL, regime TEXT)"
    )
    c.execute(
        "CREATE TABLE IF NOT EXISTS airdrops "
        "(ts REAL, amount REAL, claimer TEXT, reason TEXT, eth_address TEXT)"
    )
    conn.commit()
    conn.close()


def log_trade(side, entry, exit_p, size, pnl, equity, live="SIM"):
    conn = sqlite3.connect("firehorse.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?)",
        (time.time(), side, entry, exit_p, size, pnl, equity, live),
    )
    conn.commit()
    conn.close()


def log_metrics(sharpe, regime):
    conn = sqlite3.connect("firehorse.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO metrics VALUES (?,?,?,?,?)",
        (time.time(), sharpe, POSITION["max_drawdown"], POSITION["equity"], regime),
    )
    conn.commit()
    conn.close()


def log_airdrop(amount, claimer="Sovereign", reason="Fire Horse", eth="0x"):
    conn = sqlite3.connect("firehorse.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO airdrops VALUES (?,?,?,?,?)",
        (time.time(), amount, claimer, reason, eth),
    )
    conn.commit()
    conn.close()


# ============================================================
# RETURNS + FITNESS
# ============================================================

RETURNS: List[float] = []


def update_sharpe(pnl: float) -> float:
    RETURNS.append(pnl)
    if len(RETURNS) > 100:
        RETURNS.pop(0)
    if len(RETURNS) > 10:
        r = np.array(RETURNS)
        return float(np.mean(r) / (np.std(r) + 1e-8))
    return 0.0


def portfolio_fitness() -> float:
    if len(RETURNS) < 20:
        return 0.0
    r = np.array(RETURNS)
    return float(np.mean(r) / (np.std(r) + 1e-8) - POSITION["max_drawdown"] * 2)


# ============================================================
# REGIME
# ============================================================


def detect_regime(signal: np.ndarray) -> str:
    v = float(np.std(signal))
    return "LOW_VOL" if v < 0.5 else "HIGH_VOL" if v > 1.2 else "MID_VOL"


# ============================================================
# AIRDROP
# ============================================================

AIRDROP_POOL = 1_000_000.0
AIRDROP_CLAIMED = 0.0


def claim_airdrop(conf=0.0, regime="MID_VOL") -> float:
    global AIRDROP_CLAIMED
    if AIRDROP_CLAIMED >= AIRDROP_POOL or not ETH_ADDRESS.startswith("0x"):
        return 0.0
    mult = 1 + conf * 3 + (1.0 if regime == "LOW_VOL" else 0.5 if regime == "MID_VOL" else 0.0)
    amount = min(AIRDROP_POOL * 0.001 * mult, AIRDROP_POOL - AIRDROP_CLAIMED)
    AIRDROP_CLAIMED += amount
    log_airdrop(amount, "Sovereign", f"Fire Horse • {regime}", ETH_ADDRESS)
    return amount


# ============================================================
# SPECTRAL ENGINE
# ============================================================

CWT_WIDTHS = np.logspace(0, 6, 64, base=2)


def compute_peaks(signal: np.ndarray) -> List[SpectralPeak]:
    coeffs, freqs = pywt.cwt(signal, CWT_WIDTHS, "cmor1.5-1.0")
    power = np.abs(coeffs) ** 2
    thr = power.mean() + 2.0 * power.std()
    mask = (power == maximum_filter(power, size=(3, 5))) & (power > thr)
    si, ti = np.where(mask)
    return [
        SpectralPeak(int(t), float(freqs[s]), float(power[s, t]))
        for s, t in zip(si, ti)
    ]


# ============================================================
# RIDGE SWARM
# ============================================================


class RidgeSwarm:
    def __init__(self, jf=0.12, gap=3, minlen=6):
        self.jf = float(jf)
        self.gap = max(2, int(gap))
        self.minlen = max(4, int(minlen))
        self.ridges: List[List[SpectralPeak]] = []
        self.active: Dict[int, int] = {}

    def update(self, peaks: List[SpectralPeak]):
        for p in sorted(peaks, key=lambda x: x.time_idx):
            matched = False
            for i, lt in list(self.active.items()):
                if p.time_idx - lt > self.gap:
                    continue
                lf = self.ridges[i][-1].frequency
                if abs(p.frequency - lf) / max(lf, 1e-8) <= self.jf:
                    self.ridges[i].append(p)
                    self.active[i] = p.time_idx
                    matched = True
                    break
            if not matched:
                idx = len(self.ridges)
                self.ridges.append([p])
                self.active[idx] = p.time_idx

    def get(self) -> List[FrequencyRidge]:
        return [
            FrequencyRidge(
                np.array([p.time_idx for p in r]),
                np.array([p.frequency for p in r]),
                np.array([p.magnitude for p in r]),
            )
            for r in self.ridges
            if len(r) >= self.minlen
        ]

    def reset(self):
        if not self.active:
            return
        cutoff = max(self.active.values()) - self.gap * 10
        self.active = {i: t for i, t in self.active.items() if t > cutoff}


# ============================================================
# EVOLUTION
# ============================================================


class Evolution:
    def __init__(self, n=40):
        self.pop: List[Agent] = []
        for _ in range(n):
            pr = sympy.nextprime(random.randint(1_000_000, 9_000_000))
            params = {
                "jf": random.uniform(0.05, 0.2),
                "gap": random.randint(2, 5),
                "minlen": random.randint(4, 9),
            }
            self.pop.append(Agent(pr, params, RidgeSwarm(**params)))
        self.gen = 0

    def evolve(self):
        self.pop.sort(key=lambda a: a.fitness, reverse=True)
        keep = self.pop[: len(self.pop) // 2]
        children = []
        for a in keep:
            pr = sympy.nextprime(a.prime + random.randint(1, 1000))
            params = {k: v * (1 + random.uniform(-0.1, 0.1)) for k, v in a.params.items()}
            params["gap"] = max(2, int(params["gap"]))
            params["minlen"] = max(4, int(params["minlen"]))
            children.append(Agent(pr, params, RidgeSwarm(**params)))
        self.pop = keep + children
        self.gen += 1


# ============================================================
# SHARED STATE
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
AGENTS: List[Dict] = []
EVENTS: deque = deque(maxlen=200)
LOCK = asyncio.Lock()

PRICE_BUFFER_FAST: deque = deque(maxlen=300)
PRICE_BUFFER_SLOW: deque = deque(maxlen=600)
_tick_count = 0
_last_pf: List[SpectralPeak] = []
_last_ps: List[SpectralPeak] = []
evo: Optional[Evolution] = None
LAST_SIGNAL_TIME = 0.0


# ============================================================
# ORACLE STEP
# ============================================================


async def oracle_step(ret_fast: np.ndarray, ret_slow: np.ndarray, price: float):
    global evo, _tick_count, _last_pf, _last_ps, LAST_SIGNAL_TIME

    if evo is None:
        evo = Evolution()
        print("🔥🐎 Swarm Awakened")

    pf = PortfolioManager()
    regime = detect_regime(ret_fast)
    _tick_count += 1

    if _tick_count % CWT_RECOMPUTE_EVERY == 0:
        _last_pf = compute_peaks(ret_fast)
        _last_ps = compute_peaks(ret_slow)

    for a in evo.pop:
        a.swarm.update(_last_pf)
        a.swarm.update(_last_ps)
        if _tick_count % 50 == 0:
            a.swarm.reset()

    best = max(evo.pop, key=lambda a: a.fitness)
    ridges = best.swarm.get()
    sig = "HOLD"
    conf = 0.0
    size = 0.0

    if ridges:

        def score(r: FrequencyRidge) -> float:
            return float(np.mean(r.magnitudes) * len(r.times) / 10 / (len(ret_fast) - r.times[-1] + 1))

        r = max(ridges, key=score)
        slope_f = float(np.polyfit(r.times, r.frequencies, 1)[0]) if len(r.times) >= 2 else 0.0
        conf = float(min(np.mean(r.magnitudes) * 8, 1.0))

        slow_ridges = [
            FrequencyRidge(
                np.array([p.time_idx for p in rd]),
                np.array([p.frequency for p in rd]),
                np.array([p.magnitude for p in rd]),
            )
            for rd in best.swarm.ridges
            if len(rd) >= best.swarm.minlen
        ]

        slope_s = 0.0
        if slow_ridges:
            sr = max(slow_ridges, key=score)
            if len(sr.times) >= 2:
                slope_s = float(np.polyfit(sr.times, sr.frequencies, 1)[0])

        if regime == "HIGH_VOL":
            conf *= 0.65
            sig = "HOLD" if abs(slope_f) < 0.3 else ("LONG" if slope_f > 0 and slope_s > 0 else "SHORT")
        elif regime == "LOW_VOL":
            conf = min(conf * 1.2, 1.0)
            sig = "LONG" if slope_f > 0 and slope_s > 0 else "SHORT"
        else:
            sig = "LONG" if slope_f > 0 and slope_s > 0 else "SHORT"

        vf = 1.3 if regime == "LOW_VOL" else 0.7 if regime == "HIGH_VOL" else 1.0
        size = pf.size(conf) * vf

    now = time.time()
    action = "HOLD"
    if sig != "HOLD" and (now - LAST_SIGNAL_TIME) >= COOLDOWN:
        action = await execute_real_trade(sig, price, size)
        LAST_SIGNAL_TIME = now

    sharpe = 0.0
    airdrop = 0.0
    if action != "HOLD":
        sharpe = update_sharpe(POSITION["pnl"])
        log_metrics(sharpe, regime)
        airdrop = claim_airdrop(conf, regime)

    if POSITION["max_drawdown"] > 0.25:
        async with LOCK:
            STATE["signal"] = "PAUSED"
            EVENTS.append(f"⛔ PAUSED | DD={POSITION['max_drawdown']:.2%} | {regime}")
        await asyncio.sleep(60)
        return

    current_fitness = portfolio_fitness()
    for a in evo.pop:
        a.fitness = current_fitness
    if len(EVENTS) % 10 == 0 and len(EVENTS) > 0:
        evo.evolve()

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
            f"{action} {sig} | conf={conf:.2f} | ${size:,.0f} | "
            f"${price:,.0f} | eq=${POSITION['equity']:,.0f} | "
            f"pnl=${POSITION['pnl']:,.0f} | {regime} | 💧{airdrop:.0f}FH"
        )
        AGENTS.clear()
        for i, a in enumerate(sorted(evo.pop, key=lambda x: x.fitness, reverse=True)[:10]):
            AGENTS.append({"id": i, "fitness": round(a.fitness, 4), "params": a.params})


# ============================================================
# MARKET STREAM
# ============================================================


async def fetch_live_tick():
    url = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(url) as ws:
                    print("🟢 Binance WS connected")
                    async for msg in ws:
                        try:
                            await process_tick(float(json.loads(msg.data)["p"]))
                        except Exception as e:
                            print(f"Tick err: {e}")
        except Exception as e:
            print(f"WS drop: {e} — retry in 5s")
            await asyncio.sleep(5)


async def process_tick(price: float):
    PRICE_BUFFER_FAST.append(price)
    PRICE_BUFFER_SLOW.append(price)
    if len(PRICE_BUFFER_FAST) < 50:
        return
    pf = np.diff(np.log(np.array(PRICE_BUFFER_FAST)))
    ps = np.diff(np.log(np.array(PRICE_BUFFER_SLOW)))
    pf = (pf - pf.mean()) / (pf.std() + 1e-8)
    ps = (ps - ps.mean()) / (ps.std() + 1e-8)
    await oracle_step(pf, ps, price)


# ============================================================
# FASTAPI + DASHBOARD
# ============================================================

app = FastAPI()

# The MiniChart JS renderer — real vanilla canvas implementation (~5KB).
# No CDN, no fake bundle. Draws line, bar, and pie charts natively.
MINICHART_JS = r"""
const MiniChart = {
  _pad: {t:24,r:12,b:32,l:56},

  _clear(ctx, w, h) {
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillRect(0, 0, w, h);
  },

  _axes(ctx, w, h, minV, maxV, labels) {
    const p=this._pad, cw=w-p.l-p.r, ch=h-p.t-p.b;
    ctx.strokeStyle='#ffffff22'; ctx.lineWidth=1;
    for (let i=0;i<=4;i++) {
      const v=minV+(maxV-minV)*i/4, y=p.t+ch-ch*i/4;
      ctx.strokeStyle='#ffffff18'; ctx.beginPath();
      ctx.moveTo(p.l,y); ctx.lineTo(p.l+cw,y); ctx.stroke();
      ctx.fillStyle='#ffd70099'; ctx.font='10px monospace';
      ctx.textAlign='right'; ctx.fillText(
        Math.abs(v)>9999?(v/1000).toFixed(1)+'k':v.toFixed(v%1?2:0),
        p.l-4, y+4);
    }
    ctx.strokeStyle='#ffffff33'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(p.l,p.t); ctx.lineTo(p.l,p.t+ch); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(p.l,p.t+ch); ctx.lineTo(p.l+cw,p.t+ch); ctx.stroke();
    if (labels && labels.length) {
      const step=Math.max(1,Math.floor(labels.length/6));
      ctx.fillStyle='#ffd70066'; ctx.textAlign='center';
      for (let i=0;i<labels.length;i+=step) {
        const x=p.l+cw*i/(labels.length-1||1);
        ctx.fillText(labels[i], x, p.t+ch+18);
      }
    }
    return {cw, ch, p};
  },

  line(canvas, labels, data, color='#ff4500', title='') {
    const ctx=canvas.getContext('2d'), w=canvas.width, h=canvas.height;
    this._clear(ctx,w,h);
    if (!data.length) { ctx.fillStyle=color; ctx.font='11px monospace';
      ctx.textAlign='center'; ctx.fillText('Waiting for data…',w/2,h/2); return; }
    const mn=Math.min(...data), mx=Math.max(...data), rng=mx-mn||1;
    const {cw,ch,p}=this._axes(ctx,w,h,mn,mx,labels);
    const grad=ctx.createLinearGradient(0,p.t,0,p.t+ch);
    grad.addColorStop(0,color+'55'); grad.addColorStop(1,color+'00');
    ctx.beginPath();
    data.forEach((v,i)=>{
      const x=p.l+cw*i/(data.length-1||1), y=p.t+ch-ch*(v-mn)/rng;
      i?ctx.lineTo(x,y):ctx.moveTo(x,y);
    });
    const last=data.length-1;
    ctx.lineTo(p.l+cw*last/(last||1), p.t+ch); ctx.lineTo(p.l,p.t+ch);
    ctx.closePath(); ctx.fillStyle=grad; ctx.fill();
    ctx.beginPath();
    data.forEach((v,i)=>{
      const x=p.l+cw*i/(data.length-1||1), y=p.t+ch-ch*(v-mn)/rng;
      i?ctx.lineTo(x,y):ctx.moveTo(x,y);
    });
    ctx.strokeStyle=color; ctx.lineWidth=2; ctx.stroke();
    ctx.fillStyle=color; ctx.font='bold 11px monospace'; ctx.textAlign='left';
    ctx.fillText(title, p.l+4, p.t+14);
  },

  bar(canvas, labels, data, colors, title='') {
    const ctx=canvas.getContext('2d'), w=canvas.width, h=canvas.height;
    this._clear(ctx,w,h);
    if (!data.length) return;
    const mn=Math.min(0,...data), mx=Math.max(0,...data), rng=mx-mn||1;
    const {cw,ch,p}=this._axes(ctx,w,h,mn,mx,labels);
    const bw=Math.max(1,cw/data.length-2);
    const zero=p.t+ch-ch*(0-mn)/rng;
    data.forEach((v,i)=>{
      const x=p.l+cw*i/data.length+1, y=p.t+ch-ch*(v-mn)/rng;
      ctx.fillStyle=colors?colors[i]:'#00ffaa88';
      ctx.fillRect(x, v>=0?y:zero, bw, Math.abs(zero-y));
    });
    ctx.fillStyle='#ffd700aa'; ctx.font='bold 11px monospace'; ctx.textAlign='left';
    ctx.fillText(title, p.l+4, p.t+14);
  },

  pie(canvas, labels, data, colors) {
    const ctx=canvas.getContext('2d'), w=canvas.width, h=canvas.height;
    this._clear(ctx,w,h);
    if (!data.length) return;
    const total=data.reduce((a,b)=>a+b,0)||1;
    const cx=w*0.42, cy=h/2, r=Math.min(cx,cy)-16;
    let ang=-Math.PI/2;
    data.forEach((v,i)=>{
      const sl=v/total*Math.PI*2;
      ctx.beginPath(); ctx.moveTo(cx,cy);
      ctx.arc(cx,cy,r,ang,ang+sl); ctx.closePath();
      ctx.fillStyle=colors[i%colors.length]; ctx.fill();
      ctx.strokeStyle='#111'; ctx.lineWidth=2; ctx.stroke();
      ang+=sl;
    });
    ctx.font='11px monospace'; ctx.textAlign='left';
    labels.forEach((lbl,i)=>{
      ctx.fillStyle=colors[i%colors.length];
      ctx.fillRect(w-90,14+i*20,14,14);
      ctx.fillStyle='#ffd700'; ctx.fillText(lbl,w-72,26+i*20);
    });
  }
};
"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>🔥🐎 FIRE HORSE ORACLE v13</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{background:linear-gradient(135deg,#0a0500,#1a0800,#2a1000);
         color:#ffd700;font-family:'Courier New',monospace;padding:20px;min-height:100vh}
    h1{color:#ff4500;text-shadow:0 0 20px #ff4500;
       animation:flame 2s infinite alternate;font-size:2rem;margin-bottom:4px}
    @keyframes flame{0%{text-shadow:0 0 10px #ff4500}100%{text-shadow:0 0 30px #ffd700,0 0 60px #ff0000}}
    .header{display:flex;align-items:center;gap:20px;margin-bottom:20px;flex-wrap:wrap}
    .badge{background:linear-gradient(45deg,#ff4500,#ffd700);color:#000;
           padding:6px 14px;border-radius:20px;font-weight:bold;font-size:12px}
    .stats-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
               gap:10px;margin-bottom:20px}
    .card{background:rgba(255,69,0,.1);border:1px solid #ff4500;
          border-radius:10px;padding:10px;text-align:center}
    .card .lbl{font-size:10px;opacity:.7;margin-bottom:3px}
    .card .val{font-size:1.3rem;font-weight:bold;color:#ffd700}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:20px}
    .chart-box{background:rgba(0,0,0,.6);border:1px solid #ff450055;
               border-radius:12px;padding:12px}
    .chart-box h3{font-size:12px;margin-bottom:6px;color:#ffaa44}
    canvas{width:100%!important;display:block}
    .log{background:rgba(0,0,0,.8);border:1px solid #ffd70033;border-radius:10px;
         padding:12px;font-size:10px;max-height:200px;overflow-y:auto;line-height:1.7}
    .log-entry{border-bottom:1px solid #ffffff11;padding:1px 0}
    .LONG{color:#00ffaa}.SHORT{color:#ff4444}.HOLD{color:#888}
    .airdrop{background:linear-gradient(90deg,#ff4500,#ffd700);color:#000;
             padding:8px 16px;border-radius:8px;font-weight:bold;
             text-align:center;margin-bottom:14px;animation:pulse 3s infinite}
    @keyframes pulse{0%,100%{opacity:1}50%{opacity:.8}}
    .LONG .val{color:#00ffaa}.SHORT .val{color:#ff4444}
  </style>
</head>
<body>
<div class="header">
  <h1>🐎🔥 FIRE HORSE ORACLE <small style="font-size:.5em">v13</small></h1>
  <span class="badge">PAPER TRADING</span>
  <span class="badge" id="regimeBadge">MID_VOL</span>
  <span class="badge" id="genBadge">GEN 0</span>
</div>
<div class="airdrop">💧 AIRDROP: <span id="airdropTotal">0</span> FH claimed</div>
<div class="stats-row">
  <div class="card"><div class="lbl">PRICE</div><div class="val" id="sPrice">$0</div></div>
  <div class="card"><div class="lbl">SIGNAL</div><div class="val" id="sSignal">HOLD</div></div>
  <div class="card"><div class="lbl">CONFIDENCE</div><div class="val" id="sConf">0%</div></div>
  <div class="card"><div class="lbl">EQUITY</div><div class="val" id="sEquity">$100k</div></div>
  <div class="card"><div class="lbl">PNL</div><div class="val" id="sPnl">$0</div></div>
  <div class="card"><div class="lbl">DRAWDOWN</div><div class="val" id="sDd">0%</div></div>
  <div class="card"><div class="lbl">FITNESS</div><div class="val" id="sFit">0</div></div>
  <div class="card"><div class="lbl">POSITION</div><div class="val" id="sSide">FLAT</div></div>
</div>
<div class="grid">
  <div class="chart-box"><h3>📈 Equity Curve</h3>
    <canvas id="cEquity" height="150"></canvas></div>
  <div class="chart-box"><h3>💰 PNL per Trade</h3>
    <canvas id="cPnl" height="150"></canvas></div>
  <div class="chart-box"><h3>📉 Drawdown %</h3>
    <canvas id="cDd" height="150"></canvas></div>
  <div class="chart-box"><h3>🌋 Regime Distribution</h3>
    <canvas id="cRegime" height="150"></canvas></div>
</div>
<div class="log" id="eventLog"></div>

<script>
__MINICHART__

// Chart data buffers
const bufs = {equity:{l:[],d:[]}, pnl:{l:[],d:[],c:[]}, dd:{l:[],d:[]},
              regime:{l:[],d:[]}};
const MAX = 200;
let airdropTotal = 0;

function szCanvas(id) {
  const el = document.getElementById(id);
  el.width  = el.parentElement.clientWidth - 24;
  el.height = 150;
  return el;
}

function redraw() {
  const eq = szCanvas('cEquity');
  MiniChart.line(eq, bufs.equity.l, bufs.equity.d, '#ff4500', 'Equity ($)');

  const pn = szCanvas('cPnl');
  MiniChart.bar(pn, bufs.pnl.l, bufs.pnl.d, bufs.pnl.c, 'PNL per trade');

  const dd = szCanvas('cDd');
  MiniChart.line(dd, bufs.dd.l, bufs.dd.d, '#ffaa00', 'Drawdown %');

  const rg = szCanvas('cRegime');
  MiniChart.pie(rg, bufs.regime.l, bufs.regime.d,
                ['#00ffaa','#ffff00','#ff4444','#88aaff']);
}

async function fetchAndLoad() {
  try {
    const [tR,mR,aR] = await Promise.all([
      fetch('/api/trades'), fetch('/api/metrics'), fetch('/api/airdrops')]);
    const {trades}  = await tR.json();
    const {metrics} = await mR.json();
    const {airdrops}= await aR.json();

    trades.reverse(); metrics.reverse();
    airdropTotal = airdrops.reduce((s,a)=>s+a.amount,0);
    document.getElementById('airdropTotal').textContent = airdropTotal.toFixed(0);

    bufs.equity.l = trades.map(t=>new Date(t.ts*1000).toLocaleTimeString());
    bufs.equity.d = trades.map(t=>t.equity);
    bufs.pnl.l    = trades.map((_,i)=>`T${i+1}`);
    bufs.pnl.d    = trades.map(t=>t.pnl);
    bufs.pnl.c    = trades.map(t=>t.pnl>=0?'#00ffaa88':'#ff444488');
    bufs.dd.l     = metrics.map(m=>new Date(m.ts*1000).toLocaleTimeString());
    bufs.dd.d     = metrics.map(m=>+(m.drawdown*100).toFixed(2));

    const rc = {};
    metrics.forEach(m=>rc[m.regime]=(rc[m.regime]||0)+1);
    bufs.regime.l = Object.keys(rc);
    bufs.regime.d = Object.values(rc);
    redraw();
  } catch(e) { console.warn('fetch err',e); }
}

const ws = new WebSocket('ws://' + location.host + '/ws');
ws.onmessage = e => {
  const {state, agents, events} = JSON.parse(e.data);
  document.getElementById('sPrice').textContent    = '$'+state.price.toLocaleString(undefined,{maximumFractionDigits:0});
  const sigEl = document.getElementById('sSignal');
  sigEl.textContent = state.signal; sigEl.className = 'val ' + state.signal;
  document.getElementById('sConf').textContent     = (state.confidence*100).toFixed(1)+'%';
  document.getElementById('sEquity').textContent   = '$'+(state.equity/1000).toFixed(1)+'k';
  document.getElementById('sPnl').textContent      = '$'+state.pnl.toLocaleString(undefined,{maximumFractionDigits:0});
  document.getElementById('sDd').textContent       = (state.drawdown*100).toFixed(2)+'%';
  document.getElementById('sFit').textContent      = state.fitness.toFixed(3);
  document.getElementById('sSide').textContent     = state.side;
  document.getElementById('regimeBadge').textContent = state.regime;
  document.getElementById('genBadge').textContent  = 'GEN '+state.generation;
  const log = document.getElementById('eventLog');
  log.innerHTML = events.slice().reverse().map(ev=>{
    const cls=ev.includes('LONG')?'LONG':ev.includes('SHORT')?'SHORT':'HOLD';
    return `<div class="log-entry ${cls}">${ev}</div>`;
  }).join('');
  fetchAndLoad();
};
ws.onerror = ()=>console.warn('WS error');
window.addEventListener('resize', redraw);
setTimeout(fetchAndLoad, 1000);
</script>
</body>
</html>""".replace("__MINICHART__", MINICHART_JS)


@app.get("/")
def dash():
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/api/trades")
async def get_trades():
    conn = sqlite3.connect("firehorse.db")
    c = conn.cursor()
    c.execute("SELECT ts,side,pnl,equity,live FROM trades ORDER BY ts DESC LIMIT 200")
    rows = [
        {"ts": r[0], "side": r[1], "pnl": r[2], "equity": r[3], "live": r[4]}
        for r in c.fetchall()
    ]
    conn.close()
    return {"trades": rows}


@app.get("/api/metrics")
async def get_metrics():
    conn = sqlite3.connect("firehorse.db")
    c = conn.cursor()
    c.execute("SELECT ts,sharpe,drawdown,equity,regime FROM metrics ORDER BY ts DESC LIMIT 200")
    rows = [
        {
            "ts": r[0],
            "sharpe": r[1],
            "drawdown": r[2],
            "equity": r[3],
            "regime": r[4],
        }
        for r in c.fetchall()
    ]
    conn.close()
    return {"metrics": rows}


@app.get("/api/airdrops")
async def get_airdrops():
    conn = sqlite3.connect("firehorse.db")
    c = conn.cursor()
    c.execute("SELECT ts,amount,reason,eth_address FROM airdrops ORDER BY ts DESC LIMIT 50")
    rows = [
        {"ts": r[0], "amount": r[1], "reason": r[2], "eth_address": r[3]}
        for r in c.fetchall()
    ]
    conn.close()
    return {"airdrops": rows}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        async with LOCK:
            payload = {"state": STATE, "agents": AGENTS, "events": list(EVENTS)[-30:]}
        await ws.send_text(json.dumps(payload))
        await asyncio.sleep(1)


# ============================================================
# BOOT
# ============================================================


async def main():
    init_db()
    print("🔥🐎 Fire Horse v13 — DB ready")
    print("📡 Connecting to Binance WS (60 ticks warmup needed)")
    print("🌐 Dashboard → http://localhost:8000")
    print("📊 Charts: self-contained MiniChart renderer — no CDN required")
    print("🔌 Live trading: DISABLED")
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    await asyncio.gather(fetch_live_tick(), server.serve())


if __name__ == "__main__":
    asyncio.run(main())
