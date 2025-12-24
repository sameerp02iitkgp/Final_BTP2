import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # hard-disable CUDA on HF/Streamlit

import json
import time
import hmac
import hashlib
import tempfile
import copy
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import torch
import altair as alt

from io import StringIO
from pandas.tseries.offsets import BDay
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="TFT Stock Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# System-auto theme styling (no forced theme)
# =========================================================
st.markdown(
    """
<style>
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header {visibility: hidden;}
  [data-testid="stToolbar"] {visibility: hidden;}

  :root{
    --radius: 16px;
    --shadow: 0 10px 30px rgba(0,0,0,0.12);
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  }

  @media (prefers-color-scheme: dark) {
    :root{
      --bg: #0b0f14;
      --panel: rgba(255,255,255,0.04);
      --panel2: rgba(255,255,255,0.06);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.72);
      --border: rgba(255,255,255,0.10);
    }
  }

  @media (prefers-color-scheme: light) {
    :root{
      --bg: #ffffff;
      --panel: rgba(0,0,0,0.03);
      --panel2: rgba(0,0,0,0.05);
      --text: rgba(0,0,0,0.88);
      --muted: rgba(0,0,0,0.62);
      --border: rgba(0,0,0,0.10);
    }
  }

  .stApp { background: var(--bg); color: var(--text); }
  .block-container { padding-top: 1.25rem; padding-bottom: 2.25rem; max-width: 1200px; }

  .muted { color: var(--muted); }
  .mono { font-family: var(--mono); }
  .hr { height: 1px; background: var(--border); margin: 0.75rem 0 1.25rem 0; }

  .hero {
    border: 1px solid var(--border);
    background: linear-gradient(135deg, var(--panel), transparent);
    border-radius: calc(var(--radius) + 6px);
    padding: 18px 18px;
    box-shadow: var(--shadow);
  }
  .hero-title {
    font-size: 1.85rem;
    font-weight: 850;
    letter-spacing: -0.02em;
    line-height: 1.08;
    margin-bottom: 6px;
  }
  .hero-subtitle {
    font-size: 1.02rem;
    color: var(--muted);
    margin-bottom: 10px;
  }
  .chip {
    display: inline-block;
    border: 1px solid var(--border);
    background: var(--panel2);
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 0.85rem;
    color: var(--muted);
    margin-right: 8px;
    margin-top: 6px;
  }
  .kpi {
    border: 1px solid var(--border);
    background: var(--panel);
    border-radius: var(--radius);
    padding: 14px 14px;
    height: 100%;
  }
  .kpi-label { color: var(--muted); font-size: 0.92rem; margin-bottom: 6px; }
  .kpi-value { font-size: 1.45rem; font-weight: 800; letter-spacing: -0.01em; }
  .stButton>button { border-radius: 12px !important; padding: 0.6rem 0.9rem !important; border: 1px solid var(--border) !important; }
  [data-baseweb="input"] input, [data-baseweb="textarea"] textarea { border-radius: 12px !important; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# Files
# =========================================================
APP_DIR = Path(__file__).resolve().parent

def pick_path(*candidates: str) -> Path:
    tried = []
    for c in candidates:
        p = (APP_DIR / c).resolve()
        tried.append(str(p))
        if p.exists():
            return p
    raise FileNotFoundError("Missing required file. Tried:\n" + "\n".join(tried))

def pick_best_ckpt() -> Path:
    candidates = []
    for pat in ["*.ckpt", "artifacts/*.ckpt"]:
        candidates += list(APP_DIR.glob(pat))
    explicit = APP_DIR / "model.ckpt"
    if explicit.exists():
        candidates.append(explicit)

    uniq, seen = [], set()
    for p in candidates:
        if p.exists() and str(p) not in seen:
            uniq.append(p)
            seen.add(str(p))

    def score(p: Path) -> float:
        import re
        m = re.search(r"val_loss=([0-9.]+)", p.name)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
        return 1e9

    if uniq:
        return sorted(uniq, key=score)[0]
    return pick_path("model.ckpt", "artifacts/model.ckpt")

CKPT_PATH   = pick_best_ckpt()
PARAMS_PATH = pick_path("dataset_params.pth", "artifacts/dataset_params.pth")
CFG_PATH    = pick_path("config.json", "artifacts/config.json")

# =========================================================
# Password (HF Secrets → env vars)
# =========================================================
def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default) or default

def _pbkdf2_hex(password: str, salt: str, rounds: int = 200_000) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), rounds).hex()

def check_password() -> bool:
    if st.session_state.get("auth_ok", False):
        return True

    plain = _env("APP_PASSWORD", "")
    phash = _env("APP_PASSWORD_HASH", "")
    salt  = _env("APP_PASSWORD_SALT", "")

    if not plain and not phash:
        st.error("Owner setup missing: set APP_PASSWORD (or APP_PASSWORD_HASH + APP_PASSWORD_SALT) in Space Secrets, then restart.")
        st.stop()

    st.markdown(
        """
        <div class="hero">
          <div class="hero-title">🔒 Private TFT Stock Forecaster</div>
          <div class="hero-subtitle">
            I built this interactive TFT forecasting demo so you can explore probabilistic price forecasts (q10/q50/q90) privately.
          </div>
          <span class="chip">CPU inference</span>
          <span class="chip">Password protected</span>
          <span class="chip">System-auto theme</span>
        </div>
        <div class="hr"></div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("auth", clear_on_submit=False):
        guess = st.text_input("Password", type="password", placeholder="Enter password…")
        ok = st.form_submit_button("Unlock")

    if ok:
        good = hmac.compare_digest(_pbkdf2_hex(guess, salt), phash) if phash and salt else hmac.compare_digest(guess, plain)
        if good:
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    return False

if not check_password():
    st.stop()

with st.sidebar:
    st.markdown("### Session")
    if st.button("🔓 Logout", use_container_width=True):
        st.session_state["auth_ok"] = False
        st.rerun()
    st.caption("Password is stored in Space Secrets (environment variables).")

# =========================================================
# CPU-safe checkpoint loader
# =========================================================
@st.cache_resource
def load_tft_checkpoint_force_cpu(ckpt_path: str) -> TemporalFusionTransformer:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Unexpected checkpoint format (expected dict).")

    sd = ckpt.get("state_dict", None)
    if isinstance(sd, dict):
        for k, v in list(sd.items()):
            if torch.is_tensor(v):
                sd[k] = v.detach().cpu()

    def _force_metric_cpu(obj):
        if obj is None:
            return
        try:
            if hasattr(obj, "_device"):
                obj._device = torch.device("cpu")
        except Exception:
            pass
        try:
            if hasattr(obj, "modules"):
                for m in obj.modules():
                    if hasattr(m, "_device"):
                        try:
                            m._device = torch.device("cpu")
                        except Exception:
                            pass
        except Exception:
            pass

    hp = ckpt.get("hyper_parameters", {}) or {}
    if isinstance(hp, dict):
        _force_metric_cpu(hp.get("loss", None))
        _force_metric_cpu(hp.get("logging_metrics", None))

    tmp_name = "tft_cpu_" + hashlib.md5(str(ckpt_path).encode("utf-8")).hexdigest() + ".ckpt"
    tmp_ckpt = Path(tempfile.gettempdir()) / tmp_name
    torch.save(ckpt, tmp_ckpt)

    model = TemporalFusionTransformer.load_from_checkpoint(str(tmp_ckpt), map_location="cpu")
    _force_metric_cpu(getattr(model, "loss", None))
    _force_metric_cpu(getattr(model, "logging_metrics", None))
    model.to("cpu").eval()
    return model

# =========================================================
# Trained universe (39)
# =========================================================
TRAINED_39 = [
    "AAPL","ADBE","ADI","AMAT","AMD","AMZN","ASML","AVGO","CRM","CRWD","CSCO","DDOG","DELL",
    "GOOG","GOOGL","HPQ","IBM","INTC","KLAC","LRCX","MA","MDB","META","MRVL","MSFT","MU",
    "NET","NOW","NVDA","ORCL","PANW","PLTR","PYPL","QCOM","SNOW","TSLA","TXN","V","ZS"
]
COMPANY_MAP = {
    "AAPL":"Apple","ADBE":"Adobe","ADI":"Analog Devices","AMAT":"Applied Materials","AMD":"Advanced Micro Devices",
    "AMZN":"Amazon","ASML":"ASML Holding","AVGO":"Broadcom","CRM":"Salesforce","CRWD":"CrowdStrike","CSCO":"Cisco Systems",
    "DDOG":"Datadog","DELL":"Dell Technologies","GOOG":"Alphabet (Class C)","GOOGL":"Alphabet (Class A)","HPQ":"HP Inc.",
    "IBM":"IBM","INTC":"Intel","KLAC":"KLA","LRCX":"Lam Research","MA":"Mastercard","MDB":"MongoDB","META":"Meta Platforms",
    "MRVL":"Marvell Technology","MSFT":"Microsoft","MU":"Micron Technology","NET":"Cloudflare","NOW":"ServiceNow","NVDA":"NVIDIA",
    "ORCL":"Oracle","PANW":"Palo Alto Networks","PLTR":"Palantir","PYPL":"PayPal","QCOM":"Qualcomm","SNOW":"Snowflake","TSLA":"Tesla",
    "TXN":"Texas Instruments","V":"Visa","ZS":"Zscaler",
}

# =========================================================
# Load cfg/params/model
# =========================================================
@st.cache_resource
def load_cfg_params_model() -> Tuple[Dict[str, Any], Dict[str, Any], TemporalFusionTransformer, str, int, int, int]:
    cfg: Dict[str, Any] = {}
    try:
        cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            cfg = {}
    except Exception:
        cfg = {}

    params = torch.load(PARAMS_PATH, map_location="cpu")
    if not isinstance(params, dict):
        raise ValueError("dataset_params.pth did not load into a dict.")

    model = load_tft_checkpoint_force_cpu(str(CKPT_PATH))

    target = str(cfg.get("target", params.get("target", "log_ret")))
    max_enc = int(cfg.get("max_encoder_len", params.get("max_encoder_length", 60)))
    max_pred = int(cfg.get("max_pred_len", params.get("max_prediction_length", 5)))
    min_hist = int(cfg.get("min_history_rows", max(450, max_enc + 120)))

    return cfg, params, model, target, max_enc, max_pred, min_hist

CFG, DATASET_PARAMS, MODEL, TARGET, MAX_ENC, MAX_PRED, MIN_HIST = load_cfg_params_model()

TIME_IDX_COL = str(DATASET_PARAMS.get("time_idx", "time_idx"))
TARGET_COL   = str(DATASET_PARAMS.get("target", TARGET))
GROUP_COLS   = list(DATASET_PARAMS.get("group_ids", ["ticker"])) or ["ticker"]

def _is_return_target(name: str) -> bool:
    n = (name or "").lower()
    return ("log_ret" in n) or (n in {"log_ret", "log_return", "return", "ret"})


# =========================================================
# Data sources: yfinance (Yahoo) with Stooq fallback
# =========================================================
def _flatten_cols(cols) -> List[str]:
    out = []
    for c in cols:
        if isinstance(c, tuple):
            c = "_".join([str(x) for x in c if x and str(x) != ""])
        c = str(c).strip().lower().replace(" ", "_")
        out.append(c)
    return out

def _stooq_symbol(ticker: str) -> str:
    t = ticker.strip().lower().replace(".", "-")
    return f"{t}.us"

def _download_stooq_ohlcv(ticker: str, start: str) -> pd.DataFrame:
    sym = _stooq_symbol(ticker)
    url = "https://stooq.com/q/d/l/"
    r = requests.get(url, params={"s": sym, "i": "d"}, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    if r.status_code != 200 or not r.text or "Date" not in r.text:
        return pd.DataFrame()

    df = pd.read_csv(StringIO(r.text))
    if df is None or df.empty or "Date" not in df.columns:
        return pd.DataFrame()

    df = df.rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date","close"]).sort_values("date").reset_index(drop=True)
    df = df[df["date"] >= pd.to_datetime(start)].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    df["adj_close"] = np.nan
    df["px"] = df["close"].astype(float)
    return df[["date","open","high","low","close","adj_close","volume","px"]]

@st.cache_data(ttl=1800)
def download_ohlcv(ticker: str, start: str, cache_buster: int = 0) -> Tuple[pd.DataFrame, str]:
    _ = cache_buster
    t = str(ticker).strip().upper()

    # 1) Yahoo via yfinance
    try:
        raw = yf.download(t, start=start, auto_adjust=False, progress=False, threads=False, interval="1d")
        if raw is not None and len(raw) > 0:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = ["_".join([str(x) for x in tup if str(x) != ""]).strip() for tup in raw.columns.to_list()]
            raw = raw.reset_index()
            raw.columns = _flatten_cols(raw.columns)

            if "date" not in raw.columns and "datetime" in raw.columns:
                raw = raw.rename(columns={"datetime":"date"})
            if "date" in raw.columns:
                def pick(name: str) -> Optional[str]:
                    if name in raw.columns:
                        return name
                    for c in raw.columns:
                        if c.endswith("_" + name) or c == name:
                            return c
                    return None

                close_c = pick("close")
                adj_c   = pick("adj_close") or pick("adjclose")
                open_c  = pick("open")
                high_c  = pick("high")
                low_c   = pick("low")
                vol_c   = pick("volume")

                df = pd.DataFrame()
                df["date"] = pd.to_datetime(raw["date"], errors="coerce")
                df["open"] = pd.to_numeric(raw[open_c], errors="coerce") if open_c else np.nan
                df["high"] = pd.to_numeric(raw[high_c], errors="coerce") if high_c else np.nan
                df["low"]  = pd.to_numeric(raw[low_c], errors="coerce") if low_c else np.nan
                df["close"] = pd.to_numeric(raw[close_c], errors="coerce") if close_c else np.nan
                df["adj_close"] = pd.to_numeric(raw[adj_c], errors="coerce") if adj_c else np.nan
                df["volume"] = pd.to_numeric(raw[vol_c], errors="coerce") if vol_c else np.nan

                df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
                df["px"] = df["adj_close"]
                if df["px"].isna().all():
                    df["px"] = df["close"]
                df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["px"]).reset_index(drop=True)

                if not df.empty:
                    return df, "yahoo"
    except Exception:
        pass

    # 2) Stooq fallback
    df2 = _download_stooq_ohlcv(t, start=start)
    if not df2.empty:
        return df2, "stooq"

    raise ValueError(
        f"Data download failed for {t}. Yahoo returned no usable prices and Stooq had no data. "
        f"Try again later or keep Force refresh OFF."
    )

@st.cache_data(ttl=1800)
def download_market_context(start: str, cache_buster: int = 0) -> pd.DataFrame:
    """
    Context proxies:
      SPY -> SPY
      Nasdaq -> QQQ
      SOX -> SOXX
    Produces: spy_px_log_ret, nasdaq_px_log_ret, sox_px_log_ret (+ levels)
    """
    _ = cache_buster

    def fetch(sym: str) -> pd.DataFrame:
        df, _src = download_ohlcv(sym, start=start, cache_buster=cache_buster)
        out = df[["date","px"]].rename(columns={"px": f"{sym.lower()}_px"}).copy()
        out = out.dropna().sort_values("date").reset_index(drop=True)
        return out

    spy = fetch("SPY")
    qqq = fetch("QQQ")
    soxx = fetch("SOXX")

    ctx = spy.merge(qqq, on="date", how="outer").merge(soxx, on="date", how="outer")
    ctx = ctx.sort_values("date").ffill()

    ctx = ctx.rename(columns={"spy_px":"spy_px", "qqq_px":"nasdaq_px", "soxx_px":"sox_px"})
    for col in ["spy_px","nasdaq_px","sox_px"]:
        ctx[col] = pd.to_numeric(ctx[col], errors="coerce")
        ctx[f"{col}_log_ret"] = np.log(ctx[col]).diff().fillna(0.0)

    return ctx


# =========================================================
# Feature building (NOW includes missing indicators)
# =========================================================
def required_cols(params: dict) -> set:
    req = set()
    req.add(params.get("time_idx", "time_idx"))
    req.add(params.get("target", TARGET_COL))
    for k in [
        "group_ids",
        "static_categoricals", "static_reals",
        "time_varying_known_categoricals", "time_varying_known_reals",
        "time_varying_unknown_categoricals", "time_varying_unknown_reals",
    ]:
        for v in (params.get(k, []) or []):
            req.add(v)
    return req

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr

def add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # basic transforms
    out["log_px"] = np.log(out["px"].astype(float))
    out["log_ret"] = out["log_px"].diff().fillna(0.0)
    out["log_volume"] = np.log1p(out["volume"].fillna(0.0).clip(lower=0.0))

    dt = pd.to_datetime(out["date"])
    out["dow"] = dt.dt.dayofweek.astype(str)
    out["month"] = dt.dt.month.astype(str)
    out["year"] = dt.dt.year.astype(int)
    out["day"] = dt.dt.day.astype(int)

    # Bollinger bands (20,2) + bb_width
    win = 20
    mid = out["px"].astype(float).rolling(win, min_periods=win).mean()
    std = out["px"].astype(float).rolling(win, min_periods=win).std(ddof=0)
    out["bb_mid"] = mid
    out["bb_high"] = mid + 2.0 * std
    out["bb_low"] = mid - 2.0 * std
    # normalized width (safe + stable)
    out["bb_width"] = (out["bb_high"] - out["bb_low"]) / (out["bb_mid"].abs() + 1e-12)

    # EMA 20 / 50
    out["ema_20"] = _ema(out["px"].astype(float), 20)
    out["ema_50"] = _ema(out["px"].astype(float), 50)

    # RSI 14
    out["rsi_14"] = _rsi(out["px"].astype(float), 14)

    # MACD (12,26,9)
    ema12 = _ema(out["px"].astype(float), 12)
    ema26 = _ema(out["px"].astype(float), 26)
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_diff"] = out["macd"] - out["macd_signal"]

    # ATR 14
    # ensure OHLC exist; if missing, fill with px to avoid NaNs
    h = out["high"] if "high" in out.columns else out["px"]
    l = out["low"] if "low" in out.columns else out["px"]
    c = out["close"] if "close" in out.columns else out["px"]
    out["atr_14"] = _atr(pd.to_numeric(h, errors="coerce"),
                         pd.to_numeric(l, errors="coerce"),
                         pd.to_numeric(c, errors="coerce"),
                         14)

    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return out

def _ensure_group_time_target(out: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out = out.copy()
    for gc in GROUP_COLS:
        out[gc] = str(ticker).upper()
    if TIME_IDX_COL not in out.columns:
        out[TIME_IDX_COL] = np.arange(len(out), dtype=int)
    if TARGET_COL not in out.columns:
        out[TARGET_COL] = out["log_ret"].astype(float) if _is_return_target(TARGET_COL) else out["px"].astype(float)
    return out

def fill_missing_required(df: pd.DataFrame, needed: set) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    missing = [c for c in sorted(needed) if c not in out.columns]
    for c in missing:
        out[c] = 0.0

    cats = []
    for k in ["static_categoricals", "time_varying_known_categoricals", "time_varying_unknown_categoricals"]:
        cats += (DATASET_PARAMS.get(k, []) or [])
    for c in cats:
        if c in out.columns:
            out[c] = out[c].astype(str)

    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return out, missing

def make_future_rows(df: pd.DataFrame, horizon: int, ticker: str) -> pd.DataFrame:
    last = df.iloc[-1].copy()
    last_date = pd.to_datetime(last["date"])
    future_dates = pd.bdate_range(last_date + BDay(1), periods=horizon)

    future = pd.DataFrame([last.values] * horizon, columns=df.columns)
    future["date"] = future_dates

    dt = pd.to_datetime(future["date"])
    if "dow" in future.columns:
        future["dow"] = dt.dt.dayofweek.astype(str)
    if "month" in future.columns:
        future["month"] = dt.dt.month.astype(str)
    if "year" in future.columns:
        future["year"] = dt.dt.year.astype(int)
    if "day" in future.columns:
        future["day"] = dt.dt.day.astype(int)

    last_idx = int(last[TIME_IDX_COL])
    future[TIME_IDX_COL] = np.arange(last_idx + 1, last_idx + 1 + horizon, dtype=int)

    for gc in GROUP_COLS:
        future[gc] = str(ticker).upper()

    future[TARGET_COL] = 0.0
    return future.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

def to_q(pred, H: int) -> np.ndarray:
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    q = np.asarray(pred)
    if q.ndim == 2 and q.shape == (H, 3):
        q = q[None, :, :]
    if q.ndim == 2 and q.shape == (3, H):
        q = np.transpose(q, (1, 0))[None, :, :]
    if q.ndim == 3 and q.shape[1] == 3 and q.shape[2] == H:
        q = np.transpose(q, (0, 2, 1))
    if q.ndim != 3 or q.shape[2] != 3:
        raise ValueError(f"Unexpected quantile prediction shape: {q.shape}")
    return q

def patch_params_allow_unknown_categories(params: Dict[str, Any]) -> Dict[str, Any]:
    p = copy.deepcopy(params)
    ce = p.get("categorical_encoders", {}) or {}
    if isinstance(ce, dict):
        for _, enc in ce.items():
            try:
                if hasattr(enc, "add_nan"):
                    enc.add_nan = True
                if hasattr(enc, "_add_nan"):
                    enc._add_nan = True
            except Exception:
                pass
    p["categorical_encoders"] = ce
    return p

def forecast_ticker(ticker: str, horizon: int, years: int, bias_calibrate: bool, force_refresh: bool, allow_custom: bool):
    t = ticker.strip().upper()
    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    cache_buster = int(time.time()) if force_refresh else 0

    needed = required_cols(DATASET_PARAMS)

    base, source_used = download_ohlcv(t, start=start, cache_buster=cache_buster)
    if base is None or base.empty:
        raise ValueError(f"Data download returned no rows for {t}. Try again later.")

    base = add_core_features(base)

    ctx = download_market_context(start=start, cache_buster=cache_buster)
    if ctx is not None and not ctx.empty:
        base = base.merge(ctx, on="date", how="left").sort_values("date").ffill()

    base = _ensure_group_time_target(base, t)
    base, missing = fill_missing_required(base.reset_index(drop=True), needed)

    rows = len(base)
    need = max(MIN_HIST, MAX_ENC + 30)
    if rows < need:
        raise ValueError(f"Not enough history for {t}. Got {rows} rows — minimum {need} rows needed.")

    params = patch_params_allow_unknown_categories(DATASET_PARAMS) if allow_custom else DATASET_PARAMS

    bias = 0.0
    if bias_calibrate and len(base) > (MAX_ENC + 50):
        try:
            eval_ds = TimeSeriesDataSet.from_parameters(params, base, stop_randomization=True)
            eval_loader = eval_ds.to_dataloader(train=False, batch_size=256, num_workers=0)
            outp = MODEL.predict(eval_loader, mode="quantiles", mode_kwargs={"quantiles": [0.1, 0.5, 0.9]})
            q = to_q(outp, MAX_PRED)
            y = base[TARGET_COL].to_numpy()
            if len(y) >= len(q):
                bias = float(np.nanmean(q[:, 0, 1] - y[-len(q):]))
        except Exception:
            bias = 0.0

    future = make_future_rows(base, horizon=horizon, ticker=t)
    df_pred = pd.concat([base, future], ignore_index=True)

    pred_ds = TimeSeriesDataSet.from_parameters(params, df_pred, predict=True, stop_randomization=True)
    pred_loader = pred_ds.to_dataloader(train=False, batch_size=1, num_workers=0)

    pred = MODEL.predict(pred_loader, mode="quantiles", mode_kwargs={"quantiles": [0.1, 0.5, 0.9]})
    q = to_q(pred, horizon)
    q10, q50, q90 = q[0, :, 0] - bias, q[0, :, 1] - bias, q[0, :, 2] - bias

    last_px = float(base["px"].iloc[-1])

    if _is_return_target(TARGET_COL):
        def recon(ret_seq):
            return last_px * np.exp(np.cumsum(ret_seq))
        p10, p50, p90 = recon(q10), recon(q50), recon(q90)
    else:
        p10, p50, p90 = q10, q50, q90

    out_df = pd.DataFrame(
        {
            "date": pd.to_datetime(future["date"]).to_numpy(),
            "pred_px_q10": p10,
            "pred_px_q50": p50,
            "pred_px_q90": p90,
        }
    )
    return base, out_df, missing, source_used


# =========================================================
# Header
# =========================================================
st.markdown(
    f"""
<div class="hero">
  <div class="hero-title">📈 TFT Stock Forecaster</div>
  <div class="hero-subtitle">
    I built this quantile-aware Temporal Fusion Transformer (TFT) demo so you can explore forecasts interactively on a trained universe.
  </div>
  <span class="chip">Target: <span class="mono">{TARGET_COL}</span></span>
  <span class="chip">Encoder: <span class="mono">{MAX_ENC}</span></span>
  <span class="chip">PredLen: <span class="mono">{MAX_PRED}</span></span>
  <span class="chip">Checkpoint: <span class="mono">{CKPT_PATH.name}</span></span>
</div>
<div class="hr"></div>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["🔮 Forecast", "❓ FAQs", "ℹ️ About"])

# =========================================================
# Forecast tab
# =========================================================
with tab1:
    c1, c2, c3 = st.columns([1.6, 0.85, 0.85], gap="large")

    trained_rows = [{"ticker": t, "company": COMPANY_MAP.get(t, "")} for t in TRAINED_39]
    trained_df = pd.DataFrame(trained_rows)
    labels = [f"{r['ticker']} — {r['company']}" if r["company"] else r["ticker"] for r in trained_rows]
    label_to_ticker = {lab: lab.split(" — ")[0] for lab in labels}

    with c1:
        st.markdown("#### Ticker (trained universe)")
        q = st.text_input("Search trained tickers", value="", placeholder="Type e.g. NVDA / Microsoft / Zscaler …")
        ql = q.strip().lower()

        filtered_labels = labels
        if ql:
            filtered_labels = [lab for lab in labels if ql in lab.lower()]
            if not filtered_labels:
                filtered_labels = labels

        sel = st.selectbox("Pick a trained ticker", filtered_labels, index=0)
        trained_choice = label_to_ticker[sel]

        custom = st.text_input("Custom ticker (optional)", value="", placeholder="Try another ticker symbol…").strip().upper()
        ticker = custom if custom else trained_choice

        allow_custom = bool(custom) and (custom not in TRAINED_39)
        if allow_custom:
            st.warning(
                f"Custom ticker **{custom}** is not in the trained universe. "
                f"It is allowed if it has enough history, but reliability may drop."
            )

        with st.expander(f"View full trained universe ({len(TRAINED_39)})"):
            st.dataframe(trained_df, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("#### Forecast settings")
        horizon = int(st.selectbox("Horizon (trading days)", [1, 3, 5], index=2))
        years = int(st.selectbox("History window (years)", [5, 10, 15, 20], index=3))

    with c3:
        st.markdown("#### Reliability")
        bias_cal = st.checkbox(
            "Bias calibrate (best effort)",
            value=True,
            help="Estimates recent median drift (systematic over/under prediction) and subtracts it from the forecast."
        )
        force_refresh = st.checkbox(
            "Force refresh (live fetch)",
            value=False,
            help="Fetches fresh data immediately. Can be slower and may hit rate limits."
        )
        run = st.button("🚀 Forecast", use_container_width=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    if run:
        try:
            with st.spinner("Forecasting…"):
                base, out, missing, source_used = forecast_ticker(
                    ticker, horizon, years, bias_cal, force_refresh, allow_custom=allow_custom
                )

            st.markdown(
                f'<span class="chip">Data source: <span class="mono">{source_used}</span></span>',
                unsafe_allow_html=True
            )

            if missing:
                st.warning("Some required features were missing and were filled with 0 (forecast quality may drop).")
                with st.expander("See missing feature names"):
                    st.code(", ".join(missing))

            last_px = float(base["px"].iloc[-1])
            last_date = pd.to_datetime(base["date"].iloc[-1]).date().isoformat()
            p50_end = float(out["pred_px_q50"].iloc[-1])

            k1, k2, k3, k4 = st.columns(4, gap="large")
            k1.markdown(f'<div class="kpi"><div class="kpi-label">Last price</div><div class="kpi-value">{last_px:,.2f}</div><div class="muted">As of {last_date}</div></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="kpi"><div class="kpi-label">Median forecast (end)</div><div class="kpi-value">{p50_end:,.2f}</div><div class="muted">q50 @ +{horizon}d</div></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="kpi"><div class="kpi-label">Uncertainty</div><div class="kpi-value">q10–q90</div><div class="muted">80% band</div></div>', unsafe_allow_html=True)
            k4.markdown(f'<div class="kpi"><div class="kpi-label">Target</div><div class="kpi-value mono">{TARGET_COL}</div><div class="muted">model signal</div></div>', unsafe_allow_html=True)

            st.markdown("### Forecast table")
            st.dataframe(out, use_container_width=True)

            st.markdown("### Chart")
            hist = base[["date", "px"]].rename(columns={"px": "price"}).copy()
            fc = out.rename(columns={"pred_px_q50": "median"}).copy()
            band = out[["date", "pred_px_q10", "pred_px_q90"]].copy()

            chart_hist = alt.Chart(hist).mark_line().encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("price:Q", title="Price"),
                tooltip=["date:T", "price:Q"],
            )
            chart_band = alt.Chart(band).mark_area(opacity=0.22).encode(
                x="date:T",
                y="pred_px_q10:Q",
                y2="pred_px_q90:Q",
                tooltip=["date:T", "pred_px_q10:Q", "pred_px_q90:Q"],
            )
            chart_med = alt.Chart(fc).mark_line().encode(
                x="date:T",
                y="median:Q",
                tooltip=["date:T", "median:Q"],
            )

            st.altair_chart((chart_hist + chart_band + chart_med).interactive(), use_container_width=True)

            with st.expander("🔎 Show last 15 rows of input data"):
                st.dataframe(base.tail(15), use_container_width=True)

            st.caption("Educational / research demo — not financial advice.")

        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error("Forecast failed due to an unexpected error.")
            with st.expander("Show error details"):
                st.exception(e)


# =========================================================
# FAQs (FIRST FAQ = performance from your ipynb)
# =========================================================
with tab2:
    st.markdown("## FAQs")

    # ✅ FIRST FAQ: performance (exactly from your notebook output)
    with st.expander("How accurate is the model? (test performance summary)"):
        st.write(
            "I evaluated this TFT model on a held-out test split across **39 trained tickers**.\n\n"
            "**Global test metrics (N = 39):**\n"
            "- **H = 1 day:** RMSE 7.1895 | MAE 4.9166 | **MAPE 1.951%** | **Accuracy 98.049%** | DirAcc 48.72% | Cov80 76.92%\n"
            "- **H = 3 days:** RMSE 14.4184 | MAE 9.2617 | **MAPE 3.506%** | **Accuracy 96.494%** | DirAcc 74.36% | Cov80 92.31%\n"
            "- **H = 5 days:** RMSE 10.4689 | MAE 7.2878 | **MAPE 3.069%** | **Accuracy 96.931%** | DirAcc 53.85% | Cov80 94.87%\n\n"
            "These are aggregated over the test window and all trained tickers."
        )

    with st.expander("What do q10 / q50 / q90 mean?"):
        st.write(
            "- **q50**: the median forecast (most typical outcome)\n"
            "- **q10**: conservative scenario\n"
            "- **q90**: optimistic scenario\n\n"
            "The region between q10–q90 is an ~80% predictive uncertainty band."
        )

    with st.expander("What is “Bias calibrate (best effort)” and why would I use it?"):
        st.write(
            "This option estimates recent systematic drift (median error) and subtracts it from the next forecast. "
            "If the model is consistently over- or under-shooting recently, bias calibration can make the forecast more stable."
        )

    with st.expander("Why do some tickers fail with “Not enough history”?"):
        st.write(
            "The TFT uses a fixed-length **encoder window**. If a ticker is too new or has missing price history, "
            "the app refuses to forecast rather than produce low-quality output."
        )

    with st.expander("Can I use a ticker outside the trained universe?"):
        st.write(
            "Yes. You can type a custom ticker. The app will warn you because the model was trained on a fixed universe. "
            "If enough history exists, the forecast will run — but reliability may be lower."
        )

    with st.expander("What does “Force refresh (live fetch)” do? Should it be ON?"):
        st.write(
            "It controls **data downloading**, not the model.\n\n"
            "- **OFF (recommended default):** uses cached downloads → faster, fewer failures\n"
            "- **ON:** always downloads fresh data → newest data, but slower and can hit rate limits\n\n"
            "For a stable demo, I keep it OFF by default."
        )

    with st.expander("Which data source is used (Yahoo vs Stooq), and does it change the model?"):
        st.write(
            "The model stays the same. I fetch data from **Yahoo via yfinance** when available, "
            "and automatically fall back to **Stooq** if Yahoo returns no usable prices (common on hosted environments). "
            "You’ll see a small **“Data source”** chip after forecasting."
        )

    with st.expander("Is this financial advice?"):
        st.write(
            "No. This is an educational research demo showing how probabilistic forecasting can be deployed as an app. "
            "Use it to learn and explore, not to make trading decisions."
        )


# =========================================================
# About
# =========================================================
with tab3:
    st.markdown("## About this project")
    st.write(
        "I built this application as an **interactive, EX-grade research demo** using a **Temporal Fusion Transformer (TFT)**.\n\n"
        "**Pipeline:**\n"
        "1) Download historical OHLCV data (Yahoo when possible, Stooq fallback)\n"
        "2) Engineer technical + market-context features (RSI, MACD, ATR, Bollinger bands, SPY/QQQ/SOXX context)\n"
        "3) Run TFT inference on CPU\n"
        "4) Present quantile forecasts (q10/q50/q90) with uncertainty\n"
    )

    with st.expander("Model & runtime details"):
        st.write(
            f"- Checkpoint: `{CKPT_PATH.name}`\n"
            f"- Target: `{TARGET_COL}`\n"
            f"- Encoder length: `{MAX_ENC}`\n"
            f"- Prediction length (trained): `{MAX_PRED}`\n"
            f"- Minimum history rows: `{MIN_HIST}`\n"
            f"- Trained universe: `{len(TRAINED_39)}` tickers\n"
        )