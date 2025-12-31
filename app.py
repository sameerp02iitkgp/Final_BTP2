import os

# =========================================================
# HARD CPU MODE (prevents accidental CUDA init on HF/Streamlit Cloud)
# =========================================================
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NVIDIA_VISIBLE_DEVICES"] = "none"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import json
import time
import hmac
import hashlib
import tempfile
import copy
import base64
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from contextlib import contextmanager

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
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
# Styling (system-auto theme)
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

  /* DARK MODE REFINEMENTS */
  @media (prefers-color-scheme: dark) {
    :root{
      --bg: #0b0f14;
      --panel: #161b22;
      --panel2: #21262d;
      --text: #f0f6fc;
      --muted: #8b949e;
      --border: #30363d;
      --accent: #58a6ff;
    }
  }

  /* LIGHT MODE REFINEMENTS */
  @media (prefers-color-scheme: light) {
    :root{
      --bg: #ffffff;
      --panel: #f6f8fa;
      --panel2: #eaeef2;
      --text: #24292f;
      --muted: #57606a;
      --border: #d0d7de;
      --accent: #0969da;
    }
  }

  .stApp { background: var(--bg); color: var(--text); }
  
  /* REDUCED MARGINS: Widen container to 95% */
  .block-container { 
      padding-top: 1.25rem; 
      padding-bottom: 2.25rem; 
      max-width: 95% !important; 
  }

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
    color: var(--text);
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
  .kpi-value { font-size: 1.45rem; font-weight: 800; letter-spacing: -0.01em; color: var(--text); }

  .stButton>button { border-radius: 12px !important; border: 1px solid var(--border) !important; }
  [data-baseweb="input"] input, [data-baseweb="textarea"] textarea { border-radius: 12px !important; }

  /* Forecast button color */
  div[data-testid="stBaseButton-primary"] button,
  button[kind="primary"] {
    background: #ff3b30 !important;
    color: #ffffff !important;
    border-color: rgba(255,255,255,0.0) !important;
  }
  div[data-testid="stBaseButton-primary"] button:hover,
  button[kind="primary"]:hover {
    background: #e63229 !important;
    color: #ffffff !important;
  }

  .pdf-frame {
    border: 1px solid var(--border);
    border-radius: 14px;
    box-shadow: var(--shadow);
    overflow: hidden;
  }

  /* CUSTOM FAQ ACCORDION STYLING (Native feel, Reduced Boldness, One-at-a-time) */
  details.custom-faq {
    background-color: var(--panel); /* Adapts to light/dark */
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 12px;
    transition: all 0.2s ease;
  }
  details.custom-faq[open] {
    background-color: var(--bg);
    border-color: var(--muted);
  }
  details.custom-faq summary {
    padding: 14px 16px;
    cursor: pointer;
    font-weight: 600; /* Reduced from 800 to 600 */
    font-size: 1.05rem;
    list-style: none;
    outline: none;
    color: var(--text);
    display: flex;
    align-items: center;
  }
  details.custom-faq summary::-webkit-details-marker {
    display: none;
  }
  details.custom-faq summary::after {
    content: '+'; 
    margin-left: auto; 
    font-weight: 400;
    color: var(--muted);
    font-size: 1.2rem;
  }
  details.custom-faq[open] summary::after {
    content: '-';
  }
  .faq-answer-body {
    padding: 0 16px 16px 16px;
    line-height: 1.6;
    color: var(--text);
    opacity: 0.9;
    border-top: 1px solid var(--border);
    margin-top: -4px;
    padding-top: 12px;
  }
   
  /* Radio button horizontal fix & Font Size Increase */
  div[role="radiogroup"] {
      gap: 2rem;
  }
  div[role="radiogroup"] label p {
      font-size: 1.1rem !important;
      font-weight: 500 !important;
  }
  
  /* TABS STYLING (Middle Ground) */
  button[data-baseweb="tab"] {
      font-size: 1.1rem !important;
      font-weight: 600 !important;
  }
  /* Target text inside tabs just in case */
  button[data-baseweb="tab"] div p {
      font-size: 1.1rem !important;
      font-weight: 600 !important;
  }
</style>
""",
    unsafe_allow_html=True,
)

AUTHOR_LINE = "Sameer Pagarani (21MA20002)"
SUPERVISOR_LINE = "Supervisor: Prof. Pawan Kumar"
FOOTER_COMBINED = "Designed & engineered by Sameer Pagarani"

APP_DIR = Path(__file__).resolve().parent


# =========================================================
# Password (clears input safely without touching session_state after widget instantiation)
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
        f"""
        <div class="hero">
          <div class="hero-title">🔒 Private TFT Stock Forecaster</div>
          <div class="hero-subtitle">Enter the password to open the app.</div>
          <span class="chip">Made by: {AUTHOR_LINE}</span>
          <span class="chip">{SUPERVISOR_LINE}</span>
        </div>
        <div class="hr"></div>
        """,
        unsafe_allow_html=True,
    )

    if "pw_nonce" not in st.session_state:
        st.session_state["pw_nonce"] = 0
    pw_key = f"pw_box_{st.session_state['pw_nonce']}"

    if st.session_state.get("auth_error", ""):
        st.error(st.session_state["auth_error"])
        st.session_state["auth_error"] = ""

    with st.form("auth_form", clear_on_submit=True):
        guess = st.text_input("Password", type="password", placeholder="Enter password…", key=pw_key)
        ok = st.form_submit_button("Unlock")

    if ok:
        good = hmac.compare_digest(_pbkdf2_hex(guess, salt), phash) if phash and salt else hmac.compare_digest(guess, plain)
        if good:
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.session_state["auth_error"] = "Wrong password."
            st.session_state["pw_nonce"] += 1  # creates a NEW widget key next rerun (input clears)
            st.rerun()

    return False

if not check_password():
    st.stop()


# =========================================================
# Helpers to locate files
# =========================================================
def pick_path(*candidates: str) -> Path:
    for c in candidates:
        p = (APP_DIR / c).resolve()
        if p.exists():
            return p
    raise FileNotFoundError("Missing required file(s).")

def pick_best_ckpt() -> Path:
    candidates = []
    for pat in ["*.ckpt", "artifacts/*.ckpt", "src/*.ckpt", "src/artifacts/*.ckpt"]:
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
    return pick_path("model.ckpt", "artifacts/model.ckpt", "src/model.ckpt", "src/artifacts/model.ckpt")

CKPT_PATH   = pick_best_ckpt()
PARAMS_PATH = pick_path("dataset_params.pth", "artifacts/dataset_params.pth", "src/dataset_params.pth", "src/artifacts/dataset_params.pth")
CFG_PATH    = pick_path("config.json", "artifacts/config.json", "src/config.json", "src/artifacts/config.json")


# =========================================================
# Torchmetrics CPU patch (prevents cuda init inside Metric._apply)
# =========================================================
@contextmanager
def torchmetrics_cpu_patch():
    try:
        import torchmetrics  # noqa: F401
        from torchmetrics.metric import Metric

        orig_apply = Metric._apply

        def safe_apply(self, fn):
            try:
                self._device = torch.device("cpu")
            except Exception:
                pass
            return orig_apply(self, fn)

        Metric._apply = safe_apply
        yield
        Metric._apply = orig_apply
    except Exception:
        # If torchmetrics isn't present or patching fails, just run normally.
        yield


# =========================================================
# CPU-safe checkpoint loader
# =========================================================
@st.cache_resource
def load_tft_checkpoint_force_cpu(ckpt_path: str) -> TemporalFusionTransformer:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    sd = ckpt.get("state_dict", None)
    if isinstance(sd, dict):
        for k, v in list(sd.items()):
            if torch.is_tensor(v):
                sd[k] = v.detach().cpu()
        ckpt["state_dict"] = sd

    tmp_name = "tft_cpu_" + hashlib.md5(str(ckpt_path).encode("utf-8")).hexdigest() + ".ckpt"
    tmp_ckpt = Path(tempfile.gettempdir()) / tmp_name
    torch.save(ckpt, tmp_ckpt)

    with torchmetrics_cpu_patch():
        model = TemporalFusionTransformer.load_from_checkpoint(str(tmp_ckpt), map_location="cpu")
        model.to("cpu").eval()
    return model


# =========================================================
# Trained universe (39) + names
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
# Load cfg/params/model (after password) — CPU only
# =========================================================
@st.cache_resource
def load_cfg_params_model() -> Tuple[Dict[str, Any], Dict[str, Any], TemporalFusionTransformer, str, int, int, int]:
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
# Data download (cached)
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
    if df.empty or "Date" not in df.columns:
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

@st.cache_data(ttl=21600)
def download_ohlcv(ticker: str, start: str, cache_buster: int = 0) -> Tuple[pd.DataFrame, str]:
    _ = cache_buster
    t = str(ticker).strip().upper()

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

    df2 = _download_stooq_ohlcv(t, start=start)
    if not df2.empty:
        return df2, "stooq"

    raise ValueError(f"No usable data found for {t}. Check spelling or try a different ticker.")


@st.cache_data(ttl=21600)
def download_market_context(start: str, cache_buster: int = 0) -> pd.DataFrame:
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
# Feature building
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
    return 100.0 - (100.0 / (1.0 + rs))

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()

def add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_px"] = np.log(out["px"].astype(float))
    out["log_ret"] = out["log_px"].diff().fillna(0.0)
    out["log_volume"] = np.log1p(out["volume"].fillna(0.0).clip(lower=0.0))

    dt = pd.to_datetime(out["date"])
    out["dow"] = dt.dt.dayofweek.astype(str)
    out["month"] = dt.dt.month.astype(str)
    out["year"] = dt.dt.year.astype(int)
    out["day"] = dt.dt.day.astype(int)

    win = 20
    mid = out["px"].astype(float).rolling(win, min_periods=win).mean()
    std = out["px"].astype(float).rolling(win, min_periods=win).std(ddof=0)
    out["bb_mid"] = mid
    out["bb_high"] = mid + 2.0 * std
    out["bb_low"] = mid - 2.0 * std
    out["bb_width"] = (out["bb_high"] - out["bb_low"]) / (out["bb_mid"].abs() + 1e-12)

    out["ema_20"] = _ema(out["px"].astype(float), 20)
    out["ema_50"] = _ema(out["px"].astype(float), 50)
    out["rsi_14"] = _rsi(out["px"].astype(float), 14)

    ema12 = _ema(out["px"].astype(float), 12)
    ema26 = _ema(out["px"].astype(float), 26)
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_diff"] = out["macd"] - out["macd_signal"]

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

def to_q(pred) -> np.ndarray:
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    q = np.asarray(pred)
    # normalize to (1, H, 3)
    if q.ndim == 2 and q.shape[1] == 3:
        q = q[None, :, :]
    if q.ndim == 2 and q.shape[0] == 3:
        q = np.transpose(q, (1, 0))[None, :, :]
    if q.ndim == 3 and q.shape[1] == 3:
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

    # FETCH COMPANY NAME (Custom Only)
    company_name = COMPANY_MAP.get(t, t)
    if allow_custom:
        try:
            # lightweight fetch
            inf = yf.Ticker(t).info
            company_name = inf.get("longName") or inf.get("shortName") or t
        except Exception:
            company_name = t

    params = patch_params_allow_unknown_categories(DATASET_PARAMS) if allow_custom else DATASET_PARAMS

    bias = 0.0
    if bias_calibrate and len(base) > (MAX_ENC + 50):
        try:
            eval_ds = TimeSeriesDataSet.from_parameters(params, base, stop_randomization=True)
            eval_loader = eval_ds.to_dataloader(train=False, batch_size=256, num_workers=0)
            outp = MODEL.predict(eval_loader, mode="quantiles", mode_kwargs={"quantiles": [0.1, 0.5, 0.9]})
            q = to_q(outp)
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
    q = to_q(pred)[0]  # (H,3)
    q10, q50, q90 = q[:, 0] - bias, q[:, 1] - bias, q[:, 2] - bias

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
    return base, out_df, missing, source_used, company_name


# =========================================================
# Report: inline PDF viewer using EMBED tag (Fastest)
# =========================================================
def find_report_file() -> Optional[Path]:
    candidates = []
    for p in APP_DIR.rglob("*.pdf"):
        name = p.name.lower()
        if "report" in name or "thesis" in name:
            candidates.append(p)
    if not candidates:
        return None
    for p in candidates:
        if p.name.lower() == "report.pdf":
            return p
    return sorted(candidates, key=lambda x: len(str(x)))[0]

def render_pdf_inline(pdf_bytes: bytes, height: int = 1000):
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    # Using <embed> utilizes the browser's built-in, optimized PDF engine
    pdf_display = f"""
    <embed src="data:application/pdf;base64,{b64}" width="100%" height="{height}px" type="application/pdf">
    """
    st.markdown(pdf_display, unsafe_allow_html=True)


# =========================================================
# Header
# =========================================================
st.markdown(
    f"""
<div class="hero">
  <div class="hero-title">📈 TFT Stock Forecaster</div>
  <div class="hero-subtitle">Explore probabilistic forecasts (q10/q50/q90) on a trained universe.</div>
  <span class="chip">Target: <span class="mono">{TARGET_COL}</span></span>
  <span class="chip">Encoder: <span class="mono">{MAX_ENC}</span></span>
  <span class="chip">PredLen: <span class="mono">{MAX_PRED}</span></span>
  <span class="chip">Checkpoint: <span class="mono">{CKPT_PATH.name}</span></span>
  <span class="chip">Made by: {AUTHOR_LINE}</span>
  <span class="chip">{SUPERVISOR_LINE}</span>
</div>
<div class="hr"></div>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Forecast", "❓ FAQs", "ℹ️ About", "📄 Report"])


# =========================================================
# Forecast rendering (persist results)
# =========================================================
def render_forecast_view(payload: dict):
    base: pd.DataFrame = payload["base"]
    out: pd.DataFrame = payload["out"]
    missing: List[str] = payload["missing"]
    source_used: str = payload["source_used"]
    horizon: int = payload["horizon"]
    company_name: str = payload.get("company_name", payload["ticker"])

    st.markdown(
        f'<span class="chip">Data source: <span class="mono">{source_used}</span></span> <span class="chip">Company: <b>{company_name}</b></span>',
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

    if "focus_choice" not in st.session_state:
        st.session_state["focus_choice"] = "2Y"

    ch_l, ch_r = st.columns([0.72, 0.28])
    with ch_l:
        st.markdown("### Chart")
    with ch_r:
        st.selectbox(
            "Focus window",
            ["3M", "6M", "1Y", "2Y", "5Y", "All"],
            key="focus_choice",
            help="Keeps the forecast region clearly visible even if the full history is long."
        )

    focus_choice = st.session_state["focus_choice"]

    hist = base[["date", "px"]].rename(columns={"px": "price"}).copy()
    hist["date"] = pd.to_datetime(hist["date"])
    band = out.copy()
    band["date"] = pd.to_datetime(band["date"])
    last_hist_date = pd.to_datetime(hist["date"]).max()

    hist_over = hist
    if len(hist_over) > 2200:
        step = int(np.ceil(len(hist_over) / 2200))
        hist_over = hist_over.iloc[::step].copy()

    if focus_choice == "3M":
        focus_start = last_hist_date - pd.DateOffset(months=3)
    elif focus_choice == "6M":
        focus_start = last_hist_date - pd.DateOffset(months=6)
    elif focus_choice == "1Y":
        focus_start = last_hist_date - pd.DateOffset(years=1)
    elif focus_choice == "2Y":
        focus_start = last_hist_date - pd.DateOffset(years=2)
    elif focus_choice == "5Y":
        focus_start = last_hist_date - pd.DateOffset(years=5)
    else:
        focus_start = pd.to_datetime(hist["date"]).min()

    focus_start = max(pd.to_datetime(hist["date"]).min(), focus_start)
    hist_focus = hist[hist["date"] >= focus_start].copy()

    # Visible in both themes
    split_color = "#F97316"  # orange split line
    hist_color = "#60A5FA"   # bright blue
    pred_color = "#ff3b30"   # red

    overview = (
        alt.Chart(hist_over)
        .mark_line(color=hist_color)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("price:Q", title="Price"),
        )
        .properties(height=150)
    ) + alt.Chart(pd.DataFrame({"date":[last_hist_date]})).mark_rule(color=split_color, strokeWidth=2.6, opacity=0.95).encode(x="date:T")

    band_area = (
        alt.Chart(band)
        .mark_area(color=pred_color, opacity=0.20)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("pred_px_q10:Q", title="Price"),
            y2="pred_px_q90:Q",
        )
    )
    median_line = (
        alt.Chart(band)
        .mark_line(color=pred_color, strokeWidth=4.0)
        .encode(x="date:T", y="pred_px_q50:Q")
    )
    hist_line_focus = (
        alt.Chart(hist_focus)
        .mark_line(color=hist_color)
        .encode(x="date:T", y="price:Q")
    )
    split_rule = alt.Chart(pd.DataFrame({"date":[last_hist_date]})).mark_rule(color=split_color, strokeWidth=2.6, opacity=0.95).encode(x="date:T")

    focus = (band_area + hist_line_focus + median_line + split_rule).properties(height=420).interactive()
    st.altair_chart(alt.vconcat(overview, focus), use_container_width=True)

    with st.expander("🔎 Show last 15 rows of input data"):
        st.dataframe(base.tail(15), use_container_width=True)

    # Disclaimer ONLY on forecast page bottom (no name/roll here)
    st.markdown(
        '<div class="hr"></div><div class="muted">Disclaimer: Educational / research demo — not financial advice. Forecasts are probabilistic and can be wrong.</div>',
        unsafe_allow_html=True
    )


# =========================================================
# Forecast tab — CLUTTER-FREE 2-STEP UI
# =========================================================
with tab1:
    # 1. Ticker Selection Section
    mode = st.radio("Input Source", ["Trained Ticker", "Custom Ticker"], horizontal=True, label_visibility="collapsed")
    
    col_ticker, col_settings, col_rel = st.columns([1.5, 1, 1], gap="medium")
    
    with col_ticker:
        st.subheader("Ticker")
        if mode == "Trained Ticker":
            trained_rows = [{"ticker": t, "company": COMPANY_MAP.get(t, "")} for t in TRAINED_39]
            labels = [f"{r['ticker']} — {r['company']}" if r["company"] else r["ticker"] for r in trained_rows]
            label_to_ticker = {lab: lab.split(" — ")[0] for lab in labels}
            
            # Use index=None to start empty, and placeholder to show the message inside the box
            chosen_label = st.selectbox(
                "Select ticker (type to search)", 
                labels, 
                index=None,
                placeholder="Select ticker (type to search)",
                label_visibility="collapsed" # Hides the outer label as requested
            )
            
            if chosen_label:
                ticker = label_to_ticker[chosen_label]
            else:
                ticker = "" # Handle None case
            
            allow_custom = False
        else:
            ticker = st.text_input("Enter Custom Ticker", placeholder="Search for any ticker (e.g., RKLB,RGTI,IREN,etc)", label_visibility="collapsed").strip().upper()
            allow_custom = True
            if ticker:
                st.info(f"ℹ️ **{ticker}** is not in the trained set. I'll allow if it has enough history, but reliability may drop.")

    with col_settings:
        st.subheader("Forecast settings")
        horizon = int(st.selectbox("Horizon (trading days)", [1, 3, 5], index=2, help="Number of trading days to forecast into the future."))
        years = int(st.selectbox("History window (years)", [5, 10, 15, 20], index=3, help="Amount of past data used to generate the forecast."))

    with col_rel:
        st.subheader("Reliability")
        bias_cal = st.checkbox("Bias calibrate (best effort)", value=True, help="Adjusts forecast based on recent model error.")
        force_refresh = st.checkbox("Force refresh (live fetch)", value=False, help="Ignore cache and fetches fresh data immediately. Can be slower and may hit rate limits.")
        st.write("")
        run = st.button("🚀 Forecast", use_container_width=True, type="primary")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Persist results
    if run:
        if not ticker:
             st.error("Please select or enter a ticker symbol.")
        else:
            try:
                with st.spinner("Running forecast…"):
                    base, out, missing, source_used, company_name = forecast_ticker(
                        ticker, horizon, years, bias_cal, force_refresh, allow_custom=allow_custom
                    )

                st.session_state["forecast_payload"] = {
                    "ticker": ticker,
                    "company_name": company_name,
                    "horizon": horizon,
                    "years": years,
                    "allow_custom": allow_custom,
                    "bias_cal": bias_cal,
                    "force_refresh": force_refresh,
                    "source_used": source_used,
                    "missing": missing,
                    "base": base,
                    "out": out,
                    "ts": time.time(),
                }

            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error("Forecast failed due to an unexpected error.")
                with st.expander("Show error details"):
                    st.exception(e)

    if st.session_state.get("forecast_payload"):
        render_forecast_view(st.session_state["forecast_payload"])


# =========================================================
# FAQs (HTML DETAILS with name attribute = NATIVE ACCORDION BEHAVIOR)
# =========================================================
with tab2:
    st.markdown("## FAQs")

    faq_items: List[Tuple[str, str]] = [
        (
            "How accurate is the model? Test performance summary?",
            (
                "Evaluated on a held-out test split across <b>39 trained tickers</b>.<br/><br/>"
                "<b>Global test metrics (N = 39):</b><br/>"
                "• <b>H = 1 day</b>: RMSE 7.1895 | MAE 4.9166 | <b>MAPE 1.951%</b> | <b>Accuracy 98.049%</b> | DirAcc 48.72% | Cov80 76.92%<br/>"
                "• <b>H = 3 days</b>: RMSE 14.4184 | MAE 9.2617 | <b>MAPE 3.506%</b> | <b>Accuracy 96.494%</b> | DirAcc 74.36% | Cov80 92.31%<br/>"
                "• <b>H = 5 days</b>: RMSE 10.4689 | MAE 7.2878 | <b>MAPE 3.069%</b> | <b>Accuracy 96.931%</b> | DirAcc 53.85% | Cov80 94.87%"
            ),
        ),
        ("What do q10 / q50 / q90 mean?", "q50: the median forecast (most typical outcome)<br/>q10: conservative scenario (10% quantile)<br/>q90: optimistic scenario (90% quantile)<br/><br/>The q10–q90 band represents roughly 80% uncertainty."),
        ("Why are there two charts (overview vs focus)?", "Top shows the full selected history overview.<br/>Bottom is the focus window so the forecast stays visible without zooming manually."),
        ("What is “Bias calibrate (best effort)”?", "It estimates recent systematic drift and nudges the next forecast.<br/>Optional — safe to leave ON for potentially better alignment. If the model is consistently over- or under-shooting recently, bias calibration can improve stability."),
        ("Why do some tickers fail with “Not enough history”?", "The TFT uses a fixed-length <b>encoder window</b> (approx 450+ trading days). If a ticker is too new or has missing price history, the app refuses to forecast rather than produce low-quality output."),
        ("Can I try a ticker outside the trained list?", "Yes — if it has enough history.<br/>However, because it wasn’t seen during training, the confidence intervals might be less reliable."),
        ("What does “Force refresh (live fetch)” do?", "Forces a fresh data download now.<br/>It can be slower and may fail under rate limits. Leave OFF for fastest use, fewer failures since it uses cached downloads."),
        ("Is this financial advice?", "No.<br/>This is an educational / research demo only."),
    ]

    html_content = ""
    for q, a in faq_items:
        safe_q = q.replace('"', '&quot;')
        html_content += f"""
        <details class="custom-faq" name="faq-accordion">
            <summary>{safe_q}</summary>
            <div class="faq-answer-body">{a}</div>
        </details>
        """
      
    st.markdown(html_content, unsafe_allow_html=True)

    st.markdown(f'<div class="hr"></div><div class="muted">{FOOTER_COMBINED}</div>', unsafe_allow_html=True)


# =========================================================
# About
# =========================================================
with tab3:
    st.markdown("## About")
    st.write(
    """This app is a research-grade demo of a **Temporal Fusion Transformer (TFT)** used for stock probabilistic forecasting.\n\n

**What it does**
- Uses daily OHLCV data (Yahoo Finance with a Stooq fallback)
- Builds technical + market-context features (e.g., broad-market and sector proxies like **SPY / QQQ / SOXX**)
- Runs the trained TFT model and outputs **q10 / q50 / q90** forecasts (median + uncertainty band)
- Render the results in an interactive chart + a table you can inspect

**How to use it**
- Prefer the **trained tickers** for best reliability
- Custom tickers are allowed if enough history exists (quality may vary)
- **Force refresh** fetches the latest prices (slower, may hit rate limits)

**Reliability notes**
- Forecasts are generally most stable for **trained tickers**. Custom tickers may behave differently because they were not seen during training.
- **Force refresh** bypasses cached data to fetch “live” prices — useful for the latest update, but it can be slower or occasionally rate-limited.
- This is an academic / research project and **NOT a financial advice**."""
)

    st.markdown(f'<div class="hr"></div><div class="muted">{FOOTER_COMBINED}</div>', unsafe_allow_html=True)

# =========================================================
# Report (LAZY LOAD + DOWNLOAD)
# =========================================================
with tab4:
    rp = find_report_file()
    if rp is not None:
        file_size_mb = rp.stat().st_size / (1024 * 1024)
        pdf_bytes = rp.read_bytes()
        
        c_left, c_right = st.columns([0.7, 0.3])
        
        with c_left:
            st.markdown(f"### Project Report ({file_size_mb:.1f} MB)")
            # Simpler, punchier text
            st.write("You can download report directly or view it here (it may take a while to render the document).")
        
        with c_right:
            st.write("") # Spacer
            st.download_button("⬇️ Download report", data=pdf_bytes, file_name=rp.name, mime="application/pdf", type="primary")
        
        st.markdown("---")
        
        # Lazy Loading Button (Prevents initial page freeze)
        if "load_pdf" not in st.session_state:
            st.session_state["load_pdf"] = False
            
        if not st.session_state["load_pdf"]:
            if st.button("View Report Here"):
                st.session_state["load_pdf"] = True
                st.rerun()
        else:
            with st.spinner("Opening document..."):
                render_pdf_inline(pdf_bytes, height=1000)
            
            if st.button("Close Viewer"):
                st.session_state["load_pdf"] = False
                st.rerun()

    else:
        st.info("Report PDF not found in directory.")

    st.markdown(f'<div class="hr"></div><div class="muted">{FOOTER_COMBINED}</div>', unsafe_allow_html=True)
