import ccxt
import requests
import time
import os
import logging
from datetime import datetime
from flask import Flask
import threading
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

@app.route('/')
def home():
    return f"✅ АНАЛИТИК v6.0 АКТИВЕН. Время: {datetime.now().strftime('%H:%M:%S')}"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID        = os.getenv("CHAT_ID")

if not TELEGRAM_TOKEN or not CHAT_ID:
    logging.warning("⚠️ TELEGRAM_TOKEN или CHAT_ID не заданы!")

exchange = ccxt.mexc({
    'enableRateLimit': True,
    'timeout': 15000,
    'options': {'defaultType': 'swap'}
})

WATCHLIST = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
    'BNB/USDT:USDT', 'XRP/USDT:USDT', 'DOGE/USDT:USDT',
    'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT',
    'DOT/USDT:USDT',
]

# ─────────────────────────────────────────────
# ПОРОГИ ФИЛЬТРОВ
# ─────────────────────────────────────────────
MIN_VOLUME_SIGNAL    = 5_000_000   # $5M для СИГНАЛ
MIN_VOLUME_ATTENTION = 2_000_000   # $2M для ВНИМАНИЕ
MIN_RR_SIGNAL        = 3.0         # R/R для СИГНАЛ
MIN_RR_ATTENTION     = 2.0         # R/R для ВНИМАНИЕ
MIN_TARGET_PCT_SIGNAL    = 5.0     # цель минимум 5% для СИГНАЛ
MIN_TARGET_PCT_ATTENTION = 3.0     # цель минимум 3% для ВНИМАНИЕ
MAX_STOP_PCT         = 3.0         # стоп максимум 3%
SCORE_SIGNAL         = 7           # порог СИГНАЛ
SCORE_ATTENTION      = 4           # порог ВНИМАНИЕ
SWING_ATTENTION_PCT  = 2.0         # % от Swing High/Low для ВНИМАНИЕ

bot_status = {
    "started_at":     datetime.now().isoformat(),
    "last_iteration": None,
    "iterations":     0,
    "errors":         0,
    "signals_sent":   0,
    "attention_sent": 0,
}

oi_cache: dict = {}


# ─────────────────────────────────────────────
# SSMA (Smoothed Moving Average)
# period=24, source=(H+L+C)/3
# ГЛАВНЫЕ ВОРОТА направления торговли:
#   Лонг: цена > SSMA ИЛИ slope > 0
#   Шорт: цена < SSMA ИЛИ slope < 0
#   Оба условия нарушены = сигнал заблокирован
# ─────────────────────────────────────────────
def calculate_ssma(ohlcv, period=24):
    if len(ohlcv) < period + 10:
        return None, 'neutral', 0.0

    typical = [(c[2] + c[3] + c[4]) / 3 for c in ohlcv]
    smma = np.mean(typical[:period])
    smma_history = [smma]

    for i in range(period, len(typical)):
        smma = (smma * (period - 1) + typical[i]) / period
        smma_history.append(smma)

    current_smma  = smma_history[-1]
    prev_smma     = smma_history[-2] if len(smma_history) >= 2 else smma_history[-1]
    current_price = ohlcv[-1][4]
    slope         = (current_smma - prev_smma) / prev_smma * 100

    if current_price > current_smma and slope > 0:
        trend = 'bull_strong'
    elif current_price > current_smma and slope <= 0:
        trend = 'bull_weak'
    elif current_price < current_smma and slope < 0:
        trend = 'bear_strong'
    else:
        trend = 'bear_weak'

    return current_smma, trend, slope


def ssma_allows_long(ssma_val, ssma_trend, ssma_slope, current_price):
    """
    SSMA ворота для лонга.
    Разрешён если: цена выше SSMA ИЛИ SSMA растёт.
    Заблокирован если: цена НИЖЕ SSMA И SSMA падает.
    """
    if ssma_val is None:
        return True  # нет данных — не блокируем
    price_above = current_price > ssma_val
    ssma_rising = ssma_slope > 0
    return price_above or ssma_rising


def ssma_allows_short(ssma_val, ssma_trend, ssma_slope, current_price):
    """
    SSMA ворота для шорта.
    Разрешён если: цена ниже SSMA ИЛИ SSMA падает.
    Заблокирован если: цена ВЫШЕ SSMA И SSMA растёт.
    """
    if ssma_val is None:
        return True
    price_below = current_price < ssma_val
    ssma_falling = ssma_slope < 0
    return price_below or ssma_falling


# ─────────────────────────────────────────────
# R/R ФИЛЬТР
# Цель минимум target_pct%, стоп максимум MAX_STOP_PCT%
# Возвращает (passed, target_pct_actual, stop_pct_actual, rr_actual)
# ─────────────────────────────────────────────
def check_rr(current_price, target, stop, mode='long',
             min_target_pct=MIN_TARGET_PCT_SIGNAL,
             max_stop_pct=MAX_STOP_PCT):
    if target is None or stop is None:
        return False, 0.0, 0.0, 0.0

    if mode == 'long':
        target_pct = (target - current_price) / current_price * 100
        stop_pct   = (current_price - stop)   / current_price * 100
    else:
        target_pct = (current_price - target) / current_price * 100
        stop_pct   = (stop - current_price)   / current_price * 100

    if target_pct <= 0 or stop_pct <= 0:
        return False, target_pct, stop_pct, 0.0

    rr = target_pct / stop_pct

    passed = (target_pct >= min_target_pct
              and stop_pct  <= max_stop_pct
              and rr        >= (min_target_pct / max_stop_pct))

    return passed, target_pct, stop_pct, rr


# ─────────────────────────────────────────────
# ОБЪЁМ — расширенная информация
# ─────────────────────────────────────────────
def get_volume_info(closed, volume_24h):
    """
    Возвращает расширенный блок информации об объёме:
    - Относительный объём (x от нормы)
    - Z-score
    - Примерное соотношение покупок/продаж (по дисбалансу свечей)
    - 24H объём в $M
    """
    v_history = [x[5] for x in closed[-21:-1]]
    v_avg     = np.mean(v_history) if v_history else 1.0
    v_cur     = closed[-1][5]
    v_rel     = v_cur / v_avg if v_avg > 0 else 1.0

    # Z-score
    std = np.std(v_history) if len(v_history) > 1 else 1.0
    v_zscore = (v_cur - np.mean(v_history)) / std if std > 0 else 0.0

    # Соотношение покупок/продаж по последним 5 свечам
    buy_pct_list = []
    for c in closed[-5:]:
        h, l, cl = c[2], c[3], c[4]
        bp = (cl - l) / (h - l) * 100 if h != l else 50.0
        buy_pct_list.append(bp)
    avg_buy_pct = np.mean(buy_pct_list)
    sell_pct    = 100 - avg_buy_pct

    vol_24h_m = volume_24h / 1_000_000

    if v_rel >= 10 or v_zscore >= 3.0:
        vol_score = 3
        vol_label = f"🚀 Vol x{v_rel:.0f} Z:{v_zscore:.1f}σ ВЗРЫВ"
    elif v_rel >= 5 or v_zscore >= 2.0:
        vol_score = 2
        vol_label = f"📊 Vol x{v_rel:.1f} Z:{v_zscore:.1f}σ"
    elif v_rel >= 1.8:
        vol_score = 1
        vol_label = f"📊 Vol x{v_rel:.1f}"
    else:
        vol_score = 0
        vol_label = ""

    detail_block = (
        f"📦 Объём 24H: <b>${vol_24h_m:.1f}M</b> | "
        f"x{v_rel:.1f} от нормы | Z: {v_zscore:.1f}σ\n"
        f"   🟢 Покупки ~{avg_buy_pct:.0f}% / 🔴 Продажи ~{sell_pct:.0f}%"
    )

    return vol_score, vol_label, v_rel, v_zscore, avg_buy_pct, detail_block


# ─────────────────────────────────────────────
# BB OUTSIDE ATR
# ─────────────────────────────────────────────
def calculate_bb_outside_atr(ohlcv, bb_length=55, bb_std=0.712,
                               atr_length=20, atr_mult=0.618,
                               tilson_length=10, tilson_factor=0.3):
    if len(ohlcv) < bb_length + atr_length + 10:
        return 'neutral', 0.0, None, None, None

    closes = [c[4] for c in ohlcv]
    highs  = [c[2] for c in ohlcv]
    lows   = [c[3] for c in ohlcv]

    f  = tilson_factor
    c1 = -(f**3)
    c2 = 3 * f**2 + 3 * f**3
    c3 = -6 * f**2 - 3 * f - 3 * f**3
    c4 = 1 + 3 * f + f**3 + 3 * f**2

    chunk = closes[-min(len(closes), bb_length + tilson_length * 6):]
    k     = 2 / (tilson_length + 1)

    def make_ema(src):
        e = src[0]
        out = []
        for v in src:
            e = v * k + e * (1 - k)
            out.append(e)
        return out

    e1 = make_ema(chunk)
    e2 = make_ema(e1)
    e3 = make_ema(e2)
    e4 = make_ema(e3)
    e5 = make_ema(e4)
    e6 = make_ema(e5)

    basis = c1*e6[-1] + c2*e5[-1] + c3*e4[-1] + c4*e3[-1]

    trs = []
    for i in range(1, len(ohlcv)):
        h  = highs[i]; l = lows[i]; pc = closes[i-1]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))

    if len(trs) < atr_length:
        return 'neutral', 0.0, None, None, None

    atr_val = np.mean(trs[-atr_length:])
    upper   = basis + atr_mult * atr_val
    lower   = basis - atr_mult * atr_val
    current = closes[-1]

    dist_lower = (lower - current) / lower * 100
    dist_upper = (current - upper) / upper * 100

    if current < lower:
        return 'oversold',   dist_lower, upper, lower, basis
    elif current > upper:
        return 'overbought', dist_upper, upper, lower, basis
    else:
        return 'neutral', 0.0, upper, lower, basis


# ─────────────────────────────────────────────
# SWING HILO (34 bars)
# ─────────────────────────────────────────────
def calculate_swing_hilo(ohlcv, swing_bars=34):
    if len(ohlcv) < swing_bars * 2 + 1:
        return 100.0, 100.0, False, False, None, None

    highs   = [c[2] for c in ohlcv]
    lows    = [c[3] for c in ohlcv]
    n       = len(ohlcv)
    last_sw_low  = None
    last_sw_high = None

    for i in range(swing_bars, n - swing_bars - 1):
        if lows[i]  == min(lows[i  - swing_bars:i + swing_bars + 1]):
            last_sw_low  = lows[i]
        if highs[i] == max(highs[i - swing_bars:i + swing_bars + 1]):
            last_sw_high = highs[i]

    current = ohlcv[-1][4]

    if last_sw_low  is None: last_sw_low  = min(lows[-60:])
    if last_sw_high is None: last_sw_high = max(highs[-60:])

    sw_low_pct  = (current - last_sw_low)   / last_sw_low  * 100
    sw_high_pct = (last_sw_high - current)  / last_sw_high * 100

    near_sw_low  = 0 <= sw_low_pct  <= SWING_ATTENTION_PCT
    near_sw_high = 0 <= sw_high_pct <= SWING_ATTENTION_PCT

    return sw_low_pct, sw_high_pct, near_sw_low, near_sw_high, last_sw_low, last_sw_high


# ─────────────────────────────────────────────
# WYCKOFF ФАЗЫ
# ─────────────────────────────────────────────
def detect_wyckoff_phase(ohlcv, swing_low, swing_high, cvd_level):
    if len(ohlcv) < 60:
        return 'ranging', True, True, '⚪️ Диапазон', 0

    closes  = [c[4] for c in ohlcv]
    volumes = [c[5] for c in ohlcv]
    current = closes[-1]

    price_range  = swing_high - swing_low
    if price_range <= 0: price_range = swing_high * 0.01
    position_pct = (current - swing_low) / price_range * 100

    lower_third = swing_low + price_range * 0.33
    upper_third = swing_low + price_range * 0.67

    vol_at_lows  = np.mean([volumes[i] for i in range(len(ohlcv))
                             if closes[i] <= lower_third] or [0])
    vol_at_highs = np.mean([volumes[i] for i in range(len(ohlcv))
                             if closes[i] >= upper_third] or [0])
    vol_avg = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)

    recent_ranges  = [ohlcv[i][2] - ohlcv[i][3] for i in range(-10, 0)]
    earlier_ranges = [ohlcv[i][2] - ohlcv[i][3] for i in range(-30, -10)]
    vol_compression = (np.mean(recent_ranges) / np.mean(earlier_ranges)) \
                      if np.mean(earlier_ranges) > 0 else 1.0

    ma20_start  = np.mean(closes[-25:-20]) if len(closes) >= 25 else closes[0]
    ma20_end    = np.mean(closes[-5:])
    trend_20    = (ma20_end - ma20_start) / ma20_start * 100

    # Spring
    recent_lows  = [c[3] for c in ohlcv[-5:]]
    spring_detected = (min(recent_lows) < swing_low * 1.005
                       and current > swing_low * 1.01
                       and cvd_level in ('bull', 'bull_div'))
    if spring_detected:
        return 'accumulation', True, False, '🌱 Spring (Wyckoff)', 3

    # UTAD
    recent_highs = [c[2] for c in ohlcv[-5:]]
    utad_detected = (max(recent_highs) > swing_high * 0.995
                     and current < swing_high * 0.99
                     and cvd_level in ('bear', 'bear_div'))
    if utad_detected:
        return 'distribution', False, True, '🔝 UTAD (Wyckoff)', 3

    # Accumulation
    if (position_pct <= 25
            and vol_at_lows > vol_avg * 1.2
            and cvd_level in ('bull', 'bull_div')
            and vol_compression < 0.9):
        strength = 0
        if position_pct <= 15:        strength += 1
        if vol_at_lows > vol_avg * 2: strength += 1
        if cvd_level == 'bull_div':   strength += 1
        return 'accumulation', True, False, '🟢 Накопление (Wyckoff)', min(strength, 3)

    # Distribution
    if (position_pct >= 75
            and vol_at_highs > vol_avg * 1.2
            and cvd_level in ('bear', 'bear_div')
            and vol_compression < 0.9):
        strength = 0
        if position_pct >= 85:         strength += 1
        if vol_at_highs > vol_avg * 2: strength += 1
        if cvd_level == 'bear_div':    strength += 1
        return 'distribution', False, True, '🔴 Распределение (Wyckoff)', min(strength, 3)

    # Markup
    if trend_20 > 3.0 and position_pct > 50 and cvd_level in ('bull', 'bull_div'):
        return 'markup', True, False, '📈 Разгон (Markup)', 1

    # Markdown
    if trend_20 < -3.0 and position_pct < 50 and cvd_level in ('bear', 'bear_div'):
        return 'markdown', False, True, '📉 Снижение (Markdown)', 1

    return 'ranging', True, True, '⚪️ Диапазон', 0


# ─────────────────────────────────────────────
# ФИЛЬТР ПОСЛЕ БОЛЬШОГО ДВИЖЕНИЯ
# ─────────────────────────────────────────────
def check_post_move_filter(ohlcv, lookback=12):
    if len(ohlcv) < lookback + 1:
        return False, False, 0.0, 0

    max_down = 0.0; max_up = 0.0
    down_idx = 0;   up_idx = 0

    for i in range(len(ohlcv) - lookback, len(ohlcv)):
        candle = ohlcv[i]
        o, h, l, c = candle[1], candle[2], candle[3], candle[4]
        md = (o - l) / o * 100 if o > 0 else 0
        mu = (h - o) / o * 100 if o > 0 else 0
        if i > 0:
            pc = ohlcv[i-1][4]
            md = max(md, (pc - c) / pc * 100 if pc > 0 else 0)
            mu = max(mu, (c - pc) / pc * 100 if pc > 0 else 0)
        if md > max_down: max_down = md; down_idx = i
        if mu > max_up:   max_up   = mu; up_idx   = i

    last_idx = len(ohlcv) - 1
    after_dump = (max_down >= 15.0 and 1 <= (last_idx - down_idx) <= 6)
    after_pump = (max_up   >= 15.0 and 1 <= (last_idx - up_idx)   <= 6)

    move_pct      = max_down if after_dump else (max_up if after_pump else 0.0)
    candles_since = (last_idx - down_idx) if after_dump else \
                    ((last_idx - up_idx)  if after_pump else 0)

    return after_dump, after_pump, move_pct, candles_since


# ─────────────────────────────────────────────
# CVD
# ─────────────────────────────────────────────
def calc_cvd_level(closed):
    if len(closed) < 10:
        return 'neutral', 0.0
    deltas = []
    for c in closed:
        h, l, cl, v = c[2], c[3], c[4], c[5]
        ratio = (cl - l) / (h - l) if h != l else 0.5
        deltas.append((ratio - 0.5) * 2 * v)

    cumulative = np.cumsum(deltas)
    total_cvd  = cumulative[-1]
    lookback   = 10
    closes     = [x[4] for x in closed]
    if len(closes) < lookback + 1:
        return 'neutral', total_cvd

    price_new_low  = closes[-1] < min(closes[-lookback:-1])
    price_new_high = closes[-1] > max(closes[-lookback:-1])
    cvd_new_low    = cumulative[-1] < min(cumulative[-lookback:-1])
    cvd_new_high   = cumulative[-1] > max(cumulative[-lookback:-1])

    if price_new_low  and not cvd_new_low:  return 'bull_div', total_cvd
    if price_new_high and not cvd_new_high: return 'bear_div', total_cvd
    if total_cvd > 0:                       return 'bull',     total_cvd
    return 'bear', total_cvd


def get_cvd_divergence(ohlcv, mode='long'):
    closes     = [x[4] for x in ohlcv]
    lookback   = 10
    cumulative = 0.0
    cvd_proxy  = []
    for candle in ohlcv:
        o, h, l, c, v = candle[1], candle[2], candle[3], candle[4], candle[5]
        ratio      = (c - l) / (h - l) if h != l else 0.5
        cumulative += (ratio - 0.5) * 2 * v
        cvd_proxy.append(cumulative)
    if len(closes) < lookback + 1:
        return False
    if mode == 'long':
        return (closes[-1] < min(closes[-lookback:-1])
                and cvd_proxy[-1] > min(cvd_proxy[-lookback:-1]))
    else:
        return (closes[-1] > max(closes[-lookback:-1])
                and cvd_proxy[-1] < max(cvd_proxy[-lookback:-1]))


# ─────────────────────────────────────────────
# RSI ДИВЕРГЕНЦИЯ
# ─────────────────────────────────────────────
def calc_rsi_divergence(closed, rsi_period=14, lookback=10):
    closes = [x[4] for x in closed]
    if len(closes) < rsi_period + lookback + 2:
        return False, False

    def rsi_at(sl):
        if len(sl) < rsi_period + 1: return 50.0
        d  = np.diff(sl)
        g  = np.where(d > 0, d, 0.0)
        ls = np.where(d < 0, -d, 0.0)
        ag = np.mean(g[:rsi_period]); al = np.mean(ls[:rsi_period])
        for i in range(rsi_period, len(g)):
            ag = (ag * (rsi_period-1) + g[i])  / rsi_period
            al = (al * (rsi_period-1) + ls[i]) / rsi_period
        if al == 0: return 100.0
        return 100.0 - (100.0 / (1.0 + ag / al))

    rsi_values = []
    for i in range(lookback + 1):
        idx = len(closes) - lookback - 1 + i
        rsi_values.append(rsi_at(closes[max(0, idx - rsi_period - 1): idx + 1]))

    pw   = closes[-(lookback+1):]
    bull = pw[-1] < min(pw[:-1]) and rsi_values[-1] >= min(rsi_values[:-1])
    bear = pw[-1] > max(pw[:-1]) and rsi_values[-1] <= max(rsi_values[:-1])
    return bull, bear


# ─────────────────────────────────────────────
# OI
# ─────────────────────────────────────────────
def get_oi_data_from_ticker(symbol, tickers, price_change_pct):
    try:
        ticker   = tickers.get(symbol)
        if not ticker: return 0.0, 'neutral', '⚪️ OI: нет данных'
        info     = ticker.get('info', {})
        hold_vol = float(info.get('holdVol', 0) or 0)
        if hold_vol == 0: return 0.0, 'neutral', '⚪️ OI: нет данных'

        prev_vol         = oi_cache.get(symbol, 0)
        oi_cache[symbol] = hold_vol

        if prev_vol == 0:
            return 0.0, 'neutral', f"⚪️ OI: {hold_vol:.0f} (первое измерение)"

        oi_chg = (hold_vol - prev_vol) / prev_vol * 100
        if abs(oi_chg) < 2.0:
            return oi_chg, 'neutral', f"⚪️ OI: {oi_chg:+.1f}% (нейтраль)"

        if   oi_chg > 0 and price_change_pct >= 0:
            return oi_chg, 'bull',       f"🟢 OI: +{oi_chg:.1f}% (деньги в лонг)"
        elif oi_chg > 0 and price_change_pct < 0:
            return oi_chg, 'bear',       f"🔴 OI: +{oi_chg:.1f}% (деньги в шорт)"
        elif oi_chg < 0 and price_change_pct >= 0:
            return oi_chg, 'squeeze_up', f"⚠️ OI: {oi_chg:.1f}% (шорт-сквиз)"
        else:
            return oi_chg, 'squeeze_dn', f"⚠️ OI: {oi_chg:.1f}% (лонг-ликвидации)"
    except Exception as e:
        logging.debug(f"OI error {symbol}: {e}")
        return 0.0, 'neutral', '⚪️ OI: нет данных'


# ─────────────────────────────────────────────
# ФАНДИНГ
# ─────────────────────────────────────────────
def get_funding_signal(symbol):
    try:
        fr   = exchange.fetch_funding_rate(symbol)
        rate = float(fr.get('fundingRate', 0) or 0) * 100
        if rate < -0.02:   return rate, 'bull',    f"🟢 Фандинг: {rate:.3f}%"
        elif rate > 0.05:  return rate, 'bear',    f"🔴 Фандинг: {rate:.3f}%"
        else:              return rate, 'neutral',  f"⚪️ Фандинг: {rate:.3f}%"
    except:
        return 0.0, 'neutral', '⚪️ Фандинг: нет данных'


# ─────────────────────────────────────────────
# PIVOT УРОВНИ
# ─────────────────────────────────────────────
def get_pivot_levels(ohlcv, tolerance=0.005):
    highs = [x[2] for x in ohlcv]; lows = [x[3] for x in ohlcv]
    pl = []; ph = []
    for i in range(2, len(ohlcv) - 2):
        if lows[i]  < lows[i-1]  and lows[i]  < lows[i-2]  and lows[i]  < lows[i+1]  and lows[i]  < lows[i+2]:  pl.append(lows[i])
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]: ph.append(highs[i])

    def cluster(pts, tol):
        if not pts: return []
        pts = sorted(pts); clusters = []; group = [pts[0]]
        for p in pts[1:]:
            if (p - group[0]) / group[0] <= tol: group.append(p)
            else: clusters.append(np.mean(group)); group = [p]
        clusters.append(np.mean(group))
        return [c for c in clusters if sum(1 for p in pts if abs(p-c)/c <= tol) >= 2]

    cp = ohlcv[-1][4]
    return (sorted([s for s in cluster(pl, tolerance) if s < cp], reverse=True),
            sorted([r for r in cluster(ph, tolerance) if r > cp]))


# ─────────────────────────────────────────────
# ATR
# ─────────────────────────────────────────────
def calculate_atr(ohlcv, period=14):
    if len(ohlcv) < period + 1: return None
    trs = [max(ohlcv[i][2]-ohlcv[i][3],
               abs(ohlcv[i][2]-ohlcv[i-1][4]),
               abs(ohlcv[i][3]-ohlcv[i-1][4]))
           for i in range(1, len(ohlcv))]
    atr = np.mean(trs[:period])
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def dynamic_atr_multipliers(btc_vol):
    if btc_vol > 3.0:   sk, tk = 2.0, 3.0
    elif btc_vol > 1.5: sk, tk = 1.5, 2.5
    else:               sk, tk = 1.2, 2.0
    return sk, tk, f"1:{tk/sk:.1f}"


# ─────────────────────────────────────────────
# MFI
# ─────────────────────────────────────────────
def calculate_mfi(ohlcv, period=14):
    if len(ohlcv) < period + 1: return 50.0
    tp_prev = None; pos_mf = []; neg_mf = []
    for c in ohlcv[-(period+1):]:
        h, l, cl, v = c[2], c[3], c[4], c[5]
        tp = (h + l + cl) / 3; mf = tp * v
        if tp_prev is not None:
            if tp > tp_prev:   pos_mf.append(mf); neg_mf.append(0.0)
            elif tp < tp_prev: neg_mf.append(mf); pos_mf.append(0.0)
            else:              pos_mf.append(0.0); neg_mf.append(0.0)
        tp_prev = tp
    pmf = sum(pos_mf); nmf = sum(neg_mf)
    if nmf == 0: return 100.0
    return 100.0 - (100.0 / (1.0 + pmf / nmf))


# ─────────────────────────────────────────────
# RSI WILDER
# ─────────────────────────────────────────────
def calculate_rsi_wilder(closes, period=14):
    if len(closes) < period + 1: return 50.0
    d  = np.diff(closes)
    g  = np.where(d > 0, d, 0.0); ls = np.where(d < 0, -d, 0.0)
    ag = np.mean(g[:period]);      al = np.mean(ls[:period])
    for i in range(period, len(g)):
        ag = (ag * (period-1) + g[i])  / period
        al = (al * (period-1) + ls[i]) / period
    if al == 0: return 100.0
    return 100.0 - (100.0 / (1.0 + ag / al))


# ─────────────────────────────────────────────
# DEMARK TD
# ─────────────────────────────────────────────
def update_td_counters(ohlcv, mode='long'):
    closed = ohlcv[:-1]
    closes = [x[4] for x in closed]; lows = [x[3] for x in closed]; highs = [x[2] for x in closed]
    n = len(closes)
    if n < 14: return False, False, False

    s_count = 0; in_c = False; c_count = 0
    setup_high = None; setup_low = None; setup_bars = []
    m9_signal = False; m9_perfect = False; m13_signal = False
    last_idx  = n - 1

    for i in range(4, n):
        c = closes[i]; c4 = closes[i-4]
        sc = (c < c4) if mode == 'long' else (c > c4)

        if not in_c:
            if sc:
                s_count += 1; setup_bars.append(i)
                if s_count == 9:
                    if len(setup_bars) >= 9:
                        i6,i7,i8,i9 = setup_bars[-4],setup_bars[-3],setup_bars[-2],setup_bars[-1]
                        if mode == 'long':
                            perfect = ((lows[i8] < lows[i6] and lows[i8] < lows[i7]) or
                                       (lows[i9] < lows[i6] and lows[i9] < lows[i7]))
                        else:
                            perfect = ((highs[i8] > highs[i6] and highs[i8] > highs[i7]) or
                                       (highs[i9] > highs[i6] and highs[i9] > highs[i7]))
                    else: perfect = False
                    if i == last_idx: m9_signal = True; m9_perfect = perfect
                    setup_high = max(highs[b] for b in setup_bars)
                    setup_low  = min(lows[b]  for b in setup_bars)
                    in_c = True; c_count = 0; s_count = 0; setup_bars = []
            else: s_count = 0; setup_bars = []
        else:
            if mode == 'long' and setup_high and c > setup_high:
                in_c = False; c_count = 0; setup_high = None; setup_low = None
                s_count = 1 if sc else 0; setup_bars = [i] if sc else []; continue
            elif mode == 'short' and setup_low and c < setup_low:
                in_c = False; c_count = 0; setup_high = None; setup_low = None
                s_count = 1 if sc else 0; setup_bars = [i] if sc else []; continue
            if i >= 2:
                cd = (c <= lows[i-2]) if mode == 'long' else (c >= highs[i-2])
                if cd:
                    c_count += 1
                    if c_count == 13:
                        if i == last_idx: m13_signal = True
                        in_c = False; c_count = 0; setup_high = None; setup_low = None
            if in_c and c_count > 30:
                in_c = False; c_count = 0; setup_high = None; setup_low = None

    return m9_signal, m9_perfect, m13_signal


# ─────────────────────────────────────────────
# МОЛОТ
# ─────────────────────────────────────────────
def check_hammer(ohlcv, mode='long'):
    if len(ohlcv) < 2: return False
    o, h, l, c = ohlcv[-2][1], ohlcv[-2][2], ohlcv[-2][3], ohlcv[-2][4]
    body = abs(c - o); fr = h - l if h != l else 1e-9
    if mode == 'long':
        return (min(o,c) - l) / fr > 0.6 and body / fr < 0.3
    else:
        return (h - max(o,c)) / fr > 0.6 and body / fr < 0.3


# ─────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ
# ─────────────────────────────────────────────
def build_tv_link(symbol):
    tv = symbol.replace('/', '').replace(':USDT', '.P')
    return f"🔗 <a href='https://www.tradingview.com/chart/?symbol=MEXC:{tv}'>TradingView</a>"


def send_msg(text):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10)
        if not r.ok: logging.error(f"TG error: {r.status_code} {r.text[:200]}")
    except Exception as e:
        logging.error(f"TG send error: {e}")


def get_market_context():
    for attempt in range(3):
        try:
            btc    = exchange.fetch_ohlcv('BTC/USDT:USDT', '4h', limit=5)
            btc_ch = ((btc[-1][4] - btc[-2][4]) / btc[-2][4]) * 100
            eth    = exchange.fetch_ohlcv('ETH/USDT:USDT', '4h', limit=5)
            eth_ch = ((eth[-1][4] - eth[-2][4]) / eth[-2][4]) * 100
            btc_moves = [abs((btc[i][4]-btc[i-1][4])/btc[i-1][4])*100 for i in range(1,5)]
            return {"btc_trend": "🟢" if btc_ch > -0.3 else "🔴",
                    "btc_ch": btc_ch, "btc_p": btc[-1][4],
                    "alt_power": "🚀" if eth_ch-btc_ch > 0.5 else "⚓️",
                    "alt_ch": eth_ch - btc_ch,
                    "btc_vol": np.mean(btc_moves)}
        except Exception as e:
            logging.warning(f"market_context attempt {attempt+1}: {e}"); time.sleep(5)
    return {"btc_trend":"⚪️","btc_ch":0,"btc_p":0,"alt_power":"⚪️","alt_ch":0,"btc_vol":1.0}


def adaptive_threshold(base, btc_vol, is_priority):
    t = base
    if btc_vol > 3.0:   t += 2
    elif btc_vol < 0.5: t -= 1
    if is_priority:     t -= 1
    return max(t, 3)


# ─────────────────────────────────────────────
# ОСНОВНОЙ ЦИКЛ
# ─────────────────────────────────────────────
def analyst_loop():
    sent_signals = {}   # key → timestamp (СИГНАЛ)
    sent_attention = {} # key → timestamp (ВНИМАНИЕ)
    logging.info("Аналитик v6.0 запущен.")

    try:
        exchange.load_markets()
        logging.info("Рынки загружены.")
    except Exception as e:
        logging.error(f"Ошибка загрузки рынков: {e}")

    markets_reload_ts = time.time()

    while True:
        try:
            if time.time() - markets_reload_ts > 3600:
                try:
                    exchange.load_markets(); markets_reload_ts = time.time()
                except Exception as e:
                    logging.error(f"Перезагрузка рынков: {e}")

            ctx = get_market_context()

            try:
                tickers = exchange.fetch_tickers()
            except Exception as e:
                logging.error(f"fetch_tickers: {e}"); time.sleep(60); continue

            active_swaps = [s for s, m in exchange.markets.items()
                            if m.get('active') and m.get('type') == 'swap']

            vol_data = sorted(
                [{'s': s, 'v': tickers[s].get('quoteVolume', 0)}
                 for s in active_swaps if s in tickers],
                key=lambda x: x['v'], reverse=True)

            top250  = [x['s'] for x in vol_data[:250]]
            symbols = list(dict.fromkeys(
                [w for w in WATCHLIST if w in tickers] + top250))

            for symbol in symbols:
                try:
                    # ── Фильтр объёма монеты ──
                    vol_24h = tickers.get(symbol, {}).get('quoteVolume', 0) or 0
                    if vol_24h < MIN_VOLUME_ATTENTION:
                        continue  # меньше $2M — пропускаем совсем

                    ohlcv = None
                    for attempt in range(2):
                        try:
                            ohlcv = exchange.fetch_ohlcv(symbol, '4h', limit=120); break
                        except ccxt.NetworkError as e:
                            if attempt == 0: logging.warning(f"retry {symbol}: {e}"); time.sleep(3)
                            else: raise
                    if ohlcv is None or len(ohlcv) < 80:
                        continue

                    is_wl         = symbol in WATCHLIST
                    current_price = ohlcv[-1][4]
                    closed        = ohlcv[:-1]
                    closes_closed = [x[4] for x in closed]

                    last_closed_ts = closed[-1][0]
                    candle_id      = last_closed_ts // 14_400_000
                    l_key  = f"{symbol}_{candle_id}_l"
                    s_key  = f"{symbol}_{candle_id}_s"
                    la_key = f"{symbol}_{candle_id}_la"  # attention long
                    sa_key = f"{symbol}_{candle_id}_sa"  # attention short

                    all_done = (l_key in sent_signals and s_key in sent_signals
                                and la_key in sent_attention and sa_key in sent_attention)
                    if all_done:
                        continue

                    price_ch = ((closed[-1][4] - closed[-2][4]) / closed[-2][4] * 100) \
                               if len(closed) >= 2 else 0.0

                    # ══════════════════════════════
                    # ИНДИКАТОРЫ
                    # ══════════════════════════════
                    rsi = calculate_rsi_wilder(closes_closed)
                    mfi = calculate_mfi(closed)
                    atr = calculate_atr(closed)

                    c_high = closed[-1][2]; c_low = closed[-1][3]; c_close = closed[-1][4]
                    buy_pressure = (c_close - c_low) / (c_high - c_low) if c_high != c_low else 0.5
                    imb = (buy_pressure - 0.5) * 200

                    # Объём расширенный
                    vol_score, vol_label, v_rel, v_zscore, avg_buy_pct, vol_detail = \
                        get_volume_info(closed, vol_24h)

                    # ── SSMA — ГЛАВНЫЕ ВОРОТА ──
                    ssma_val, ssma_trend, ssma_slope = calculate_ssma(closed, period=24)
                    long_gate  = ssma_allows_long(ssma_val, ssma_trend, ssma_slope, current_price)
                    short_gate = ssma_allows_short(ssma_val, ssma_trend, ssma_slope, current_price)

                    ssma_label = ""
                    if ssma_val:
                        icon = "📈" if 'bull' in ssma_trend else "📉"
                        ssma_label = f"{icon} SSMA: {ssma_val:.4g} ({ssma_slope:+.2f}%/св)"

                    # BB ATR
                    bb_signal, bb_dist, bb_upper, bb_lower, bb_basis = \
                        calculate_bb_outside_atr(closed)

                    # Swing HILO
                    sw_low_pct, sw_high_pct, near_sw_low, near_sw_high, sw_low, sw_high = \
                        calculate_swing_hilo(closed, swing_bars=34)

                    # CVD
                    cvd_level, cvd_total = calc_cvd_level(closed)
                    cvd_div_l = get_cvd_divergence(closed, 'long')
                    cvd_div_s = get_cvd_divergence(closed, 'short')

                    # RSI div
                    rsi_div_bull, rsi_div_bear = calc_rsi_divergence(closed)

                    # Молот
                    hammer_l = check_hammer(ohlcv, 'long')
                    hammer_s = check_hammer(ohlcv, 'short')

                    # DeMark
                    m9_l, m9_perfect_l, m13_l = update_td_counters(ohlcv, 'long')
                    m9_s, m9_perfect_s, m13_s = update_td_counters(ohlcv, 'short')

                    # Pivot
                    supports, resistances = get_pivot_levels(closed)
                    sup = supports[0]    if supports    else min(x[3] for x in closed[-60:])
                    res = resistances[0] if resistances else max(x[2] for x in closed[-60:])

                    # ATR цели
                    stop_k, take_k, rr_str = dynamic_atr_multipliers(ctx['btc_vol'])
                    if atr:
                        stop_l = current_price - stop_k * atr; target_l = current_price + take_k * atr
                        stop_s = current_price + stop_k * atr; target_s = current_price - take_k * atr
                    else:
                        stop_l = stop_s = target_l = target_s = None

                    # Wyckoff
                    wyckoff_phase, wyckoff_long_ok, wyckoff_short_ok, wyckoff_label, wyckoff_strength = \
                        detect_wyckoff_phase(closed, sw_low or sup, sw_high or res, cvd_level)

                    # Фильтр после большого движения
                    after_dump, after_pump, big_move_pct, candles_since_move = \
                        check_post_move_filter(ohlcv, lookback=12)

                    if after_dump and not wyckoff_long_ok:
                        wyckoff_long_ok  = True; wyckoff_short_ok = False
                        wyckoff_label    = f"🌱 После дампа -{big_move_pct:.0f}%"
                    if after_pump and not wyckoff_short_ok:
                        wyckoff_short_ok = True; wyckoff_long_ok  = False
                        wyckoff_label    = f"🔝 После памп +{big_move_pct:.0f}%"

                    # OI и фандинг
                    has_any = (cvd_div_l or cvd_div_s or hammer_l or hammer_s or
                               m9_l or m9_s or m13_l or m13_s or
                               rsi_div_bull or rsi_div_bear or vol_score >= 2 or
                               near_sw_low or near_sw_high or bb_signal != 'neutral')

                    oi_chg=0.0; oi_signal='neutral'; oi_label='⚪️ OI: нет данных'
                    fr_val=0.0; fr_signal='neutral'; fr_label='⚪️ Фандинг: нет данных'

                    if has_any:
                        oi_chg, oi_signal, oi_label = get_oi_data_from_ticker(symbol, tickers, price_ch)
                        fr_val, fr_signal, fr_label  = get_funding_signal(symbol)

                    wl_badge = " ⭐️" if is_wl else ""
                    cvd_emoji = {"bull":"🟢","bull_div":"🟢✨","bear":"🔴","bear_div":"🔴✨"}.get(cvd_level,"⚪️")

                    # ══════════════════════════════════════════════════
                    # ФУНКЦИЯ СКОРИНГА (общая для ВНИМАНИЕ и СИГНАЛ)
                    # ══════════════════════════════════════════════════
                    def calc_score(mode):
                        score = 0; details = []

                        # Wyckoff бонус
                        if mode == 'long':
                            if wyckoff_phase == 'accumulation' and wyckoff_strength >= 2:
                                score += 2; details.append("🟢 Накопление")
                            elif wyckoff_phase == 'accumulation':
                                score += 1; details.append("🟢 Накопление")
                        else:
                            if wyckoff_phase == 'distribution' and wyckoff_strength >= 2:
                                score += 2; details.append("🔴 Распределение")
                            elif wyckoff_phase == 'distribution':
                                score += 1; details.append("🔴 Распределение")

                        # BB ATR
                        if mode == 'long' and bb_signal == 'oversold':
                            score += 2; details.append(f"📉 BB ATR Перепродан ({bb_dist:.1f}%)")
                        if mode == 'short' and bb_signal == 'overbought':
                            score += 2; details.append(f"📈 BB ATR Перекуплен ({bb_dist:.1f}%)")

                        # Swing HILO
                        if mode == 'long' and near_sw_low:
                            score += 2; details.append(f"🏔️ Swing Low (+{sw_low_pct:.1f}%)")
                        if mode == 'short' and near_sw_high:
                            score += 2; details.append(f"🏔️ Swing High (-{sw_high_pct:.1f}%)")

                        # SSMA бонус (не ворота — ворота выше)
                        if mode == 'long' and ssma_trend == 'bull_strong':
                            score += 1; details.append(f"📈 SSMA Бык ({ssma_slope:+.2f}%)")
                        if mode == 'short' and ssma_trend == 'bear_strong':
                            score += 1; details.append(f"📉 SSMA Медведь ({ssma_slope:+.2f}%)")

                        # OI
                        if mode == 'long':
                            if oi_signal == 'bull':   score += 2; details.append(f"💹 OI +{oi_chg:.1f}%")
                            elif oi_signal == 'bear': score -= 1
                        else:
                            if oi_signal == 'bear':   score += 2; details.append(f"💹 OI +{oi_chg:.1f}%")
                            elif oi_signal == 'bull': score -= 1

                        # CVD дивергенция
                        if mode == 'long' and cvd_div_l:
                            score += 2; details.append("🔥 CVD Дивер")
                        if mode == 'short' and cvd_div_s:
                            score += 2; details.append("🔥 CVD Дивер")

                        # RSI дивергенция
                        if mode == 'long'  and rsi_div_bull: score += 2; details.append("📉 RSI Дивер")
                        if mode == 'short' and rsi_div_bear: score += 2; details.append("📈 RSI Дивер")

                        # DeMark
                        if mode == 'long':
                            if m9_l:
                                pts = 3 if m9_perfect_l else 2
                                score += pts; details.append("⏱ M9✨" if m9_perfect_l else "⏱ M9")
                            if m13_l: score += 3; details.append("⏱ M13")
                        else:
                            if m9_s:
                                pts = 3 if m9_perfect_s else 2
                                score += pts; details.append("⏱ M9✨" if m9_perfect_s else "⏱ M9")
                            if m13_s: score += 3; details.append("⏱ M13")

                        # CVD уровень
                        if mode == 'long'  and cvd_level in ('bull','bull_div'): score += 1; details.append("📍 CVD")
                        if mode == 'short' and cvd_level in ('bear','bear_div'): score += 1; details.append("📍 CVD")

                        # Pivot
                        if mode == 'long'  and current_price <= sup * 1.015: score += 2; details.append("🧱 Pivot Sup")
                        if mode == 'short' and current_price >= res * 0.985: score += 2; details.append("🧱 Pivot Res")

                        # Объём
                        if vol_score > 0: score += vol_score; details.append(vol_label)

                        # Фандинг
                        if mode == 'long'  and fr_signal == 'bull': score += 1; details.append(f"💸 FR {fr_val:.3f}%")
                        if mode == 'short' and fr_signal == 'bear': score += 1; details.append(f"💸 FR {fr_val:.3f}%")

                        # Молот
                        if mode == 'long'  and hammer_l: score += 1; details.append("⚓️ Фитиль")
                        if mode == 'short' and hammer_s: score += 1; details.append("🏹 Фитиль↑")

                        # RSI/MFI экстремум
                        if mode == 'long'  and rsi < 30:  score += 1; details.append(f"📉 RSI {rsi:.0f}")
                        if mode == 'long'  and mfi < 20:  score += 1; details.append(f"💰 MFI {mfi:.0f}")
                        if mode == 'short' and rsi > 70:  score += 1; details.append(f"📈 RSI {rsi:.0f}")
                        if mode == 'short' and mfi > 80:  score += 1; details.append(f"💰 MFI {mfi:.0f}")

                        return max(score, 0), details

                    # ══════════════════════════════════════════════════
                    # ЛОНГ — ВНИМАНИЕ
                    # Условие: цена в пределах 2% от Swing Low
                    # SSMA ворота обязательны
                    # ══════════════════════════════════════════════════
                    if (la_key not in sent_attention
                            and near_sw_low
                            and long_gate
                            and wyckoff_phase != 'ranging'
                            and vol_24h >= MIN_VOLUME_ATTENTION):

                        a_score, a_details = calc_score('long')
                        rr_ok_a, tgt_pct_a, stp_pct_a, rr_a = check_rr(
                            current_price, target_l, stop_l, 'long',
                            MIN_TARGET_PCT_ATTENTION, MAX_STOP_PCT)

                        if a_score >= SCORE_ATTENTION and rr_ok_a:
                            msg = (
                                f"🔔 <b>ВНИМАНИЕ ЛОНГ 4H ({a_score}/10){wl_badge}</b>\n"
                                f"Монета: <b>{symbol}</b>\n"
                                f"Цена: <code>{current_price:.6g}</code> | "
                                f"Swing Low: +{sw_low_pct:.1f}%\n"
                                f"Сигналы: {', '.join(a_details)}\n"
                                f"───────────────────\n"
                                f"🌊 Wyckoff: <b>{wyckoff_label}</b>\n"
                                f"{ssma_label}\n"
                                f"───────────────────\n"
                                f"📊 RSI: {rsi:.1f} | MFI: {mfi:.1f}\n"
                                f"{vol_detail}\n"
                                f"{cvd_emoji} CVD: <b>{cvd_level}</b>\n"
                                f"{oi_label}\n"
                                f"───────────────────\n"
                                f"🎯 Цель: <code>{target_l:.6g}</code> "
                                f"(+{tgt_pct_a:.1f}%) | R/R {rr_a:.1f}\n"
                                f"🛑 Стоп: <code>{stop_l:.6g}</code> "
                                f"(-{stp_pct_a:.1f}%)\n"
                                f"───────────────────\n"
                                f"👑 BTC: {ctx['btc_trend']} {ctx['btc_ch']:.1f}%\n"
                                f"{build_tv_link(symbol)}"
                            )
                            send_msg(msg)
                            sent_attention[la_key] = time.time()
                            bot_status["attention_sent"] += 1
                            logging.info(f"ВНИМАНИЕ ЛОНГ: {symbol} score={a_score} sw_low={sw_low_pct:.1f}%")

                    # ══════════════════════════════════════════════════
                    # ШОРТ — ВНИМАНИЕ
                    # Условие: цена в пределах 2% от Swing High
                    # ══════════════════════════════════════════════════
                    if (sa_key not in sent_attention
                            and near_sw_high
                            and short_gate
                            and wyckoff_phase != 'ranging'
                            and vol_24h >= MIN_VOLUME_ATTENTION):

                        a_score, a_details = calc_score('short')
                        rr_ok_a, tgt_pct_a, stp_pct_a, rr_a = check_rr(
                            current_price, target_s, stop_s, 'short',
                            MIN_TARGET_PCT_ATTENTION, MAX_STOP_PCT)

                        if a_score >= SCORE_ATTENTION and rr_ok_a:
                            msg = (
                                f"🔔 <b>ВНИМАНИЕ ШОРТ 4H ({a_score}/10){wl_badge}</b>\n"
                                f"Монета: <b>{symbol}</b>\n"
                                f"Цена: <code>{current_price:.6g}</code> | "
                                f"Swing High: -{sw_high_pct:.1f}%\n"
                                f"Сигналы: {', '.join(a_details)}\n"
                                f"───────────────────\n"
                                f"🌊 Wyckoff: <b>{wyckoff_label}</b>\n"
                                f"{ssma_label}\n"
                                f"───────────────────\n"
                                f"📊 RSI: {rsi:.1f} | MFI: {mfi:.1f}\n"
                                f"{vol_detail}\n"
                                f"{cvd_emoji} CVD: <b>{cvd_level}</b>\n"
                                f"{oi_label}\n"
                                f"───────────────────\n"
                                f"🎯 Цель: <code>{target_s:.6g}</code> "
                                f"(-{tgt_pct_a:.1f}%) | R/R {rr_a:.1f}\n"
                                f"🛑 Стоп: <code>{stop_s:.6g}</code> "
                                f"(+{stp_pct_a:.1f}%)\n"
                                f"───────────────────\n"
                                f"👑 BTC: {ctx['btc_trend']} {ctx['btc_ch']:.1f}%\n"
                                f"{build_tv_link(symbol)}"
                            )
                            send_msg(msg)
                            sent_attention[sa_key] = time.time()
                            bot_status["attention_sent"] += 1
                            logging.info(f"ВНИМАНИЕ ШОРТ: {symbol} score={a_score} sw_high={sw_high_pct:.1f}%")

                    # ══════════════════════════════════════════════════
                    # ЛОНГ — СИГНАЛ
                    # Дополнительно к ВНИМАНИЕ:
                    #   - скор 7+
                    #   - OI обязателен (bull или squeeze_dn)
                    #   - R/R 3:1, цель 5%+, стоп макс 3%
                    #   - объём $5M+
                    #   - Wyckoff не диапазон
                    # ══════════════════════════════════════════════════
                    if (l_key not in sent_signals
                            and long_gate
                            and wyckoff_long_ok
                            and wyckoff_phase != 'ranging'
                            and vol_24h >= MIN_VOLUME_SIGNAL):

                        score, details = calc_score('long')

                        # OI обязателен для СИГНАЛ
                        oi_ok_l = oi_signal in ('bull', 'squeeze_dn')

                        # R/R фильтр
                        rr_ok, tgt_pct, stp_pct, rr_val = check_rr(
                            current_price, target_l, stop_l, 'long',
                            MIN_TARGET_PCT_SIGNAL, MAX_STOP_PCT)

                        # OI-ворота: без OI максимум 4
                        if not oi_ok_l and not (m13_l or cvd_div_l):
                            score = min(score, 4)

                        # Wyckoff OI смягчение
                        if wyckoff_phase == 'accumulation' and not oi_ok_l:
                            score = min(score, 5)

                        # ВЕТО
                        btc_red     = ctx['btc_trend'] == "🔴"
                        is_strong_l = oi_signal == 'bull' or m13_l or (v_rel > 5 and imb > 60)
                        veto_macro  = btc_red and not is_strong_l and not after_dump
                        veto_dump   = v_rel > 10 and imb < -60 and price_ch < -5 and not after_dump
                        veto_ob     = (rsi > 75 and mfi > 80) and not m13_l

                        if (score >= SCORE_SIGNAL
                                and oi_ok_l
                                and rr_ok
                                and not veto_macro
                                and not veto_dump
                                and not veto_ob):

                            status = "⚡️ СИЛЬНЕЕ РЫНКА" if (is_strong_l and btc_red) else "✅ ТРЕНД"

                            msg = (
                                f"🚨 <b>СИГНАЛ ЛОНГ 4H ({score}/10){wl_badge}</b>\n"
                                f"Монета: <b>{symbol}</b> | {status}\n"
                                f"Цена: <code>{current_price:.6g}</code>\n"
                                f"Сигналы: {', '.join(details)}\n"
                                f"───────────────────\n"
                                f"🌊 Wyckoff: <b>{wyckoff_label}</b>\n"
                                f"{'📉 BB ATR: Перепродан' if bb_signal=='oversold' else ''}\n"
                                f"{'🏔️ Swing Low: +' + f'{sw_low_pct:.1f}%' if near_sw_low else ''}\n"
                                f"{ssma_label}\n"
                                f"───────────────────\n"
                                f"📊 RSI: <b>{rsi:.1f}</b> | MFI: <b>{mfi:.1f}</b>\n"
                                f"⚖️ Дисбаланс: {'🟢' if imb>0 else '🔴'} <b>{abs(imb):.0f}%</b>\n"
                                f"{vol_detail}\n"
                                f"{cvd_emoji} CVD: <b>{cvd_level}</b>\n"
                                f"{oi_label}\n"
                                f"{fr_label}\n"
                                f"───────────────────\n"
                                f"🎯 Цель: <code>{target_l:.6g}</code> (+{tgt_pct:.1f}%)\n"
                                f"🛑 Стоп: <code>{stop_l:.6g}</code> (-{stp_pct:.1f}%)\n"
                                f"⚡️ ATR: <code>{atr:.6g}</code> | R/R {rr_val:.1f}\n"
                                f"───────────────────\n"
                                f"🌍 BTC: {ctx['btc_trend']} {ctx['btc_ch']:.1f}% "
                                f"({ctx['btc_p']:.0f}) | Alt: {ctx['alt_power']} {ctx['alt_ch']:.2f}%\n"
                                f"{build_tv_link(symbol)}"
                            )
                            send_msg(msg)
                            sent_signals[l_key] = time.time()
                            bot_status["signals_sent"] += 1
                            logging.info(f"СИГНАЛ ЛОНГ: {symbol} score={score} "
                                         f"OI={oi_signal} RR={rr_val:.1f} tgt={tgt_pct:.1f}%")

                    # ══════════════════════════════════════════════════
                    # ШОРТ — СИГНАЛ
                    # ══════════════════════════════════════════════════
                    if (s_key not in sent_signals
                            and short_gate
                            and wyckoff_short_ok
                            and wyckoff_phase != 'ranging'
                            and vol_24h >= MIN_VOLUME_SIGNAL):

                        score, details = calc_score('short')

                        oi_ok_s = oi_signal in ('bear', 'squeeze_up')

                        rr_ok, tgt_pct, stp_pct, rr_val = check_rr(
                            current_price, target_s, stop_s, 'short',
                            MIN_TARGET_PCT_SIGNAL, MAX_STOP_PCT)

                        if not oi_ok_s and not (m13_s or cvd_div_s):
                            score = min(score, 4)
                        if wyckoff_phase == 'distribution' and not oi_ok_s:
                            score = min(score, 5)

                        btc_green    = ctx['btc_trend'] == "🟢"
                        is_strong_s  = oi_signal == 'bear' or m13_s or (v_rel > 5 and imb < -60)
                        veto_macro   = btc_green and not is_strong_s and not after_pump
                        veto_pump    = v_rel > 8 and imb > 60 and price_ch > 3 and not after_pump
                        veto_os      = (rsi < 25 and mfi < 20) and not m13_s

                        if (score >= SCORE_SIGNAL
                                and oi_ok_s
                                and rr_ok
                                and not veto_macro
                                and not veto_pump
                                and not veto_os):

                            status = "⚡️ ПРОТИВ РЫНКА" if (is_strong_s and btc_green) else "🔻 ШОРТ"

                            msg = (
                                f"🚨 <b>СИГНАЛ ШОРТ 4H ({score}/10){wl_badge}</b>\n"
                                f"Монета: <b>{symbol}</b> | {status}\n"
                                f"Цена: <code>{current_price:.6g}</code>\n"
                                f"Сигналы: {', '.join(details)}\n"
                                f"───────────────────\n"
                                f"🌊 Wyckoff: <b>{wyckoff_label}</b>\n"
                                f"{'📈 BB ATR: Перекуплен' if bb_signal=='overbought' else ''}\n"
                                f"{'🏔️ Swing High: -' + f'{sw_high_pct:.1f}%' if near_sw_high else ''}\n"
                                f"{ssma_label}\n"
                                f"───────────────────\n"
                                f"📊 RSI: <b>{rsi:.1f}</b> | MFI: <b>{mfi:.1f}</b>\n"
                                f"⚖️ Дисбаланс: {'🔴' if imb>0 else '🟢'} <b>{abs(imb):.0f}%</b>\n"
                                f"{vol_detail}\n"
                                f"{cvd_emoji} CVD: <b>{cvd_level}</b>\n"
                                f"{oi_label}\n"
                                f"{fr_label}\n"
                                f"───────────────────\n"
                                f"🎯 Цель: <code>{target_s:.6g}</code> (-{tgt_pct:.1f}%)\n"
                                f"🛑 Стоп: <code>{stop_s:.6g}</code> (+{stp_pct:.1f}%)\n"
                                f"⚡️ ATR: <code>{atr:.6g}</code> | R/R {rr_val:.1f}\n"
                                f"───────────────────\n"
                                f"🌍 BTC: {ctx['btc_trend']} {ctx['btc_ch']:.1f}% "
                                f"({ctx['btc_p']:.0f}) | Alt: {ctx['alt_power']} {ctx['alt_ch']:.2f}%\n"
                                f"{build_tv_link(symbol)}"
                            )
                            send_msg(msg)
                            sent_signals[s_key] = time.time()
                            bot_status["signals_sent"] += 1
                            logging.info(f"СИГНАЛ ШОРТ: {symbol} score={score} "
                                         f"OI={oi_signal} RR={rr_val:.1f} tgt={tgt_pct:.1f}%")

                    time.sleep(0.15)

                except ccxt.RateLimitExceeded:
                    logging.warning(f"Rate limit {symbol}, пауза 30с"); time.sleep(30)
                except ccxt.NetworkError as e:
                    logging.error(f"Network {symbol}: {e}")
                except Exception as e:
                    logging.error(f"Ошибка {symbol}: {e}")

            now = time.time()
            sent_signals   = {k: v for k, v in sent_signals.items()   if now - v < 86400}
            sent_attention = {k: v for k, v in sent_attention.items() if now - v < 86400}
            bot_status["iterations"]    += 1
            bot_status["last_iteration"] = datetime.now().strftime('%H:%M:%S')
            logging.info(f"Итерация. Символов: {len(symbols)} | "
                         f"Сигналов: {bot_status['signals_sent']} | "
                         f"Внимание: {bot_status['attention_sent']} | "
                         f"BTC vol: {ctx['btc_vol']:.2f}%")
            time.sleep(300)

        except ccxt.NetworkError as e:
            logging.error(f"Глобальная сеть: {e}"); bot_status["errors"] += 1; time.sleep(60)
        except Exception as e:
            logging.error(f"Критическая ошибка: {e}"); bot_status["errors"] += 1; time.sleep(60)


@app.route('/health')
def health():
    uptime = str(datetime.now() - datetime.fromisoformat(bot_status["started_at"])).split('.')[0]
    return (f"✅ OK | v6.0\n"
            f"Uptime: {uptime}\n"
            f"Итераций: {bot_status['iterations']}\n"
            f"Ошибок: {bot_status['errors']}\n"
            f"Сигналов 🚨: {bot_status['signals_sent']}\n"
            f"Внимание 🔔: {bot_status['attention_sent']}\n"
            f"Последняя итерация: {bot_status['last_iteration']}")


def keepalive_loop():
    time.sleep(30)
    port         = int(os.environ.get("PORT", 10000))
    external_url = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
    local_url    = f"http://localhost:{port}/health"
    while True:
        for url in ([f"{external_url}/health"] if external_url else []) + [local_url]:
            try:
                r = requests.get(url, timeout=15)
                logging.info(f"Keepalive [{url}]: {r.status_code}"); break
            except Exception as e:
                logging.warning(f"Keepalive [{url}]: {e}")
        time.sleep(240)


def watchdog():
    time.sleep(60)
    while True:
        global analyst_thread
        if not analyst_thread.is_alive():
            logging.error("analyst_loop упал, перезапуск...")
            bot_status["errors"] += 1
            analyst_thread = threading.Thread(target=analyst_loop, daemon=True, name="analyst")
            analyst_thread.start()
        time.sleep(60)


analyst_thread = threading.Thread(target=analyst_loop, daemon=True, name="analyst")
analyst_thread.start()
threading.Thread(target=keepalive_loop, daemon=True, name="keepalive").start()
threading.Thread(target=watchdog,       daemon=True, name="watchdog").start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
