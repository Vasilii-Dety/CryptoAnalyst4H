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
    return f"✅ АНАЛИТИК v5.0 АКТИВЕН. Время: {datetime.now().strftime('%H:%M:%S')}"

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

bot_status = {
    "started_at":     datetime.now().isoformat(),
    "last_iteration": None,
    "iterations":     0,
    "errors":         0,
    "signals_sent":   0,
}

oi_cache: dict = {}


# ─────────────────────────────────────────────
# SSMA (Smoothed Moving Average) — из PpSignal
# period=24, source=(H+L+C)/3
# Направление SSMA определяет глобальный тренд
# ─────────────────────────────────────────────
def calculate_ssma(ohlcv, period=24):
    """
    Smoothed MA (Wilder's SMMA) на типичной цене (H+L+C)/3
    Если цена > SSMA → восходящий тренд (бычий контекст)
    Если цена < SSMA → нисходящий тренд (медвежий контекст)
    Возвращает: (ssma_value, trend_direction, slope)
    """
    if len(ohlcv) < period + 10:
        return None, 'neutral', 0.0

    typical = [(c[2] + c[3] + c[4]) / 3 for c in ohlcv]

    # Инициализация: простое среднее первых period баров
    smma = np.mean(typical[:period])
    smma_history = [smma]

    for i in range(period, len(typical)):
        smma = (smma * (period - 1) + typical[i]) / period
        smma_history.append(smma)

    current_smma = smma_history[-1]
    prev_smma    = smma_history[-2] if len(smma_history) >= 2 else smma_history[-1]
    current_price = ohlcv[-1][4]

    slope = (current_smma - prev_smma) / prev_smma * 100

    if current_price > current_smma and slope > 0:
        trend = 'bull_strong'
    elif current_price > current_smma and slope <= 0:
        trend = 'bull_weak'
    elif current_price < current_smma and slope < 0:
        trend = 'bear_strong'
    else:
        trend = 'bear_weak'

    return current_smma, trend, slope


# ─────────────────────────────────────────────
# BB OUTSIDE ATR — из PpSignal
# length=20, mult=0.618, ATR mode, bars=55, StdDev=0.712
# Определяет: цена вышла за "нормальную" волатильность
# Используется как фильтр перепроданности/перекупленности
# ─────────────────────────────────────────────
def calculate_bb_outside_atr(ohlcv,
                               bb_length=55,
                               bb_std=0.712,
                               atr_length=20,
                               atr_mult=0.618,
                               tilson_length=10,
                               tilson_factor=0.3):  # factor = 3 * 0.10 = 0.3
    """
    BB Outside ATR — аналог PpSignal логики:
    1. Считаем ATR-полосу вокруг Tilson T3 MA
    2. Если цена ниже нижней границы → перепроданность (потенциальный лонг)
    3. Если цена выше верхней границы → перекупленность (потенциальный шорт)

    Tilson T3: сглаженная MA, реагирует быстрее EMA
    Возвращает: (signal, distance_pct, upper, lower, basis)
      signal: 'oversold' / 'overbought' / 'neutral'
    """
    if len(ohlcv) < bb_length + atr_length + 10:
        return 'neutral', 0.0, None, None, None

    closes = [c[4] for c in ohlcv]
    highs  = [c[2] for c in ohlcv]
    lows   = [c[3] for c in ohlcv]

    # ── Tilson T3 MA (быстрый трендовый фильтр) ──
    f  = tilson_factor
    c1 = -(f**3)
    c2 = 3 * f**2 + 3 * f**3
    c3 = -6 * f**2 - 3 * f - 3 * f**3
    c4 = 1 + 3 * f + f**3 + 3 * f**2

    def ema(data, p):
        e = data[0]
        k = 2 / (p + 1)
        for v in data[1:]:
            e = v * k + e * (1 - k)
        return e

    # Нужны 6 EMA для T3
    t_len = tilson_length
    if len(closes) < t_len * 6:
        basis = np.mean(closes[-bb_length:])
    else:
        e1 = ema(closes, t_len)
        # Для T3 используем скользящие EMA от EMA
        # Упрощение: итеративная цепочка на последних данных
        chunk = closes[-min(len(closes), bb_length + t_len * 6):]
        e1_arr = []
        e_v = chunk[0]
        k   = 2 / (t_len + 1)
        for v in chunk:
            e_v = v * k + e_v * (1 - k)
            e1_arr.append(e_v)

        e2_arr = []
        e_v = e1_arr[0]
        for v in e1_arr:
            e_v = v * k + e_v * (1 - k)
            e2_arr.append(e_v)

        e3_arr = []
        e_v = e2_arr[0]
        for v in e2_arr:
            e_v = v * k + e_v * (1 - k)
            e3_arr.append(e_v)

        e4_arr = []
        e_v = e3_arr[0]
        for v in e3_arr:
            e_v = v * k + e_v * (1 - k)
            e4_arr.append(e_v)

        e5_arr = []
        e_v = e4_arr[0]
        for v in e4_arr:
            e_v = v * k + e_v * (1 - k)
            e5_arr.append(e_v)

        e6_arr = []
        e_v = e5_arr[0]
        for v in e5_arr:
            e_v = v * k + e_v * (1 - k)
            e6_arr.append(e_v)

        basis = c1 * e6_arr[-1] + c2 * e5_arr[-1] + c3 * e4_arr[-1] + c4 * e3_arr[-1]

    # ── ATR полоса ──
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
    dist_lower = (lower - current) / lower * 100  # положительное = ниже нижней
    dist_upper = (current - upper) / upper * 100  # положительное = выше верхней

    if current < lower:
        return 'oversold', dist_lower, upper, lower, basis
    elif current > upper:
        return 'overbought', dist_upper, upper, lower, basis
    else:
        return 'neutral', 0.0, upper, lower, basis


# ─────────────────────────────────────────────
# SWING HILO — из PpSignal (34 bars lookback)
# Определяет структурные максимумы и минимумы
# Используется для: цена у основания = зона лонга
#                   цена у вершины = зона шорта
# ─────────────────────────────────────────────
def calculate_swing_hilo(ohlcv, swing_bars=34):
    """
    Swing High/Low — структурные экстремумы.
    Свинг-лоу: бар с минимумом ниже swing_bars баров до и после
    Свинг-хай: бар с максимумом выше swing_bars баров до и после

    Возвращает:
      swing_low_pct:  насколько % цена выше последнего свинг-лоу
      swing_high_pct: насколько % цена ниже последнего свинг-хай
      near_swing_low:  True если цена в пределах 2% от свинг-лоу
      near_swing_high: True если цена в пределах 2% от свинг-хай
      last_swing_low:  уровень последнего свинг-лоу
      last_swing_high: уровень последнего свинг-хай
    """
    if len(ohlcv) < swing_bars * 2 + 1:
        return 100.0, 100.0, False, False, None, None

    highs  = [c[2] for c in ohlcv]
    lows   = [c[3] for c in ohlcv]
    n      = len(ohlcv)

    # Ищем только исторические свинги (не текущий бар)
    last_swing_low  = None
    last_swing_high = None

    for i in range(swing_bars, n - swing_bars - 1):
        # Свинг-лоу: минимум ниже всех соседей
        is_swing_low = (lows[i] == min(lows[i - swing_bars:i + swing_bars + 1]))
        if is_swing_low:
            last_swing_low = lows[i]

        # Свинг-хай: максимум выше всех соседей
        is_swing_high = (highs[i] == max(highs[i - swing_bars:i + swing_bars + 1]))
        if is_swing_high:
            last_swing_high = highs[i]

    current = ohlcv[-1][4]

    if last_swing_low is None:
        last_swing_low = min(lows[-60:])
    if last_swing_high is None:
        last_swing_high = max(highs[-60:])

    swing_low_pct  = (current - last_swing_low)  / last_swing_low  * 100
    swing_high_pct = (last_swing_high - current) / last_swing_high * 100

    # "Рядом" = в пределах 3% от свинг-уровня
    near_swing_low  = swing_low_pct  <= 3.0
    near_swing_high = swing_high_pct <= 3.0

    return swing_low_pct, swing_high_pct, near_swing_low, near_swing_high, last_swing_low, last_swing_high


# ─────────────────────────────────────────────
# WYCKOFF ФАЗЫ — ключевое улучшение
#
# Определяем в какой фазе находится монета:
#   ACCUMULATION (накопление) → ищем ЛОНГ
#   DISTRIBUTION (распределение) → ищем ШОРТ
#   MARKUP (рост) → лонг в откатах
#   MARKDOWN (падение) → шорт в откатах
#
# Логика на основе:
#   1. Относительное положение цены (где мы сейчас в диапазоне)
#   2. Объёмный профиль (высокий объём на лоях = накопление)
#   3. CVD (покупатели поглощают продавцов = накопление)
#   4. Волатильность снижается после дампа = Spring фаза
# ─────────────────────────────────────────────
def detect_wyckoff_phase(ohlcv, swing_low, swing_high, cvd_level):
    """
    Определяет Wyckoff фазу и разрешённые направления торговли.

    Возвращает:
      phase: 'accumulation' / 'distribution' / 'markup' / 'markdown' / 'ranging'
      long_allowed:  bool
      short_allowed: bool
      phase_label:   читаемое описание
      phase_strength: 0-3 (уверенность)
    """
    if len(ohlcv) < 60:
        return 'ranging', True, True, '⚪️ Нет данных', 0

    closes = [c[4] for c in ohlcv]
    volumes = [c[5] for c in ohlcv]
    current = closes[-1]

    # ── Положение в диапазоне ──
    price_range = swing_high - swing_low
    if price_range <= 0:
        price_range = swing_high * 0.01

    # Где цена относительно диапазона (0% = лоу, 100% = хай)
    position_pct = (current - swing_low) / price_range * 100

    # ── Объёмный анализ по зонам диапазона ──
    # Разбиваем историю на нижнюю треть / верхнюю треть
    lower_third = swing_low + price_range * 0.33
    upper_third = swing_low + price_range * 0.67

    vol_at_lows  = np.mean([volumes[i] for i in range(len(ohlcv))
                             if closes[i] <= lower_third] or [0])
    vol_at_highs = np.mean([volumes[i] for i in range(len(ohlcv))
                             if closes[i] >= upper_third] or [0])
    vol_avg = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)

    # ── Волатильность — снижение = конец движения ──
    recent_ranges  = [ohlcv[i][2] - ohlcv[i][3] for i in range(-10, 0)]
    earlier_ranges = [ohlcv[i][2] - ohlcv[i][3] for i in range(-30, -10)]
    vol_compression = (np.mean(recent_ranges) / np.mean(earlier_ranges)) if np.mean(earlier_ranges) > 0 else 1.0

    # ── Тренд последних 20 свечей ──
    ma20_start = np.mean(closes[-25:-20]) if len(closes) >= 25 else closes[0]
    ma20_end   = np.mean(closes[-5:])
    trend_20   = (ma20_end - ma20_start) / ma20_start * 100

    # ═══════════════════════════════════════
    # АЛГОРИТМ ОПРЕДЕЛЕНИЯ ФАЗЫ
    # ═══════════════════════════════════════

    # ACCUMULATION: цена в нижней зоне + высокий объём на лоях +
    #               CVD бычий (покупатели поглощают) + волатильность сжимается
    if (position_pct <= 25
            and vol_at_lows > vol_avg * 1.2
            and cvd_level in ('bull', 'bull_div')
            and vol_compression < 0.9):
        strength = 0
        if position_pct <= 15:       strength += 1
        if vol_at_lows > vol_avg * 2: strength += 1
        if cvd_level == 'bull_div':   strength += 1
        return 'accumulation', True, False, '🟢 Накопление (Wyckoff)', min(strength, 3)

    # SPRING: ложный пробой лоя с разворотом — сильнейший лонг-сигнал
    # Цена недавно была ниже swing_low, но вернулась выше
    recent_lows = [c[3] for c in ohlcv[-5:]]
    spring_detected = (min(recent_lows) < swing_low * 1.005
                       and current > swing_low * 1.01
                       and cvd_level in ('bull', 'bull_div'))
    if spring_detected:
        return 'accumulation', True, False, '🌱 Spring (Wyckoff)', 3

    # DISTRIBUTION: цена в верхней зоне + высокий объём на хаях +
    #               CVD медвежий (продавцы доминируют)
    if (position_pct >= 75
            and vol_at_highs > vol_avg * 1.2
            and cvd_level in ('bear', 'bear_div')
            and vol_compression < 0.9):
        strength = 0
        if position_pct >= 85:        strength += 1
        if vol_at_highs > vol_avg * 2: strength += 1
        if cvd_level == 'bear_div':    strength += 1
        return 'distribution', False, True, '🔴 Распределение (Wyckoff)', min(strength, 3)

    # UTAD: ложный пробой хая — сильнейший шорт-сигнал
    recent_highs = [c[2] for c in ohlcv[-5:]]
    utad_detected = (max(recent_highs) > swing_high * 0.995
                     and current < swing_high * 0.99
                     and cvd_level in ('bear', 'bear_div'))
    if utad_detected:
        return 'distribution', False, True, '🔝 UTAD (Wyckoff)', 3

    # MARKUP: устойчивый рост, цена выше середины диапазона
    if trend_20 > 3.0 and position_pct > 50 and cvd_level in ('bull', 'bull_div'):
        return 'markup', True, False, '📈 Разгон (Markup)', 1

    # MARKDOWN: устойчивое падение, цена ниже середины диапазона
    if trend_20 < -3.0 and position_pct < 50 and cvd_level in ('bear', 'bear_div'):
        return 'markdown', False, True, '📉 Снижение (Markdown)', 1

    # RANGING: диапазон без явной фазы — осторожно оба направления
    return 'ranging', True, True, '⚪️ Диапазон', 0


# ─────────────────────────────────────────────
# ФИЛЬТР НАПРАВЛЕНИЯ ПОСЛЕ БОЛЬШОГО ДВИЖЕНИЯ
#
# Ключевое исправление для SIREN:
# После дампа -30%+ бот НЕ ДОЛЖЕН давать шорт.
# После памп +30%+ бот НЕ ДОЛЖЕН давать лонг.
#
# Логика: смотрим на максимальное движение за последние N свечей
# ─────────────────────────────────────────────
def check_post_move_filter(ohlcv, lookback=10):
    """
    Анализирует последнее большое движение.
    Возвращает:
      after_dump:  True = был дамп, только лонг-зона
      after_pump:  True = был памп, только шорт-зона
      move_pct:    величина движения
      candles_since: сколько свечей прошло с движения
    """
    if len(ohlcv) < lookback + 1:
        return False, False, 0.0, 0

    # Ищем свечу с максимальным движением в обе стороны
    max_down  = 0.0
    max_up    = 0.0
    down_idx  = 0
    up_idx    = 0

    for i in range(len(ohlcv) - lookback, len(ohlcv)):
        candle = ohlcv[i]
        o, h, l, c = candle[1], candle[2], candle[3], candle[4]

        # Движение свечи относительно открытия
        move_pct_down = (o - l) / o * 100 if o > 0 else 0  # дамп внутри свечи
        move_pct_up   = (h - o) / o * 100 if o > 0 else 0  # памп внутри свечи

        # Движение от закрытия предыдущей свечи
        if i > 0:
            prev_close = ohlcv[i-1][4]
            close_move_down = (prev_close - c) / prev_close * 100 if prev_close > 0 else 0
            close_move_up   = (c - prev_close) / prev_close * 100 if prev_close > 0 else 0
            move_pct_down = max(move_pct_down, close_move_down)
            move_pct_up   = max(move_pct_up,   close_move_up)

        if move_pct_down > max_down:
            max_down = move_pct_down
            down_idx = i
        if move_pct_up > max_up:
            max_up = move_pct_up
            up_idx = i

    last_idx = len(ohlcv) - 1
    candles_since_dump = last_idx - down_idx
    candles_since_pump = last_idx - up_idx

    # Дамп: >15% за 1-3 свечи → теперь только лонг-зона
    # Ждём стабилизации (не сразу после дампа, а через 1-3 свечи)
    after_dump = (max_down >= 15.0 and 1 <= candles_since_dump <= 6)

    # Памп: >15% за 1-3 свечи → теперь только шорт-зона
    after_pump = (max_up >= 15.0 and 1 <= candles_since_pump <= 6)

    move_pct = max_down if after_dump else (max_up if after_pump else 0.0)
    candles_since = candles_since_dump if after_dump else (candles_since_pump if after_pump else 0)

    return after_dump, after_pump, move_pct, candles_since


# ─────────────────────────────────────────────
# Z-SCORE ОБЪЁМА
# ─────────────────────────────────────────────
def calc_volume_zscore(closed, period=20):
    vols = [x[5] for x in closed[-(period+1):-1]]
    if len(vols) < period:
        return 0.0
    cur_vol = closed[-1][5]
    mean    = np.mean(vols)
    std     = np.std(vols)
    if std == 0:
        return 0.0
    return (cur_vol - mean) / std


# ─────────────────────────────────────────────
# CVD УРОВЕНЬ
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

    if price_new_low  and not cvd_new_low:   return 'bull_div', total_cvd
    if price_new_high and not cvd_new_high:  return 'bear_div', total_cvd
    if total_cvd > 0:                        return 'bull', total_cvd
    return 'bear', total_cvd


# ─────────────────────────────────────────────
# RSI ДИВЕРГЕНЦИЯ
# ─────────────────────────────────────────────
def calc_rsi_divergence(closed, rsi_period=14, lookback=10):
    closes = [x[4] for x in closed]
    if len(closes) < rsi_period + lookback + 2:
        return False, False

    def rsi_at(closes_slice):
        if len(closes_slice) < rsi_period + 1:
            return 50.0
        deltas   = np.diff(closes_slice)
        gains    = np.where(deltas > 0, deltas, 0.0)
        losses   = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[:rsi_period])
        avg_loss = np.mean(losses[:rsi_period])
        for i in range(rsi_period, len(gains)):
            avg_gain = (avg_gain * (rsi_period - 1) + gains[i]) / rsi_period
            avg_loss = (avg_loss * (rsi_period - 1) + losses[i]) / rsi_period
        if avg_loss == 0:
            return 100.0
        return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

    rsi_values = []
    for i in range(lookback + 1):
        idx    = len(closes) - lookback - 1 + i
        window = closes[max(0, idx - rsi_period - 1): idx + 1]
        rsi_values.append(rsi_at(window))

    price_window = closes[-(lookback+1):]
    price_new_low  = price_window[-1] < min(price_window[:-1])
    price_new_high = price_window[-1] > max(price_window[:-1])
    rsi_new_low    = rsi_values[-1] < min(rsi_values[:-1])
    rsi_new_high   = rsi_values[-1] > max(rsi_values[:-1])

    bull_div = price_new_low  and not rsi_new_low
    bear_div = price_new_high and not rsi_new_high
    return bull_div, bear_div


# ─────────────────────────────────────────────
# CVD ДИВЕРГЕНЦИЯ
# ─────────────────────────────────────────────
def get_cvd_divergence(ohlcv, mode='long'):
    closes     = [x[4] for x in ohlcv]
    lookback   = 10
    cvd_proxy  = []
    cumulative = 0.0
    for candle in ohlcv:
        o, h, l, c, v = candle[1], candle[2], candle[3], candle[4], candle[5]
        ratio      = (c - l) / (h - l) if h != l else 0.5
        delta      = (ratio - 0.5) * 2 * v
        cumulative += delta
        cvd_proxy.append(cumulative)
    if len(closes) < lookback + 1:
        return False
    if mode == 'long':
        return closes[-1] < min(closes[-lookback:-1]) and cvd_proxy[-1] > min(cvd_proxy[-lookback:-1])
    else:
        return closes[-1] > max(closes[-lookback:-1]) and cvd_proxy[-1] < max(cvd_proxy[-lookback:-1])


# ─────────────────────────────────────────────
# OI
# ─────────────────────────────────────────────
def get_oi_data_from_ticker(symbol, tickers, price_change_pct):
    try:
        ticker = tickers.get(symbol)
        if not ticker:
            return 0.0, 'neutral', '⚪️ OI: нет данных'

        info     = ticker.get('info', {})
        hold_vol = float(info.get('holdVol', 0) or 0)

        if hold_vol == 0:
            return 0.0, 'neutral', '⚪️ OI: нет данных'

        prev_vol = oi_cache.get(symbol, 0)
        oi_cache[symbol] = hold_vol

        if prev_vol == 0:
            return 0.0, 'neutral', f"⚪️ OI: {hold_vol:.0f} (первое измерение)"

        oi_chg = (hold_vol - prev_vol) / prev_vol * 100

        if abs(oi_chg) < 2.0:
            return oi_chg, 'neutral', f"⚪️ OI: {oi_chg:+.1f}% (нейтраль)"

        if oi_chg > 0 and price_change_pct >= 0:
            signal = 'bull'
            label  = f"🟢 OI: +{oi_chg:.1f}% (деньги заходят в лонг)"
        elif oi_chg > 0 and price_change_pct < 0:
            signal = 'bear'
            label  = f"🔴 OI: +{oi_chg:.1f}% (деньги заходят в шорт)"
        elif oi_chg < 0 and price_change_pct >= 0:
            signal = 'squeeze_up'
            label  = f"⚠️ OI: {oi_chg:.1f}% (шорт-сквиз)"
        else:
            signal = 'squeeze_dn'
            label  = f"⚠️ OI: {oi_chg:.1f}% (лонг-ликвидации)"

        return oi_chg, signal, label

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
        if rate < -0.02:
            signal = 'bull'
            label  = f"🟢 Фандинг: {rate:.3f}%"
        elif rate > 0.05:
            signal = 'bear'
            label  = f"🔴 Фандинг: {rate:.3f}%"
        else:
            signal = 'neutral'
            label  = f"⚪️ Фандинг: {rate:.3f}%"
        return rate, signal, label
    except Exception as e:
        logging.debug(f"Funding error {symbol}: {e}")
        return 0.0, 'neutral', '⚪️ Фандинг: нет данных'


# ─────────────────────────────────────────────
# PIVOT УРОВНИ
# ─────────────────────────────────────────────
def get_pivot_levels(ohlcv, tolerance=0.005):
    highs  = [x[2] for x in ohlcv]
    lows   = [x[3] for x in ohlcv]
    pivots_low  = []
    pivots_high = []
    for i in range(2, len(ohlcv) - 2):
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            pivots_low.append(lows[i])
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            pivots_high.append(highs[i])

    def cluster(points, tol):
        if not points:
            return []
        points   = sorted(points)
        clusters = []
        group    = [points[0]]
        for p in points[1:]:
            if (p - group[0]) / group[0] <= tol:
                group.append(p)
            else:
                clusters.append(np.mean(group))
                group = [p]
        clusters.append(np.mean(group))
        return [c for c in clusters
                if sum(1 for p in points if abs(p - c) / c <= tol) >= 2]

    supports    = cluster(pivots_low, tolerance)
    resistances = cluster(pivots_high, tolerance)
    cp = ohlcv[-1][4]
    return (sorted([s for s in supports if s < cp], reverse=True),
            sorted([r for r in resistances if r > cp]))


# ─────────────────────────────────────────────
# ATR
# ─────────────────────────────────────────────
def calculate_atr(ohlcv, period=14):
    if len(ohlcv) < period + 1:
        return None
    trs = []
    for i in range(1, len(ohlcv)):
        h  = ohlcv[i][2]; l = ohlcv[i][3]; pc = ohlcv[i-1][4]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = np.mean(trs[:period])
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def dynamic_atr_multipliers(btc_vol: float):
    if btc_vol > 3.0:   stop_k, take_k = 2.0, 3.0
    elif btc_vol > 1.5: stop_k, take_k = 1.5, 2.5
    else:               stop_k, take_k = 1.2, 2.0
    return stop_k, take_k, f"1 : {take_k/stop_k:.1f}"


# ─────────────────────────────────────────────
# MFI
# ─────────────────────────────────────────────
def calculate_mfi(ohlcv, period=14):
    if len(ohlcv) < period + 1:
        return 50.0
    tp_prev = None
    pos_mf  = []; neg_mf = []
    for candle in ohlcv[-(period + 1):]:
        h, l, c, v = candle[2], candle[3], candle[4], candle[5]
        tp = (h + l + c) / 3
        mf = tp * v
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
    if len(closes) < period + 1:
        return 50.0
    deltas   = np.diff(closes)
    gains    = np.where(deltas > 0, deltas, 0.0)
    losses   = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0: return 100.0
    return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))


# ─────────────────────────────────────────────
# DEMARK TD
# ─────────────────────────────────────────────
def update_td_counters(ohlcv, mode='long'):
    closed = ohlcv[:-1]
    closes = [x[4] for x in closed]
    lows   = [x[3] for x in closed]
    highs  = [x[2] for x in closed]
    n      = len(closes)

    if n < 14:
        return False, False, False

    s_count    = 0
    in_c       = False
    c_count    = 0
    setup_high = None
    setup_low  = None
    setup_bars = []
    m9_signal  = False
    m9_perfect = False
    m13_signal = False
    last_idx   = n - 1

    for i in range(4, n):
        c  = closes[i]
        c4 = closes[i - 4]
        setup_cond = (c < c4) if mode == 'long' else (c > c4)

        if not in_c:
            if setup_cond:
                s_count += 1
                setup_bars.append(i)
                if s_count == 9:
                    if len(setup_bars) >= 9:
                        idx6, idx7 = setup_bars[-4], setup_bars[-3]
                        idx8, idx9 = setup_bars[-2], setup_bars[-1]
                        if mode == 'long':
                            perfect = (lows[idx8] < lows[idx6] and lows[idx8] < lows[idx7]) or \
                                      (lows[idx9] < lows[idx6] and lows[idx9] < lows[idx7])
                        else:
                            perfect = (highs[idx8] > highs[idx6] and highs[idx8] > highs[idx7]) or \
                                      (highs[idx9] > highs[idx6] and highs[idx9] > highs[idx7])
                    else:
                        perfect = False
                    if i == last_idx:
                        m9_signal = True; m9_perfect = perfect
                    setup_high  = max(highs[b] for b in setup_bars)
                    setup_low   = min(lows[b]  for b in setup_bars)
                    in_c = True; c_count = 0; s_count = 0; setup_bars = []
            else:
                s_count = 0; setup_bars = []

        elif in_c:
            if mode == 'long' and setup_high is not None and c > setup_high:
                in_c = False; c_count = 0; setup_high = None; setup_low = None
                s_count = 1 if setup_cond else 0
                setup_bars = [i] if setup_cond else []
                continue
            elif mode == 'short' and setup_low is not None and c < setup_low:
                in_c = False; c_count = 0; setup_high = None; setup_low = None
                s_count = 1 if setup_cond else 0
                setup_bars = [i] if setup_cond else []
                continue

            if i >= 2:
                cd_cond = (c <= lows[i - 2]) if mode == 'long' else (c >= highs[i - 2])
                if cd_cond:
                    c_count += 1
                    if c_count == 13:
                        if i == last_idx:
                            m13_signal = True
                        in_c = False; c_count = 0; setup_high = None; setup_low = None

            if in_c and c_count > 30:
                in_c = False; c_count = 0; setup_high = None; setup_low = None

    return m9_signal, m9_perfect, m13_signal


# ─────────────────────────────────────────────
# МОЛОТ
# ─────────────────────────────────────────────
def check_hammer(ohlcv, mode='long'):
    if len(ohlcv) < 2:
        return False
    candle     = ohlcv[-2]
    o, h, l, c = candle[1], candle[2], candle[3], candle[4]
    body       = abs(c - o)
    full_range = h - l if h != l else 1e-9
    if mode == 'long':
        lower_wick = min(o, c) - l
        return (lower_wick / full_range) > 0.6 and (body / full_range) < 0.3
    else:
        upper_wick = h - max(o, c)
        return (upper_wick / full_range) > 0.6 and (body / full_range) < 0.3


def build_tv_link(symbol: str) -> str:
    tv_sym = symbol.replace('/', '').replace(':USDT', '.P')
    return f"🔗 <a href='https://www.tradingview.com/chart/?symbol=MEXC:{tv_sym}'>Открыть в TradingView</a>"


def send_msg(text):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    try:
        url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(
            url,
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10
        )
        if not resp.ok:
            logging.error(f"TG error: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        logging.error(f"Ошибка отправки TG: {e}")


# ─────────────────────────────────────────────
# РЫНОЧНЫЙ КОНТЕКСТ
# ─────────────────────────────────────────────
def get_market_context():
    for attempt in range(3):
        try:
            btc       = exchange.fetch_ohlcv('BTC/USDT:USDT', '4h', limit=5)
            btc_ch    = ((btc[-1][4] - btc[-2][4]) / btc[-2][4]) * 100
            btc_trend = "🟢" if btc_ch > -0.3 else "🔴"
            eth       = exchange.fetch_ohlcv('ETH/USDT:USDT', '4h', limit=5)
            eth_ch    = ((eth[-1][4] - eth[-2][4]) / eth[-2][4]) * 100
            alt_diff  = eth_ch - btc_ch
            alt_power = "🚀" if alt_diff > 0.5 else "⚓️"
            btc_moves = [abs((btc[i][4] - btc[i-1][4]) / btc[i-1][4]) * 100 for i in range(1, 5)]
            return {"btc_trend": btc_trend, "btc_ch": btc_ch, "btc_p": btc[-1][4],
                    "alt_power": alt_power, "alt_ch": alt_diff, "btc_vol": np.mean(btc_moves)}
        except Exception as e:
            logging.warning(f"get_market_context попытка {attempt+1}/3: {e}")
            time.sleep(5)
    return {"btc_trend": "⚪️", "btc_ch": 0, "btc_p": 0,
            "alt_power": "⚪️", "alt_ch": 0, "btc_vol": 1.0}


def adaptive_threshold(base: int, btc_vol: float, is_priority: bool) -> int:
    t = base
    if btc_vol > 3.0:   t += 2
    elif btc_vol < 0.5: t -= 1
    if is_priority:     t -= 1
    return max(t, 3)


# ─────────────────────────────────────────────
# ОСНОВНОЙ ЦИКЛ
# ─────────────────────────────────────────────
def analyst_loop():
    sent_signals = {}
    logging.info("Аналитик v5.0 запущен.")

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
                    exchange.load_markets()
                    markets_reload_ts = time.time()
                    logging.info("Рынки перезагружены.")
                except Exception as e:
                    logging.error(f"Ошибка перезагрузки: {e}")

            ctx = get_market_context()

            try:
                tickers = exchange.fetch_tickers()
            except Exception as e:
                logging.error(f"fetch_tickers error: {e}")
                time.sleep(60)
                continue

            active_swaps = [s for s, m in exchange.markets.items()
                            if m.get('active') and m.get('type') == 'swap']

            vol_data = sorted(
                [{'s': s, 'v': tickers[s].get('quoteVolume', 0)}
                 for s in active_swaps if s in tickers],
                key=lambda x: x['v'], reverse=True
            )
            top250  = [x['s'] for x in vol_data[:250]]
            symbols = list(dict.fromkeys(
                [w for w in WATCHLIST if w in tickers] + top250
            ))

            for symbol in symbols:
                try:
                    ohlcv = None
                    for attempt in range(2):
                        try:
                            ohlcv = exchange.fetch_ohlcv(symbol, '4h', limit=120)
                            break
                        except ccxt.NetworkError as e:
                            if attempt == 0:
                                logging.warning(f"fetch_ohlcv retry {symbol}: {e}")
                                time.sleep(3)
                            else:
                                raise
                    if ohlcv is None or len(ohlcv) < 80:
                        continue

                    is_wl         = symbol in WATCHLIST
                    threshold     = adaptive_threshold(5, ctx['btc_vol'], is_wl)
                    current_price = ohlcv[-1][4]
                    closed        = ohlcv[:-1]
                    closes_closed = [x[4] for x in closed]

                    last_closed_ts = closed[-1][0]
                    candle_id      = last_closed_ts // 14_400_000
                    l_key = f"{symbol}_{candle_id}_l"
                    s_key = f"{symbol}_{candle_id}_s"

                    if l_key in sent_signals and s_key in sent_signals:
                        continue

                    # ══════════════════════════════════════════
                    # 1. БАЗОВЫЕ ИНДИКАТОРЫ
                    # ══════════════════════════════════════════
                    v_history = [x[5] for x in closed[-21:-1]]
                    v_avg     = np.mean(v_history) if v_history else 1.0
                    v_rel     = closed[-1][5] / v_avg if v_avg > 0 else 1.0
                    v_zscore  = calc_volume_zscore(closed)

                    if v_rel >= 10 or v_zscore >= 3.0:
                        vol_score = 3; vol_label = f"🚀 Vol x{v_rel:.0f} Z:{v_zscore:.1f}σ ВЗРЫВ"
                    elif v_rel >= 5 or v_zscore >= 2.0:
                        vol_score = 2; vol_label = f"📊 Vol x{v_rel:.1f} Z:{v_zscore:.1f}σ"
                    elif v_rel >= 1.8:
                        vol_score = 1; vol_label = f"📊 Vol x{v_rel:.1f}"
                    else:
                        vol_score = 0; vol_label = ""

                    rsi = calculate_rsi_wilder(closes_closed)
                    mfi = calculate_mfi(closed)
                    atr = calculate_atr(closed)

                    c_high  = closed[-1][2]; c_low = closed[-1][3]; c_close = closed[-1][4]
                    buy_pressure = (c_close - c_low) / (c_high - c_low) if c_high != c_low else 0.5
                    imb = (buy_pressure - 0.5) * 200

                    price_ch = ((closed[-1][4] - closed[-2][4]) / closed[-2][4] * 100) \
                               if len(closed) >= 2 else 0.0

                    # ══════════════════════════════════════════
                    # 2. НОВЫЕ ИНДИКАТОРЫ (PpSignal-логика)
                    # ══════════════════════════════════════════

                    # SSMA тренд
                    ssma_val, ssma_trend, ssma_slope = calculate_ssma(closed, period=24)

                    # BB Outside ATR — перепроданность/перекупленность
                    bb_signal, bb_dist, bb_upper, bb_lower, bb_basis = calculate_bb_outside_atr(closed)

                    # Swing HILO — структурные уровни
                    sw_low_pct, sw_high_pct, near_sw_low, near_sw_high, sw_low, sw_high = \
                        calculate_swing_hilo(closed, swing_bars=34)

                    # CVD
                    cvd_level, cvd_total = calc_cvd_level(closed)
                    cvd_div_l = get_cvd_divergence(closed, 'long')
                    cvd_div_s = get_cvd_divergence(closed, 'short')

                    # RSI дивергенция
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

                    # ATR мультипликаторы
                    stop_k, take_k, rr_str = dynamic_atr_multipliers(ctx['btc_vol'])
                    if atr:
                        stop_l = current_price - stop_k * atr; target_l = current_price + take_k * atr
                        stop_s = current_price + stop_k * atr; target_s = current_price - take_k * atr
                    else:
                        stop_l = stop_s = target_l = target_s = None

                    # ══════════════════════════════════════════
                    # 3. WYCKOFF ФАЗА — главный фильтр направления
                    # ══════════════════════════════════════════
                    wyckoff_phase, wyckoff_long_ok, wyckoff_short_ok, wyckoff_label, wyckoff_strength = \
                        detect_wyckoff_phase(closed, sw_low or sup, sw_high or res, cvd_level)

                    # ══════════════════════════════════════════
                    # 4. ФИЛЬТР ПОСЛЕ БОЛЬШОГО ДВИЖЕНИЯ
                    # Исправляет проблему SIREN:
                    # после дампа -64% нельзя давать шорт
                    # ══════════════════════════════════════════
                    after_dump, after_pump, big_move_pct, candles_since_move = \
                        check_post_move_filter(ohlcv, lookback=12)

                    # Уточняем Wyckoff после большого движения
                    if after_dump and not wyckoff_long_ok:
                        # Дамп произошёл — переопределяем в накопление
                        wyckoff_long_ok  = True
                        wyckoff_short_ok = False
                        wyckoff_label    = f"🌱 После дампа -{big_move_pct:.0f}% ({candles_since_move} св.)"

                    if after_pump and not wyckoff_short_ok:
                        # Памп произошёл — переопределяем в распределение
                        wyckoff_short_ok = True
                        wyckoff_long_ok  = False
                        wyckoff_label    = f"🔝 После памп +{big_move_pct:.0f}% ({candles_since_move} св.)"

                    # ══════════════════════════════════════════
                    # 5. OI И ФАНДИНГ
                    # ══════════════════════════════════════════
                    has_any_signal = (cvd_div_l or cvd_div_s or hammer_l or hammer_s or
                                      m9_l or m9_s or m13_l or m13_s or
                                      rsi_div_bull or rsi_div_bear or vol_score >= 2 or
                                      near_sw_low or near_sw_high or
                                      bb_signal != 'neutral')

                    oi_chg    = 0.0; oi_signal = 'neutral'; oi_label  = '⚪️ OI: нет данных'
                    fr_val    = 0.0; fr_signal = 'neutral'; fr_label  = '⚪️ Фандинг: нет данных'

                    if has_any_signal:
                        oi_chg, oi_signal, oi_label = get_oi_data_from_ticker(symbol, tickers, price_ch)
                        fr_val, fr_signal, fr_label  = get_funding_signal(symbol)

                    wl_badge = " | ⭐️ WL" if is_wl else ""

                    # ══════════════════════════════════════════
                    # 6. СКОРИНГ ЛОНГ
                    # ══════════════════════════════════════════
                    if l_key not in sent_signals and wyckoff_long_ok:

                        l_score = 0; l_details = []

                        # Wyckoff бонус
                        if wyckoff_phase in ('accumulation',) and wyckoff_strength >= 2:
                            l_score += 2; l_details.append(f"🟢 Накопление")
                        elif wyckoff_phase == 'accumulation' and wyckoff_strength == 1:
                            l_score += 1; l_details.append(f"🟢 Накопление")

                        # BB Outside ATR — перепроданность
                        if bb_signal == 'oversold':
                            l_score += 2; l_details.append(f"📉 BB ATR Перепродан ({bb_dist:.1f}%)")
                        
                        # Swing HILO — цена у основания
                        if near_sw_low:
                            l_score += 2; l_details.append(f"🏔️ Swing Low (-{sw_low_pct:.1f}%)")

                        # SSMA тренд
                        if ssma_trend in ('bull_strong', 'bull_weak') and ssma_slope > 0:
                            l_score += 1; l_details.append(f"📈 SSMA Бык ({ssma_slope:+.2f}%)")

                        # OI движение денег
                        if oi_signal == 'bull':
                            l_score += 2; l_details.append(f"💹 OI +{oi_chg:.1f}%")
                        elif oi_signal == 'bear':
                            l_score -= 1

                        # CVD
                        if cvd_div_l:
                            l_score += 2; l_details.append("🔥 CVD Дивер")
                        if rsi_div_bull:
                            l_score += 2; l_details.append("📉 RSI Дивер")

                        # DeMark
                        if m9_l:
                            pts = 3 if m9_perfect_l else 2
                            l_score += pts
                            l_details.append("⏱ M9 Setup ✨" if m9_perfect_l else "⏱ M9 Setup")
                        if m13_l:
                            l_score += 3; l_details.append("⏱ M13 Reversal")

                        # CVD уровень
                        if cvd_level in ('bull', 'bull_div'):
                            l_score += 1; l_details.append("📍 CVD Уровень")

                        # Pivot
                        if current_price <= sup * 1.015:
                            l_score += 2; l_details.append("🧱 Pivot Support")

                        # Объём
                        if vol_score > 0:
                            l_score += vol_score; l_details.append(vol_label)

                        # Фандинг
                        if fr_signal == 'bull':
                            l_score += 1; l_details.append(f"💸 Фандинг {fr_val:.3f}%")

                        # Молот
                        if hammer_l:
                            l_score += 1; l_details.append("⚓️ Фитиль")

                        # RSI/MFI
                        if rsi < 30:
                            l_score += 1; l_details.append(f"📉 RSI {rsi:.0f}")
                        elif mfi < 20:
                            l_score += 1; l_details.append(f"💰 MFI {mfi:.0f}")

                        l_score = max(l_score, 0)

                        # ВЕТО
                        btc_is_red   = ctx['btc_trend'] == "🔴"
                        is_strong_l  = oi_signal == 'bull' or m13_l or (v_rel > 5 and imb > 60)
                        macro_veto_l = btc_is_red and not is_strong_l and not after_dump

                        # После дампа объём-взрыв идёт В ПОЛЬЗУ лонга, не против
                        dump_veto_l  = v_rel > 10 and imb < -60 and price_ch < -5 and not after_dump

                        overbought_veto_l = (rsi > 75 and mfi > 80) and not m13_l

                        # OI-ворота (смягчаем если Wyckoff накопление)
                        if oi_signal not in ('bull', 'squeeze_dn') and not (m13_l or cvd_div_l):
                            if wyckoff_phase != 'accumulation':
                                l_score = min(l_score, 4)

                        if (l_score >= threshold
                                and not macro_veto_l
                                and not dump_veto_l
                                and not overbought_veto_l):

                            if is_strong_l and btc_is_red:
                                status = "⚡️ СИЛЬНЕЕ РЫНКА"
                            else:
                                status = "✅ ТРЕНД"

                            tv_link   = build_tv_link(symbol)
                            atr_block = ""
                            if atr and stop_l and target_l:
                                atr_block = (
                                    f"───────────────────\n"
                                    f"🎯 Цель: <code>{target_l:.6g}</code> (+{((target_l/current_price-1)*100):.1f}%)\n"
                                    f"🛑 Стоп: <code>{stop_l:.6g}</code> (-{((1-stop_l/current_price)*100):.1f}%)\n"
                                    f"⚡️ ATR: <code>{atr:.6g}</code> | R/R ≈ {rr_str}\n"
                                )

                            cvd_emoji = {"bull": "🟢", "bull_div": "🟢✨",
                                         "bear": "🔴", "bear_div": "🔴✨"}.get(cvd_level, "⚪️")

                            ssma_label = f"{'📈' if 'bull' in ssma_trend else '📉'} SSMA: {ssma_val:.4g} ({ssma_slope:+.2f}%/св)" if ssma_val else ""

                            msg = (
                                f"🚨 <b>ЛОНГ 4H ({l_score}/10){wl_badge}</b>\n"
                                f"Монета: <b>{symbol}</b> | {status}\n"
                                f"Цена: <code>{current_price:.6g}</code>\n"
                                f"Сигналы: {', '.join(l_details)}\n"
                                f"───────────────────\n"
                                f"🌊 Wyckoff: <b>{wyckoff_label}</b>\n"
                                f"{'📉 BB ATR: Перепродан' if bb_signal == 'oversold' else ''}\n"
                                f"{'🏔️ Swing Low: ' + f'{sw_low_pct:.1f}% от лоя' if near_sw_low else ''}\n"
                                f"{ssma_label}\n"
                                f"───────────────────\n"
                                f"📊 RSI: <b>{rsi:.1f}</b>  |  MFI: <b>{mfi:.1f}</b>\n"
                                f"⚖️ Дисбаланс: {'🟢' if imb > 0 else '🔴'} <b>{abs(imb):.0f}%</b>\n"
                                f"📦 Объём: <b>x{v_rel:.1f}</b> | Z-Score: <b>{v_zscore:.1f}σ</b>\n"
                                f"{cvd_emoji} CVD уровень: <b>{cvd_level}</b>\n"
                                f"{oi_label}\n"
                                f"{fr_label}\n"
                                f"{atr_block}"
                                f"───────────────────\n"
                                f"🌍 <b>МАРКЕТ:</b>\n"
                                f"👑 BTC: {ctx['btc_trend']} {ctx['btc_ch']:.1f}% ({ctx['btc_p']:.0f})\n"
                                f"🚀 Alt-Strength: {ctx['alt_power']} {ctx['alt_ch']:.2f}% (ETH vs BTC)\n"
                                f"───────────────────\n"
                                f"{tv_link}"
                            )
                            send_msg(msg)
                            sent_signals[l_key] = time.time()
                            bot_status["signals_sent"] += 1
                            logging.info(f"ЛОНГ: {symbol} score={l_score} thr={threshold} "
                                         f"wyckoff={wyckoff_phase} bb={bb_signal} "
                                         f"OI={oi_signal}({oi_chg:.1f}%) | {l_details}")

                    # ══════════════════════════════════════════
                    # 7. СКОРИНГ ШОРТ
                    # ══════════════════════════════════════════
                    if s_key not in sent_signals and wyckoff_short_ok:

                        s_score = 0; s_details = []

                        # Wyckoff бонус
                        if wyckoff_phase in ('distribution',) and wyckoff_strength >= 2:
                            s_score += 2; s_details.append(f"🔴 Распределение")
                        elif wyckoff_phase == 'distribution' and wyckoff_strength == 1:
                            s_score += 1; s_details.append(f"🔴 Распределение")

                        # BB Outside ATR — перекупленность
                        if bb_signal == 'overbought':
                            s_score += 2; s_details.append(f"📈 BB ATR Перекуплен ({bb_dist:.1f}%)")

                        # Swing HILO — цена у вершины
                        if near_sw_high:
                            s_score += 2; s_details.append(f"🏔️ Swing High (-{sw_high_pct:.1f}%)")

                        # SSMA тренд
                        if ssma_trend in ('bear_strong', 'bear_weak') and ssma_slope < 0:
                            s_score += 1; s_details.append(f"📉 SSMA Медведь ({ssma_slope:+.2f}%)")

                        # OI
                        if oi_signal == 'bear':
                            s_score += 2; s_details.append(f"💹 OI +{oi_chg:.1f}%")
                        elif oi_signal == 'bull':
                            s_score -= 1

                        # CVD
                        if cvd_div_s:
                            s_score += 2; s_details.append("🔥 CVD Дивер")
                        if rsi_div_bear:
                            s_score += 2; s_details.append("📈 RSI Дивер")

                        # DeMark
                        if m9_s:
                            pts = 3 if m9_perfect_s else 2
                            s_score += pts
                            s_details.append("⏱ M9 Setup ✨" if m9_perfect_s else "⏱ M9 Setup")
                        if m13_s:
                            s_score += 3; s_details.append("⏱ M13 Reversal")

                        # CVD уровень
                        if cvd_level in ('bear', 'bear_div'):
                            s_score += 1; s_details.append("📍 CVD Уровень")

                        # Pivot
                        if current_price >= res * 0.985:
                            s_score += 2; s_details.append("🧱 Pivot Resistance")

                        # Объём
                        if vol_score > 0:
                            s_score += vol_score; s_details.append(vol_label)

                        # Фандинг
                        if fr_signal == 'bear':
                            s_score += 1; s_details.append(f"💸 Фандинг {fr_val:.3f}%")

                        # Молот
                        if hammer_s:
                            s_score += 1; s_details.append("🏹 Фитиль вверх")

                        # RSI/MFI
                        if rsi > 70:
                            s_score += 1; s_details.append(f"📈 RSI {rsi:.0f}")
                        elif mfi > 80:
                            s_score += 1; s_details.append(f"💰 MFI {mfi:.0f}")

                        s_score = max(s_score, 0)

                        # ВЕТО
                        btc_is_green = ctx['btc_trend'] == "🟢"
                        is_strong_s  = oi_signal == 'bear' or m13_s or (v_rel > 5 and imb < -60)
                        macro_veto_s = btc_is_green and not is_strong_s and not after_pump

                        pump_veto_s = v_rel > 8 and imb > 60 and price_ch > 3 and not after_pump

                        oversold_veto_s = (rsi < 25 and mfi < 20) and not m13_s

                        # OI-ворота (смягчаем если Wyckoff распределение)
                        if oi_signal not in ('bear', 'squeeze_up') and not (m13_s or cvd_div_s):
                            if wyckoff_phase != 'distribution':
                                s_score = min(s_score, 4)

                        if (s_score >= threshold
                                and not macro_veto_s
                                and not pump_veto_s
                                and not oversold_veto_s):

                            if is_strong_s and btc_is_green:
                                status = "⚡️ ПРОТИВ РЫНКА"
                            else:
                                status = "🔻 ШОРТ"

                            tv_link   = build_tv_link(symbol)
                            atr_block = ""
                            if atr and stop_s and target_s:
                                atr_block = (
                                    f"───────────────────\n"
                                    f"🎯 Цель: <code>{target_s:.6g}</code> (-{((1-target_s/current_price)*100):.1f}%)\n"
                                    f"🛑 Стоп: <code>{stop_s:.6g}</code> (+{((stop_s/current_price-1)*100):.1f}%)\n"
                                    f"⚡️ ATR: <code>{atr:.6g}</code> | R/R ≈ {rr_str}\n"
                                )

                            cvd_emoji = {"bull": "🟢", "bull_div": "🟢✨",
                                         "bear": "🔴", "bear_div": "🔴✨"}.get(cvd_level, "⚪️")

                            ssma_label = f"{'📈' if 'bull' in ssma_trend else '📉'} SSMA: {ssma_val:.4g} ({ssma_slope:+.2f}%/св)" if ssma_val else ""

                            msg = (
                                f"🚨 <b>ШОРТ 4H ({s_score}/10){wl_badge}</b>\n"
                                f"Монета: <b>{symbol}</b> | {status}\n"
                                f"Цена: <code>{current_price:.6g}</code>\n"
                                f"Сигналы: {', '.join(s_details)}\n"
                                f"───────────────────\n"
                                f"🌊 Wyckoff: <b>{wyckoff_label}</b>\n"
                                f"{'📈 BB ATR: Перекуплен' if bb_signal == 'overbought' else ''}\n"
                                f"{'🏔️ Swing High: ' + f'{sw_high_pct:.1f}% до хая' if near_sw_high else ''}\n"
                                f"{ssma_label}\n"
                                f"───────────────────\n"
                                f"📊 RSI: <b>{rsi:.1f}</b>  |  MFI: <b>{mfi:.1f}</b>\n"
                                f"⚖️ Дисбаланс: {'🔴' if imb > 0 else '🟢'} <b>{abs(imb):.0f}%</b>\n"
                                f"📦 Объём: <b>x{v_rel:.1f}</b> | Z-Score: <b>{v_zscore:.1f}σ</b>\n"
                                f"{cvd_emoji} CVD уровень: <b>{cvd_level}</b>\n"
                                f"{oi_label}\n"
                                f"{fr_label}\n"
                                f"{atr_block}"
                                f"───────────────────\n"
                                f"🌍 <b>МАРКЕТ:</b>\n"
                                f"👑 BTC: {ctx['btc_trend']} {ctx['btc_ch']:.1f}% ({ctx['btc_p']:.0f})\n"
                                f"🚀 Alt-Strength: {ctx['alt_power']} {ctx['alt_ch']:.2f}% (ETH vs BTC)\n"
                                f"───────────────────\n"
                                f"{tv_link}"
                            )
                            send_msg(msg)
                            sent_signals[s_key] = time.time()
                            bot_status["signals_sent"] += 1
                            logging.info(f"ШОРТ: {symbol} score={s_score} thr={threshold} "
                                         f"wyckoff={wyckoff_phase} bb={bb_signal} "
                                         f"OI={oi_signal}({oi_chg:.1f}%) | {s_details}")

                    time.sleep(0.15)

                except ccxt.RateLimitExceeded:
                    logging.warning(f"Rate limit: {symbol}, пауза 30с")
                    time.sleep(30)
                except ccxt.NetworkError as e:
                    logging.error(f"Network error {symbol}: {e}")
                except Exception as e:
                    logging.error(f"Ошибка {symbol}: {e}")

            now          = time.time()
            sent_signals = {k: v for k, v in sent_signals.items() if now - v < 86400}
            bot_status["iterations"]    += 1
            bot_status["last_iteration"] = datetime.now().strftime('%H:%M:%S')
            logging.info(f"Итерация. Символов: {len(symbols)}. Кэш: {len(sent_signals)}. BTC vol: {ctx['btc_vol']:.2f}%")
            time.sleep(300)

        except ccxt.NetworkError as e:
            logging.error(f"Глобальная сетевая ошибка: {e}")
            bot_status["errors"] += 1; time.sleep(60)
        except Exception as e:
            logging.error(f"Критическая ошибка: {e}")
            bot_status["errors"] += 1; time.sleep(60)


@app.route('/health')
def health():
    uptime = str(datetime.now() - datetime.fromisoformat(bot_status["started_at"])).split('.')[0]
    return (f"✅ OK\nUptime: {uptime}\nИтераций: {bot_status['iterations']}\n"
            f"Ошибок: {bot_status['errors']}\nСигналов: {bot_status['signals_sent']}\n"
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
                logging.info(f"Keepalive OK [{url}]: {r.status_code}"); break
            except Exception as e:
                logging.warning(f"Keepalive ошибка [{url}]: {e}")
        time.sleep(240)


def watchdog():
    time.sleep(60)
    while True:
        global analyst_thread
        if not analyst_thread.is_alive():
            logging.error("⚠️ analyst_loop упал! Перезапускаем...")
            bot_status["errors"] += 1
            analyst_thread = threading.Thread(target=analyst_loop, daemon=True, name="analyst")
            analyst_thread.start()
            logging.info("analyst_loop перезапущен.")
        time.sleep(60)


analyst_thread = threading.Thread(target=analyst_loop, daemon=True, name="analyst")
analyst_thread.start()
threading.Thread(target=keepalive_loop, daemon=True, name="keepalive").start()
threading.Thread(target=watchdog,       daemon=True, name="watchdog").start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
