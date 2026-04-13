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
    return f"✅ АНАЛИТИК v4.0 АКТИВЕН. Время: {datetime.now().strftime('%H:%M:%S')}"

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
# Z-SCORE ОБЪЁМА
# Показывает насколько текущий объём аномален
# в единицах стандартного отклонения.
# Z > 2 = значимая аномалия, Z > 3 = экстремум.
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
# Считаем накопленную дельту покупок/продаж
# на каждом ценовом уровне (Volume Profile by CVD).
# Возвращает: 'bull' если на текущей цене исторически
# больше покупок, 'bear' если продаж, 'neutral' иначе.
# ─────────────────────────────────────────────
def calc_cvd_level(closed):
    if len(closed) < 10:
        return 'neutral', 0.0

    # Считаем CVD дельту каждой свечи
    deltas = []
    for c in closed:
        h, l, cl, v = c[2], c[3], c[4], c[5]
        ratio = (cl - l) / (h - l) if h != l else 0.5
        deltas.append((ratio - 0.5) * 2 * v)

    # Накопленный CVD
    cumulative = np.cumsum(deltas)
    total_cvd  = cumulative[-1]

    # CVD дивергенция: цена обновила экстремум — CVD нет
    lookback   = 10
    closes     = [x[4] for x in closed]

    if len(closes) < lookback + 1:
        return 'neutral', total_cvd

    price_made_new_low  = closes[-1] < min(closes[-lookback:-1])
    price_made_new_high = closes[-1] > max(closes[-lookback:-1])
    cvd_made_new_low    = cumulative[-1] < min(cumulative[-lookback:-1])
    cvd_made_new_high   = cumulative[-1] > max(cumulative[-lookback:-1])

    bull_div = price_made_new_low  and not cvd_made_new_low   # цена ↓ CVD ↗ → покупатели держат
    bear_div = price_made_new_high and not cvd_made_new_high  # цена ↑ CVD ↘ → продавцы давят

    if bull_div:
        return 'bull_div', total_cvd
    if bear_div:
        return 'bear_div', total_cvd
    if total_cvd > 0:
        return 'bull', total_cvd
    return 'bear', total_cvd


# ─────────────────────────────────────────────
# RSI ДИВЕРГЕНЦИЯ
# Бычья: цена обновила минимум — RSI нет → разворот вверх.
# Медвежья: цена обновила максимум — RSI нет → разворот вниз.
# ─────────────────────────────────────────────
def calc_rsi_divergence(closed, rsi_period=14, lookback=10):
    closes = [x[4] for x in closed]
    if len(closes) < rsi_period + lookback + 2:
        return False, False

    # Считаем RSI для последних lookback+1 свечей
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
    rsi_window   = rsi_values

    price_new_low = price_window[-1] < min(price_window[:-1])
    price_new_high = price_window[-1] > max(price_window[:-1])
    rsi_new_low   = rsi_window[-1] < min(rsi_window[:-1])
    rsi_new_high  = rsi_window[-1] > max(rsi_window[:-1])

    bull_div = price_new_low  and not rsi_new_low
    bear_div = price_new_high and not rsi_new_high

    return bull_div, bear_div


# ─────────────────────────────────────────────
# ОТКРЫТЫЙ ИНТЕРЕС (OI)
# Сравниваем текущий OI с предыдущим.
# OI растёт + цена растёт = новые лонги (бычье)
# OI растёт + цена падает = новые шорты (медвежье)
# OI падает + цена растёт = шорт-сквиз (слабое движение)
# OI падает + цена падает = лонг-ликвидации (слабое)
# Возвращает (oi_change_pct, oi_signal)
# ─────────────────────────────────────────────
def get_oi_signal(symbol, price_change_pct):
    try:
        oi_data = exchange.fetch_open_interest_history(
            symbol, '4h', limit=3
        )
        if not oi_data or len(oi_data) < 2:
            return 0.0, 'neutral'

        oi_prev = float(oi_data[-2].get('openInterestAmount', 0) or 0)
        oi_curr = float(oi_data[-1].get('openInterestAmount', 0) or 0)

        if oi_prev == 0:
            return 0.0, 'neutral'

        oi_chg = (oi_curr - oi_prev) / oi_prev * 100

        if oi_chg > 2 and price_change_pct > 0:
            return oi_chg, 'bull'   # новые лонги — движение настоящее
        if oi_chg > 2 and price_change_pct < 0:
            return oi_chg, 'bear'   # новые шорты — движение настоящее
        if oi_chg < -2 and price_change_pct > 0:
            return oi_chg, 'squeeze_up'   # шорт-сквиз — осторожно с лонгом
        if oi_chg < -2 and price_change_pct < 0:
            return oi_chg, 'squeeze_dn'   # лонг-ликвидации
        return oi_chg, 'neutral'

    except Exception as e:
        logging.debug(f"OI error {symbol}: {e}")
        return 0.0, 'neutral'


# ─────────────────────────────────────────────
# ФАНДИНГ
# Отрицательный фандинг = шортисты переплачивают лонгистам
# → давление на шорт-сквиз → вероятен рост
# Положительный = лонги переплачивают → давление на снижение
# ─────────────────────────────────────────────
def get_funding_signal(symbol):
    try:
        fr = exchange.fetch_funding_rate(symbol)
        rate = float(fr.get('fundingRate', 0) or 0) * 100  # в %
        if rate < -0.05:
            return rate, 'bull'   # шортисты перегружены → лонг давление
        if rate > 0.05:
            return rate, 'bear'   # лонги перегружены → шорт давление
        return rate, 'neutral'
    except Exception as e:
        logging.debug(f"Funding error {symbol}: {e}")
        return 0.0, 'neutral'


# ─────────────────────────────────────────────
# CVD ДИВЕРГЕНЦИЯ (для скоринга, без уровней)
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
# PIVOT УРОВНИ
# ─────────────────────────────────────────────
def get_pivot_levels(ohlcv, tolerance=0.005):
    highs  = [x[2] for x in ohlcv]
    lows   = [x[3] for x in ohlcv]
    closes = [x[4] for x in ohlcv]
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
        result = []
        for c in clusters:
            touches = sum(1 for p in points if abs(p - c) / c <= tol)
            if touches >= 2:
                result.append(c)
        return result

    supports    = cluster(pivots_low, tolerance)
    resistances = cluster(pivots_high, tolerance)
    cp = closes[-1]
    supports    = sorted([s for s in supports if s < cp], reverse=True)
    resistances = sorted([r for r in resistances if r > cp])
    return supports, resistances


# ─────────────────────────────────────────────
# ATR (Wilder)
# ─────────────────────────────────────────────
def calculate_atr(ohlcv, period=14):
    if len(ohlcv) < period + 1:
        return None
    trs = []
    for i in range(1, len(ohlcv)):
        h  = ohlcv[i][2]
        l  = ohlcv[i][3]
        pc = ohlcv[i-1][4]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = np.mean(trs[:period])
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def dynamic_atr_multipliers(btc_vol: float):
    if btc_vol > 3.0:
        stop_k, take_k = 2.0, 3.0
    elif btc_vol > 1.5:
        stop_k, take_k = 1.5, 2.5
    else:
        stop_k, take_k = 1.2, 2.0
    return stop_k, take_k, f"1 : {take_k/stop_k:.1f}"


# ─────────────────────────────────────────────
# MFI
# ─────────────────────────────────────────────
def calculate_mfi(ohlcv, period=14):
    if len(ohlcv) < period + 1:
        return 50.0
    tp_prev = None
    pos_mf  = []
    neg_mf  = []
    for candle in ohlcv[-(period + 1):]:
        h, l, c, v = candle[2], candle[3], candle[4], candle[5]
        tp = (h + l + c) / 3
        mf = tp * v
        if tp_prev is not None:
            if tp > tp_prev:
                pos_mf.append(mf); neg_mf.append(0.0)
            elif tp < tp_prev:
                neg_mf.append(mf); pos_mf.append(0.0)
            else:
                pos_mf.append(0.0); neg_mf.append(0.0)
        tp_prev = tp
    pmf = sum(pos_mf)
    nmf = sum(neg_mf)
    if nmf == 0:
        return 100.0
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
    if avg_loss == 0:
        return 100.0
    return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))


# ─────────────────────────────────────────────
# DEMARK TD — исправленная логика
# Perfect Setup: low 8 или 9 ниже low 6 и 7 (buy).
# Отмена Countdown: цена вышла за Setup диапазон.
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
                        m9_signal  = True
                        m9_perfect = perfect
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
# МОЛОТ / SHOOTING STAR
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
            return {
                "btc_trend": btc_trend, "btc_ch": btc_ch,
                "btc_p": btc[-1][4], "alt_power": alt_power,
                "alt_ch": alt_diff, "btc_vol": np.mean(btc_moves),
            }
        except Exception as e:
            logging.warning(f"get_market_context попытка {attempt+1}/3: {e}")
            time.sleep(5)
    return {"btc_trend": "⚪️", "btc_ch": 0, "btc_p": 0,
            "alt_power": "⚪️", "alt_ch": 0, "btc_vol": 1.0}


def adaptive_threshold(base: int, btc_vol: float, is_priority: bool) -> int:
    threshold = base
    if btc_vol > 3.0:
        threshold += 2
    elif btc_vol < 0.5:
        threshold -= 1
    if is_priority:
        threshold -= 1
    return max(threshold, 3)


# ─────────────────────────────────────────────
# ОСНОВНОЙ ЦИКЛ
# ─────────────────────────────────────────────
def analyst_loop():
    sent_signals = {}
    logging.info("Аналитик v4.0 запущен.")

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

            active_swaps = [
                s for s, m in exchange.markets.items()
                if m.get('active') and m.get('type') == 'swap'
            ]

            vol_data = sorted(
                [{'s': s, 'v': tickers[s].get('quoteVolume', 0)}
                 for s in active_swaps if s in tickers],
                key=lambda x: x['v'], reverse=True
            )
            top250 = [x['s'] for x in vol_data[:250]]

            symbols = list(dict.fromkeys(
                [w for w in WATCHLIST if w in tickers] + top250
            ))

            for symbol in symbols:
                try:
                    ohlcv = None
                    for attempt in range(2):
                        try:
                            ohlcv = exchange.fetch_ohlcv(symbol, '4h', limit=100)
                            break
                        except ccxt.NetworkError as e:
                            if attempt == 0:
                                logging.warning(f"fetch_ohlcv retry {symbol}: {e}")
                                time.sleep(3)
                            else:
                                raise
                    if ohlcv is None or len(ohlcv) < 60:
                        continue

                    is_wl         = symbol in WATCHLIST
                    threshold     = adaptive_threshold(5, ctx['btc_vol'], is_wl)
                    current_price = ohlcv[-1][4]
                    closed        = ohlcv[:-1]
                    closes_closed = [x[4] for x in closed]

                    last_closed_ts = closed[-1][0]
                    candle_id      = last_closed_ts // 14_400_000
                    l_key          = f"{symbol}_{candle_id}_l"
                    s_key          = f"{symbol}_{candle_id}_s"

                    if l_key in sent_signals and s_key in sent_signals:
                        continue

                    # ── Базовые индикаторы ──
                    v_history  = [x[5] for x in closed[-21:-1]]
                    v_avg      = np.mean(v_history) if v_history else 1.0
                    v_rel      = closed[-1][5] / v_avg if v_avg > 0 else 1.0
                    v_zscore   = calc_volume_zscore(closed, period=20)

                    supports, resistances = get_pivot_levels(closed)
                    sup = supports[0]    if supports    else min(x[3] for x in closed[-60:])
                    res = resistances[0] if resistances else max(x[2] for x in closed[-60:])

                    rsi = calculate_rsi_wilder(closes_closed)
                    mfi = calculate_mfi(closed)
                    atr = calculate_atr(closed)

                    c_high  = closed[-1][2]
                    c_low   = closed[-1][3]
                    c_close = closed[-1][4]
                    buy_pressure = (c_close - c_low) / (c_high - c_low) if c_high != c_low else 0.5
                    imb = (buy_pressure - 0.5) * 200

                    # Изменение цены на закрытой свече
                    price_ch = ((closed[-1][4] - closed[-2][4]) / closed[-2][4] * 100) if len(closed) >= 2 else 0.0

                    # ── CVD уровень и дивергенция ──
                    cvd_level, cvd_total = calc_cvd_level(closed)
                    cvd_div_l = get_cvd_divergence(closed, 'long')
                    cvd_div_s = get_cvd_divergence(closed, 'short')

                    # ── RSI дивергенция ──
                    rsi_div_bull, rsi_div_bear = calc_rsi_divergence(closed)

                    # ── Молот ──
                    hammer_l = check_hammer(ohlcv, 'long')
                    hammer_s = check_hammer(ohlcv, 'short')

                    # ── DeMark TD ──
                    m9_l, m9_perfect_l, m13_l = update_td_counters(ohlcv, 'long')
                    m9_s, m9_perfect_s, m13_s = update_td_counters(ohlcv, 'short')

                    # ── ATR ──
                    stop_k, take_k, rr_str = dynamic_atr_multipliers(ctx['btc_vol'])
                    if atr:
                        stop_l   = current_price - stop_k * atr
                        target_l = current_price + take_k * atr
                        stop_s   = current_price + stop_k * atr
                        target_s = current_price - take_k * atr
                    else:
                        stop_l = stop_s = target_l = target_s = None

                    # ── Скоринг объёма ──
                    if v_rel >= 10 or v_zscore >= 3:
                        vol_score = 3
                        vol_label = f"🚀 Vol x{v_rel:.0f} Z:{v_zscore:.1f}σ"
                    elif v_rel >= 5 or v_zscore >= 2:
                        vol_score = 2
                        vol_label = f"📊 Vol x{v_rel:.1f} Z:{v_zscore:.1f}σ"
                    elif v_rel >= 1.8:
                        vol_score = 1
                        vol_label = f"📊 Vol x{v_rel:.1f}"
                    else:
                        vol_score = 0
                        vol_label = ""

                    wl_badge = " | ⭐️ WL" if is_wl else ""

                    # ── OI и фандинг (только если есть базовый сигнал) ──
                    # Запрашиваем только когда уже есть зацепка — экономим запросы
                    base_l = cvd_div_l or hammer_l or m9_l or m13_l or rsi_div_bull or vol_score >= 2
                    base_s = cvd_div_s or hammer_s or m9_s or m13_s or rsi_div_bear or vol_score >= 2

                    oi_chg = 0.0; oi_signal = 'neutral'
                    fr_val = 0.0; fr_signal = 'neutral'

                    if base_l or base_s:
                        oi_chg, oi_signal = get_oi_signal(symbol, price_ch)
                        fr_val, fr_signal = get_funding_signal(symbol)

                    # ════════════════════════════════
                    # ЛОНГ
                    # ════════════════════════════════
                    if l_key not in sent_signals:
                        l_score   = 0
                        l_details = []

                        # CVD дивергенция — цена ↓ но покупатели держат
                        if cvd_div_l:
                            l_score += 2; l_details.append("🔥 CVD Дивер")

                        # CVD уровень — накоплен на текущей цене
                        if cvd_level in ('bull', 'bull_div'):
                            l_score += 1; l_details.append("💹 CVD Уровень")

                        # RSI дивергенция
                        if rsi_div_bull:
                            l_score += 2; l_details.append("📉 RSI Дивер")

                        # Молот
                        if hammer_l:
                            l_score += 1; l_details.append("⚓️ Фитиль")

                        # RSI/MFI — только информационные, не в скор
                        # (убраны из скора — слишком часто при памп/дамп обманывают)

                        # Объём (Z-Score)
                        if vol_score > 0:
                            l_score += vol_score; l_details.append(vol_label)

                        # Pivot уровень
                        if current_price <= sup * 1.015:
                            l_score += 2; l_details.append("🧱 Pivot Support")
                        elif current_price <= min(x[3] for x in closed[-60:]) * 1.015:
                            l_score += 1; l_details.append("🧱 Уровень")

                        # OI
                        if oi_signal == 'bull':
                            l_score += 2; l_details.append(f"📈 OI +{oi_chg:.1f}%")
                        elif oi_signal == 'squeeze_dn':
                            l_score += 1; l_details.append(f"📈 OI сквиз")

                        # Фандинг
                        if fr_signal == 'bull':
                            l_score += 1; l_details.append(f"💸 FR {fr_val:.3f}%")

                        # DeMark
                        if m9_l:
                            pts = 3 if m9_perfect_l else 2
                            l_score += pts
                            l_details.append("⏱ M9 Setup ✨" if m9_perfect_l else "⏱ M9 Setup")
                        if m13_l:
                            l_score += 3; l_details.append("⏱ M13 Reversal")

                        # ── ВЕТО ─────────────────────────────────────────
                        # 1. BTC красный + нет сильного сигнала
                        btc_is_red   = ctx['btc_trend'] == "🔴"
                        is_strong_l  = (v_rel > 3.5 and imb > 75) or oi_signal == 'bull'
                        macro_veto_l = btc_is_red and not (is_strong_l or m13_l)

                        # 2. OI падает при росте цены = сквиз, слабое основание для лонга
                        # (не блокируем полностью, просто не добавляем баллы — уже учтено выше)

                        if l_score >= threshold and not macro_veto_l:
                            status  = "⚡️ СИЛЬНЕЕ РЫНКА" if is_strong_l and btc_is_red else "✅ ТРЕНД"
                            tv_link = build_tv_link(symbol)

                            atr_block = ""
                            if atr and stop_l and target_l:
                                atr_block = (
                                    f"───────────────────\n"
                                    f"🎯 Цель:  <code>{target_l:.6g}</code>  (+{((target_l/current_price-1)*100):.1f}%)\n"
                                    f"🛑 Стоп:  <code>{stop_l:.6g}</code>  (-{((1-stop_l/current_price)*100):.1f}%)\n"
                                    f"⚡️ ATR:   <code>{atr:.6g}</code> | R/R ≈ {rr_str}\n"
                                )

                            oi_line = ""
                            if oi_chg != 0:
                                oi_icon = "🟢" if oi_signal == 'bull' else ("🔴" if oi_signal == 'bear' else "⚪️")
                                oi_line = f"📈 OI: {oi_icon} {oi_chg:+.1f}% ({oi_signal})\n"

                            fr_line = ""
                            if fr_val != 0:
                                fr_line = f"💸 Фандинг: {fr_val:.4f}%\n"

                            cvd_icon = "🟢" if cvd_level in ('bull', 'bull_div') else "🔴"

                            msg = (
                                f"🚨 <b>ЛОНГ 4H ({l_score}/10){wl_badge}</b>\n"
                                f"Монета: <b>{symbol}</b> | {status}\n"
                                f"Цена: <code>{current_price:.6g}</code>\n"
                                f"Сигналы: {', '.join(l_details)}\n"
                                f"───────────────────\n"
                                f"📊 RSI: <b>{rsi:.1f}</b>  |  MFI: <b>{mfi:.1f}</b>\n"
                                f"⚖️ Дисбаланс: {'🟢' if imb > 0 else '🔴'} <b>{abs(imb):.0f}%</b>\n"
                                f"📦 Объём: <b>x{v_rel:.1f}</b> | Z: <b>{v_zscore:.1f}σ</b>\n"
                                f"{cvd_icon} CVD уровень: <b>{cvd_level}</b>\n"
                                f"{oi_line}"
                                f"{fr_line}"
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
                            logging.info(f"ЛОНГ: {symbol} score={l_score} thr={threshold} v_rel={v_rel:.1f} | {l_details}")

                    # ════════════════════════════════
                    # ШОРТ
                    # ════════════════════════════════
                    if s_key not in sent_signals:
                        s_score   = 0
                        s_details = []

                        if cvd_div_s:
                            s_score += 2; s_details.append("🔥 CVD Дивер")

                        if cvd_level in ('bear', 'bear_div'):
                            s_score += 1; s_details.append("💹 CVD Уровень")

                        if rsi_div_bear:
                            s_score += 2; s_details.append("📈 RSI Дивер")

                        if hammer_s:
                            s_score += 1; s_details.append("🏹 Фитиль вверх")

                        if vol_score > 0:
                            s_score += vol_score; s_details.append(vol_label)

                        if current_price >= res * 0.985:
                            s_score += 2; s_details.append("🧱 Pivot Resistance")
                        elif current_price >= max(x[2] for x in closed[-60:]) * 0.985:
                            s_score += 1; s_details.append("🧱 Уровень")

                        if oi_signal == 'bear':
                            s_score += 2; s_details.append(f"📉 OI +{oi_chg:.1f}%")
                        elif oi_signal == 'squeeze_up':
                            # Шорт-сквиз = цена растёт на ликвидациях шортов
                            # НЕ добавляем балл, но и не блокируем — это важное предупреждение
                            s_details.append(f"⚠️ Сквиз вверх")

                        if fr_signal == 'bear':
                            s_score += 1; s_details.append(f"💸 FR {fr_val:.3f}%")

                        if m9_s:
                            pts = 3 if m9_perfect_s else 2
                            s_score += pts
                            s_details.append("⏱ M9 Setup ✨" if m9_perfect_s else "⏱ M9 Setup")
                        if m13_s:
                            s_score += 3; s_details.append("⏱ M13 Reversal")

                        # ── ВЕТО ─────────────────────────────────────────
                        btc_is_green = ctx['btc_trend'] == "🟢"
                        is_strong_s  = (v_rel > 3.5 and imb < -75) or oi_signal == 'bear'

                        # КЛЮЧЕВОЕ ВЕТО: памп с большим объёмом → не шортим
                        # Если цена летит вверх сильно + большой объём → это не разворот
                        pump_veto = (v_rel > 8 and imb > 50 and price_ch > 5)

                        # BTC зелёный + нет сильного медвежьего сигнала
                        macro_veto_s = btc_is_green and not (is_strong_s or m13_s)

                        if s_score >= threshold and not macro_veto_s and not pump_veto:
                            status  = "⚡️ ПРОТИВ РЫНКА" if is_strong_s and btc_is_green else "🔻 ШОРТ"
                            tv_link = build_tv_link(symbol)

                            atr_block = ""
                            if atr and stop_s and target_s:
                                atr_block = (
                                    f"───────────────────\n"
                                    f"🎯 Цель:  <code>{target_s:.6g}</code>  (-{((1-target_s/current_price)*100):.1f}%)\n"
                                    f"🛑 Стоп:  <code>{stop_s:.6g}</code>  (+{((stop_s/current_price-1)*100):.1f}%)\n"
                                    f"⚡️ ATR:   <code>{atr:.6g}</code> | R/R ≈ {rr_str}\n"
                                )

                            oi_line = ""
                            if oi_chg != 0:
                                oi_icon = "🟢" if oi_signal == 'bull' else ("🔴" if oi_signal == 'bear' else "⚪️")
                                oi_line = f"📉 OI: {oi_icon} {oi_chg:+.1f}% ({oi_signal})\n"

                            fr_line = ""
                            if fr_val != 0:
                                fr_line = f"💸 Фандинг: {fr_val:.4f}%\n"

                            cvd_icon = "🟢" if cvd_level in ('bull', 'bull_div') else "🔴"

                            msg = (
                                f"🚨 <b>ШОРТ 4H ({s_score}/10){wl_badge}</b>\n"
                                f"Монета: <b>{symbol}</b> | {status}\n"
                                f"Цена: <code>{current_price:.6g}</code>\n"
                                f"Сигналы: {', '.join(s_details)}\n"
                                f"───────────────────\n"
                                f"📊 RSI: <b>{rsi:.1f}</b>  |  MFI: <b>{mfi:.1f}</b>\n"
                                f"⚖️ Дисбаланс: {'🔴' if imb > 0 else '🟢'} <b>{abs(imb):.0f}%</b>\n"
                                f"📦 Объём: <b>x{v_rel:.1f}</b> | Z: <b>{v_zscore:.1f}σ</b>\n"
                                f"{cvd_icon} CVD уровень: <b>{cvd_level}</b>\n"
                                f"{oi_line}"
                                f"{fr_line}"
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
                            logging.info(f"ШОРТ: {symbol} score={s_score} thr={threshold} v_rel={v_rel:.1f} | {s_details}")

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
            logging.info(
                f"Итерация завершена. Символов: {len(symbols)}. "
                f"Кэш: {len(sent_signals)}. BTC vol: {ctx['btc_vol']:.2f}%"
            )
            time.sleep(300)

        except ccxt.NetworkError as e:
            logging.error(f"Глобальная сетевая ошибка: {e}")
            bot_status["errors"] += 1
            time.sleep(60)
        except Exception as e:
            logging.error(f"Критическая ошибка: {e}")
            bot_status["errors"] += 1
            time.sleep(60)


@app.route('/health')
def health():
    uptime = str(datetime.now() - datetime.fromisoformat(bot_status["started_at"])).split('.')[0]
    return (
        f"✅ OK\n"
        f"Uptime: {uptime}\n"
        f"Итераций: {bot_status['iterations']}\n"
        f"Ошибок: {bot_status['errors']}\n"
        f"Сигналов отправлено: {bot_status['signals_sent']}\n"
        f"Последняя итерация: {bot_status['last_iteration']}"
    )


def keepalive_loop():
    time.sleep(30)
    port         = int(os.environ.get("PORT", 10000))
    external_url = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
    local_url    = f"http://localhost:{port}/health"
    while True:
        urls = ([f"{external_url}/health"] if external_url else []) + [local_url]
        for url in urls:
            try:
                r = requests.get(url, timeout=15)
                logging.info(f"Keepalive OK [{url}]: {r.status_code}")
                break
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
