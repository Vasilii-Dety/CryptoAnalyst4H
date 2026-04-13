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
    return f"✅ АНАЛИТИК v4.1 АКТИВЕН. Время: {datetime.now().strftime('%H:%M:%S')}"

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
# Z > 2 = аномалия, Z > 3 = экстремум
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
# Накопленная дельта покупок/продаж по истории.
# bull_div = цена ↓ но CVD держится → покупатели сильнее
# bear_div = цена ↑ но CVD падает → продавцы давят
# bull / bear = общее накопление по всей истории
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

    # Дивергенции — самые сильные сигналы
    if price_new_low  and not cvd_new_low:   return 'bull_div', total_cvd
    if price_new_high and not cvd_new_high:  return 'bear_div', total_cvd
    if total_cvd > 0:                        return 'bull', total_cvd
    return 'bear', total_cvd


# ─────────────────────────────────────────────
# RSI ДИВЕРГЕНЦИЯ
# Считаем RSI на lookback последних свечей
# и ищем расхождение между ценой и RSI
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
# CVD ДИВЕРГЕНЦИЯ (для скоринга)
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
# ОТКРЫТЫЙ ИНТЕРЕС (OI)
# ─────────────────────────────────────────────
def get_oi_signal(symbol, price_change_pct):
    """
    MEXC не поддерживает OI через ccxt (has=False).
    Используем прямой REST API: contract.mexc.com

    Запрашиваем текущий OI дважды с паузой — считаем изменение.
    Это менее точно чем история но работает без авторизации.

    Символ BTC/USDT:USDT → BTC_USDT для MEXC REST API.

    Возвращает (oi_chg_pct, signal, label).
    signal: 'bull' / 'bear' / 'squeeze_up' / 'squeeze_dn' / 'neutral'
    """
    try:
        # Конвертируем символ в формат MEXC futures REST API
        mexc_sym = symbol.replace('/USDT:USDT', '_USDT')
        if '/' in mexc_sym:
            mexc_sym = mexc_sym.replace('/', '_')

        # Используем kline эндпоинт для получения OI по свечам
        # MEXC: /api/v1/contract/kline/openInterest/{symbol}
        url = f"https://contract.mexc.com/api/v1/contract/kline/openInterest/{mexc_sym}"
        params = {'interval': 'Hour4', 'limit': 3}

        resp = requests.get(url, params=params, timeout=8)
        if not resp.ok:
            return 0.0, 'neutral', '⚪️ OI: нет данных'

        data = resp.json()
        if not data.get('success') or not data.get('data'):
            return 0.0, 'neutral', '⚪️ OI: нет данных'

        oi_list = data['data'].get('openInterest', [])
        if not oi_list or len(oi_list) < 2:
            return 0.0, 'neutral', '⚪️ OI: нет данных'

        oi_prev = float(oi_list[-2] or 0)
        oi_curr = float(oi_list[-1] or 0)

        if oi_prev == 0:
            return 0.0, 'neutral', '⚪️ OI: нет данных'

        oi_chg = (oi_curr - oi_prev) / oi_prev * 100

        if abs(oi_chg) < 1.0:
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
# ФАНДИНГ (информационный + мягкий сигнал)
# Отрицательный = шортисты платят лонгам → давление вверх
# Положительный = лонги платят шортам → давление вниз
# ─────────────────────────────────────────────
def get_funding_signal(symbol):
    """
    Возвращает (rate_pct, signal, label).
    signal: 'bull' / 'bear' / 'neutral'
    """
    try:
        fr   = exchange.fetch_funding_rate(symbol)
        rate = float(fr.get('fundingRate', 0) or 0) * 100
        if rate < -0.02:
            signal = 'bull'
            label  = f"🟢 Фандинг: {rate:.3f}% (шортистов много → давление вверх)"
        elif rate > 0.05:
            signal = 'bear'
            label  = f"🔴 Фандинг: {rate:.3f}% (лонгистов много → давление вниз)"
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
        return [c for c in clusters
                if sum(1 for p in points if abs(p - c) / c <= tol) >= 2]

    supports    = cluster(pivots_low, tolerance)
    resistances = cluster(pivots_high, tolerance)
    cp = closes[-1]
    return (sorted([s for s in supports if s < cp], reverse=True),
            sorted([r for r in resistances if r > cp]))


# ─────────────────────────────────────────────
# ATR (Wilder)
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
# DEMARK TD — полный пересчёт с нуля
#
# M9 Setup: 9 последовательных свечей где
#   close < close[-4] (buy) / > close[-4] (sell)
#   Прерывание = сброс счётчика
#   Perfect Setup: low 8 или 9 ниже low 6 и 7 (buy)
#
# M13 Countdown: НЕ ПОДРЯД, каждый бар где
#   close <= low[-2] (buy) / >= high[-2] (sell)
#   ОТМЕНА: цена вышла за setup_high (buy)
#   Защита: > 30 баров без завершения → сброс
#
# Сигналит True ТОЛЬКО если завершилось на
# ПОСЛЕДНЕЙ закрытой свече
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
            # Отмена Countdown если цена вышла за Setup диапазон
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

            # Countdown: не подряд
            if i >= 2:
                cd_cond = (c <= lows[i - 2]) if mode == 'long' else (c >= highs[i - 2])
                if cd_cond:
                    c_count += 1
                    if c_count == 13:
                        if i == last_idx:
                            m13_signal = True
                        in_c = False; c_count = 0; setup_high = None; setup_low = None

            # Защита от бесконечного Countdown
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
# РЫНОЧНЫЙ КОНТЕКСТ (3 попытки)
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
    logging.info("Аналитик v4.1 запущен.")

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
                    l_key = f"{symbol}_{candle_id}_l"
                    s_key = f"{symbol}_{candle_id}_s"

                    if l_key in sent_signals and s_key in sent_signals:
                        continue

                    # ── Объём и Z-Score ──
                    v_history = [x[5] for x in closed[-21:-1]]
                    v_avg     = np.mean(v_history) if v_history else 1.0
                    v_rel     = closed[-1][5] / v_avg if v_avg > 0 else 1.0
                    v_zscore  = calc_volume_zscore(closed)

                    # Скоринг объёма: v_rel И z_score учитываются вместе
                    if v_rel >= 10 or v_zscore >= 3.0:
                        vol_score = 3; vol_label = f"🚀 Vol x{v_rel:.0f} Z:{v_zscore:.1f}σ ВЗРЫВ"
                    elif v_rel >= 5 or v_zscore >= 2.0:
                        vol_score = 2; vol_label = f"📊 Vol x{v_rel:.1f} Z:{v_zscore:.1f}σ"
                    elif v_rel >= 1.8:
                        vol_score = 1; vol_label = f"📊 Vol x{v_rel:.1f}"
                    else:
                        vol_score = 0; vol_label = ""

                    # ── Индикаторы ──
                    rsi = calculate_rsi_wilder(closes_closed)
                    mfi = calculate_mfi(closed)
                    atr = calculate_atr(closed)

                    # ── Дисбаланс свечи ──
                    c_high  = closed[-1][2]; c_low = closed[-1][3]; c_close = closed[-1][4]
                    buy_pressure = (c_close - c_low) / (c_high - c_low) if c_high != c_low else 0.5
                    imb = (buy_pressure - 0.5) * 200

                    # Изменение цены на закрытой свече
                    price_ch = ((closed[-1][4] - closed[-2][4]) / closed[-2][4] * 100) \
                               if len(closed) >= 2 else 0.0

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

                    # ── Pivot уровни ──
                    supports, resistances = get_pivot_levels(closed)
                    sup = supports[0]    if supports    else min(x[3] for x in closed[-60:])
                    res = resistances[0] if resistances else max(x[2] for x in closed[-60:])

                    # ── ATR ──
                    stop_k, take_k, rr_str = dynamic_atr_multipliers(ctx['btc_vol'])
                    if atr:
                        stop_l = current_price - stop_k * atr; target_l = current_price + take_k * atr
                        stop_s = current_price + stop_k * atr; target_s = current_price - take_k * atr
                    else:
                        stop_l = stop_s = target_l = target_s = None

                    # ── OI и фандинг ──
                    # Запрашиваем ВСЕГДА когда есть хоть один базовый сигнал.
                    # OI критичен для понимания движения денег.
                    has_any_signal = (cvd_div_l or cvd_div_s or hammer_l or hammer_s or
                                      m9_l or m9_s or m13_l or m13_s or
                                      rsi_div_bull or rsi_div_bear or vol_score >= 2 or
                                      current_price <= sup * 1.015 or
                                      current_price >= res * 0.985)

                    # Инициализируем как "нет данных" — ВСЕГДА будет показано в алерте
                    oi_chg    = 0.0
                    oi_signal = 'neutral'
                    oi_label  = '⚪️ OI: нет данных'
                    fr_val    = 0.0
                    fr_signal = 'neutral'
                    fr_label  = '⚪️ Фандинг: нет данных'

                    if has_any_signal:
                        oi_chg, oi_signal, oi_label = get_oi_signal(symbol, price_ch)
                        fr_val, fr_signal, fr_label  = get_funding_signal(symbol)
                        time.sleep(0.1)

                    wl_badge = " | ⭐️ WL" if is_wl else ""

                    # ════════════════════════════════════════
                    # СКОРИНГ — ЛОГИКА ДЕНЕГ
                    #
                    # Основной принцип: сигнал должен подтвердить
                    # что ДЕНЬГИ движутся в нужную сторону:
                    #
                    # СИЛЬНЫЕ (движение денег):
                    #   OI bull (+2 лонг) / OI bear (+2 шорт)
                    #   CVD дивергенция (+2) — умные деньги держат
                    #   RSI дивергенция (+2) — истощение тренда
                    #   M9 Setup (+2/+3) — DeMark истощение
                    #   M13 Reversal (+3) — DeMark разворот
                    #
                    # СРЕДНИЕ (подтверждение):
                    #   Pivot Support/Resistance (+2)
                    #   CVD уровень — зона накопления (+1)
                    #   Vol x5+ / Z>2 (+2) — аномальный поток денег
                    #   Фандинг bull/bear (+1) — перекос позиций
                    #
                    # ЛЁГКИЕ (дополнение):
                    #   Молот/Фитиль (+1)
                    #   Vol x1.8 (+1)
                    #   RSI <30/>70 (+1) — только экстремум
                    # ════════════════════════════════════════

                    # ════════════════════════
                    # ЛОНГ
                    # ════════════════════════
                    if l_key not in sent_signals:
                        l_score = 0; l_details = []

                        # Движение денег — главные сигналы
                        if oi_signal == 'bull':
                            l_score += 2; l_details.append(f"💹 OI +{oi_chg:.1f}%")
                        elif oi_signal == 'bear':
                            l_score -= 1  # деньги идут против лонга — штраф

                        if cvd_div_l:
                            l_score += 2; l_details.append("🔥 CVD Дивер")

                        if rsi_div_bull:
                            l_score += 2; l_details.append("📉 RSI Дивер")

                        if m9_l:
                            pts = 3 if m9_perfect_l else 2
                            l_score += pts
                            l_details.append("⏱ M9 Setup ✨" if m9_perfect_l else "⏱ M9 Setup")
                        if m13_l:
                            l_score += 3; l_details.append("⏱ M13 Reversal")

                        # Подтверждающие сигналы
                        if cvd_level in ('bull', 'bull_div'):
                            l_score += 1; l_details.append("📍 CVD Уровень")

                        if current_price <= sup * 1.015:
                            l_score += 2; l_details.append("🧱 Pivot Support")
                        elif current_price <= min(x[3] for x in closed[-60:]) * 1.015:
                            l_score += 1; l_details.append("🧱 Уровень")

                        if vol_score > 0:
                            l_score += vol_score; l_details.append(vol_label)

                        if fr_signal == 'bull':
                            l_score += 1; l_details.append(f"💸 Фандинг {fr_val:.3f}%")

                        # Лёгкие сигналы
                        if hammer_l:
                            l_score += 1; l_details.append("⚓️ Фитиль")

                        if rsi < 30:
                            l_score += 1; l_details.append(f"📉 RSI {rsi:.0f}")
                        elif mfi < 20:
                            l_score += 1; l_details.append(f"💰 MFI {mfi:.0f}")

                        l_score = max(l_score, 0)

                        # ── ВЕТО для лонга ──────────────────────────────
                        btc_is_red   = ctx['btc_trend'] == "🔴"
                        is_strong_l  = oi_signal == 'bull' or m13_l or (v_rel > 5 and imb > 60)
                        macro_veto_l = btc_is_red and not is_strong_l

                        # Дамп-вето: палка вниз с огромным объёмом
                        # При таком движении лонг преждевременен
                        dump_veto_l  = v_rel > 10 and imb < -60 and price_ch < -5

                        if l_score >= threshold and not macro_veto_l and not dump_veto_l:
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

                            # CVD метка
                            cvd_emoji = {"bull": "🟢", "bull_div": "🟢✨",
                                         "bear": "🔴", "bear_div": "🔴✨"}.get(cvd_level, "⚪️")

                            msg = (
                                f"🚨 <b>ЛОНГ 4H ({l_score}/10){wl_badge}</b>\n"
                                f"Монета: <b>{symbol}</b> | {status}\n"
                                f"Цена: <code>{current_price:.6g}</code>\n"
                                f"Сигналы: {', '.join(l_details)}\n"
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
                                         f"OI={oi_signal}({oi_chg:.1f}%) z={v_zscore:.1f} | {l_details}")

                    # ════════════════════════
                    # ШОРТ
                    # ════════════════════════
                    if s_key not in sent_signals:
                        s_score = 0; s_details = []

                        # Движение денег
                        if oi_signal == 'bear':
                            s_score += 2; s_details.append(f"💹 OI +{oi_chg:.1f}%")
                        elif oi_signal == 'bull':
                            s_score -= 1  # деньги против шорта — штраф

                        if cvd_div_s:
                            s_score += 2; s_details.append("🔥 CVD Дивер")

                        if rsi_div_bear:
                            s_score += 2; s_details.append("📈 RSI Дивер")

                        if m9_s:
                            pts = 3 if m9_perfect_s else 2
                            s_score += pts
                            s_details.append("⏱ M9 Setup ✨" if m9_perfect_s else "⏱ M9 Setup")
                        if m13_s:
                            s_score += 3; s_details.append("⏱ M13 Reversal")

                        # Подтверждающие
                        if cvd_level in ('bear', 'bear_div'):
                            s_score += 1; s_details.append("📍 CVD Уровень")

                        if current_price >= res * 0.985:
                            s_score += 2; s_details.append("🧱 Pivot Resistance")
                        elif current_price >= max(x[2] for x in closed[-60:]) * 0.985:
                            s_score += 1; s_details.append("🧱 Уровень")

                        if vol_score > 0:
                            s_score += vol_score; s_details.append(vol_label)

                        if fr_signal == 'bear':
                            s_score += 1; s_details.append(f"💸 Фандинг {fr_val:.3f}%")

                        # Лёгкие
                        if hammer_s:
                            s_score += 1; s_details.append("🏹 Фитиль вверх")

                        if rsi > 70:
                            s_score += 1; s_details.append(f"📈 RSI {rsi:.0f}")
                        elif mfi > 80:
                            s_score += 1; s_details.append(f"💰 MFI {mfi:.0f}")

                        s_score = max(s_score, 0)

                        # ── ВЕТО для шорта ──────────────────────────────
                        btc_is_green = ctx['btc_trend'] == "🟢"
                        is_strong_s  = oi_signal == 'bear' or m13_s or (v_rel > 5 and imb < -60)
                        macro_veto_s = btc_is_green and not is_strong_s

                        # Памп-вето: палка вверх с огромным объёмом
                        # Шортить памп с x8+ объёма = SKYAI/WET кейс
                        pump_veto_s  = v_rel > 8 and imb > 60 and price_ch > 3

                        if s_score >= threshold and not macro_veto_s and not pump_veto_s:
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

                            msg = (
                                f"🚨 <b>ШОРТ 4H ({s_score}/10){wl_badge}</b>\n"
                                f"Монета: <b>{symbol}</b> | {status}\n"
                                f"Цена: <code>{current_price:.6g}</code>\n"
                                f"Сигналы: {', '.join(s_details)}\n"
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
                                         f"OI={oi_signal}({oi_chg:.1f}%) z={v_zscore:.1f} | {s_details}")

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
