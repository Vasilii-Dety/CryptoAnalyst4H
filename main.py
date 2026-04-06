import ccxt
import requests
import time
import os
import logging
from datetime import datetime
from flask import Flask
import threading
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

@app.route('/')
def home():
    return f"✅ АНАЛИТИК v3.7 (DeMark Exhaustion) АКТИВЕН. Время: {datetime.now().strftime('%H:%M:%S')}"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not TELEGRAM_TOKEN or not CHAT_ID:
    logging.warning("⚠️ TELEGRAM_TOKEN или CHAT_ID не заданы! Сообщения не будут отправляться.")

exchange = ccxt.mexc({
    'enableRateLimit': True,
    'timeout': 30000,
    'options': {'defaultType': 'swap'}
})

# Состояние TD для каждого символа и таймфрейма
# {symbol: {tf: {mode: {'s_count': 0, 'c_count': 0, 'in_c': False}}}}
td_states = defaultdict(lambda: defaultdict(lambda: {
    'long':  {'s_count': 0, 'c_count': 0, 'in_c': False},
    'short': {'s_count': 0, 'c_count': 0, 'in_c': False},
}))


def send_msg(text):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logging.warning("Сообщение не отправлено: токен или chat_id не заданы.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(
            url,
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10
        )
        if not resp.ok:
            logging.error(f"TG ответил ошибкой: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        logging.error(f"Ошибка отправки TG: {e}")


# --- DEMARK TD (корректная реализация) ---
# Setup M9: 9 последовательных свечей, каждая закрывается ниже (для buy) / выше (для sell)
#           закрытия свечи, которая была 4 бара назад.
# Countdown M13: после завершения Setup считаем бары, где close <= low 2 бара назад (buy)
#                или close >= high 2 бара назад (sell). Необязательно подряд.
def update_td_counters(ohlcv, symbol, tf, mode='long'):
    """
    Итерируется по всем свечам ohlcv (кроме первых 4, нужных для lookback).
    Возвращает (m9_signal: bool, m13_signal: bool).
    """
    state = td_states[symbol][tf][mode]
    closes = [x[4] for x in ohlcv]
    lows   = [x[3] for x in ohlcv]
    highs  = [x[2] for x in ohlcv]

    if len(closes) < 14:
        return False, False

    m9_signal  = False
    m13_signal = False

    for i in range(4, len(closes)):
        c  = closes[i]
        c4 = closes[i - 4]

        if mode == 'long':
            setup_condition = c < c4
        else:
            setup_condition = c > c4

        if setup_condition:
            if not state['in_c']:
                state['s_count'] += 1
                if state['s_count'] >= 9:
                    if i == len(closes) - 1:
                        m9_signal = True
                    state['s_count'] = 0
                    state['in_c'] = True
                    state['c_count'] = 0
        else:
            if not state['in_c']:
                state['s_count'] = 0

        if state['in_c'] and i >= 2:
            if mode == 'long':
                countdown_condition = c <= lows[i - 2]
            else:
                countdown_condition = c >= highs[i - 2]

            if countdown_condition:
                state['c_count'] += 1
                if state['c_count'] >= 13:
                    if i == len(closes) - 1:
                        m13_signal = True
                    state['in_c'] = False
                    state['c_count'] = 0

            if state['c_count'] > 25:
                logging.debug(f"TD Countdown отменён (>25 баров): {symbol} {tf} {mode}")
                state['in_c'] = False
                state['c_count'] = 0

    return m9_signal, m13_signal


# --- RSI по методу Wilder ---
def calculate_rsi_wilder(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# --- Суррогатный CVD (на основе OHLCV, без тиков) ---
def get_cvd_data(ohlcv):
    cvd = []
    cumulative = 0.0
    for candle in ohlcv:
        o, h, l, c, v = candle[1], candle[2], candle[3], candle[4], candle[5]
        if h != l:
            buy_ratio = (c - l) / (h - l)
        else:
            buy_ratio = 0.5
        delta = (buy_ratio - 0.5) * 2 * v
        cumulative += delta
        cvd.append(cumulative)
    return cvd


# --- Паттерны разворота ---
def check_reversal_patterns(ohlcv, cvd, mode='long'):
    """
    Возвращает (cvd_divergence: bool, hammer: bool).
    """
    if len(ohlcv) < 10 or len(cvd) < 10:
        return False, False

    closes = [x[4] for x in ohlcv]
    lows   = [x[3] for x in ohlcv]
    highs  = [x[2] for x in ohlcv]

    cvd_div = False
    hammer  = False

    lookback = 10
    if mode == 'long':
        price_new_low   = closes[-1] < min(closes[-lookback:-1])
        cvd_not_new_low = cvd[-1] > min(cvd[-lookback:-1])
        cvd_div = price_new_low and cvd_not_new_low
    else:
        price_new_high    = closes[-1] > max(closes[-lookback:-1])
        cvd_not_new_high  = cvd[-1] < max(cvd[-lookback:-1])
        cvd_div = price_new_high and cvd_not_new_high

    candle = ohlcv[-1]
    o, h, l, c = candle[1], candle[2], candle[3], candle[4]
    body       = abs(c - o)
    full_range = h - l if h != l else 1e-9

    if mode == 'long':
        lower_wick = min(o, c) - l
        hammer = (lower_wick / full_range) > 0.6 and body / full_range < 0.3
    else:
        upper_wick = h - max(o, c)
        hammer = (upper_wick / full_range) > 0.6 and body / full_range < 0.3

    return cvd_div, hammer


def get_market_context():
    try:
        btc = exchange.fetch_ohlcv('BTC/USDT:USDT', '4h', limit=5)
        btc_ch = ((btc[-1][4] - btc[-2][4]) / btc[-2][4]) * 100
        btc_trend = "🟢" if btc_ch > -0.3 else "🔴"
        eth_btc = exchange.fetch_ohlcv('ETH/BTC', '4h', limit=5)
        eth_btc_ch = ((eth_btc[-1][4] - eth_btc[-2][4]) / eth_btc[-2][4]) * 100
        alt_power = "🚀" if eth_btc_ch > 0.1 else "⚓️"
        return {
            "btc_trend": btc_trend,
            "btc_ch": btc_ch,
            "btc_p": btc[-1][4],
            "alt_power": alt_power,
            "alt_ch": eth_btc_ch
        }
    except Exception as e:
        logging.error(f"Ошибка get_market_context: {e}")
        return {"btc_trend": "⚪️", "btc_ch": 0, "btc_p": 0, "alt_power": "⚪️", "alt_ch": 0}


def get_levels(ohlcv):
    lows  = [x[3] for x in ohlcv[-60:]]
    highs = [x[2] for x in ohlcv[-60:]]
    return min(lows), max(highs)


def analyst_loop():
    sent_signals = {}
    logging.info("Аналитик v3.7 (DeMark) запущен.")

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
                    logging.error(f"Ошибка перезагрузки рынков: {e}")

            ctx = get_market_context()

            try:
                tickers = exchange.fetch_tickers()
            except Exception as e:
                logging.error(f"Ошибка fetch_tickers: {e}")
                time.sleep(60)
                continue

            active_swaps = [
                s for s, m in exchange.markets.items()
                if m.get('active') and m.get('type') == 'swap'
            ]
            symbols = [
                x['s'] for x in sorted(
                    [{'s': s, 'v': tickers[s].get('quoteVolume', 0)}
                     for s in active_swaps if s in tickers],
                    key=lambda x: x['v'], reverse=True
                )[:150]
            ]

            for symbol in symbols:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, '4h', limit=100)
                    if len(ohlcv) < 60:
                        continue

                    closes = [x[4] for x in ohlcv]
                    price  = closes[-1]

                    v_history = [x[5] for x in ohlcv[-21:-1]]
                    v_avg = np.mean(v_history) if v_history else 1.0
                    v_rel = ohlcv[-1][5] / v_avg if v_avg > 0 else 1.0

                    sup, res = get_levels(ohlcv)
                    rsi      = calculate_rsi_wilder(closes)
                    cvd      = get_cvd_data(ohlcv)

                    candle_high = ohlcv[-1][2]
                    candle_low  = ohlcv[-1][3]
                    if candle_high != candle_low:
                        buy_pressure = (price - candle_low) / (candle_high - candle_low)
                    else:
                        buy_pressure = 0.5
                    imb = (buy_pressure - 0.5) * 200

                    m9_l,  m13_l  = update_td_counters(ohlcv, symbol, '4h', 'long')
                    m9_s,  m13_s  = update_td_counters(ohlcv, symbol, '4h', 'short')

                    div_l, hammer_l = check_reversal_patterns(ohlcv, cvd, 'long')
                    div_s, hammer_s = check_reversal_patterns(ohlcv, cvd, 'short')

                    # ======== ЛОНГ ========
                    l_score   = 0
                    l_details = []

                    if div_l:    l_score += 2; l_details.append("🔥 CVD Дивер")
                    if hammer_l: l_score += 1; l_details.append("⚓️ Фитиль")
                    if rsi < 35: l_score += 1; l_details.append("📉 RSI")
                    if v_rel > 1.8: l_score += 1; l_details.append(f"📊 Vol x{v_rel:.1f}")
                    if price <= sup * 1.015: l_score += 1; l_details.append("🧱 Уровень")
                    if ctx['alt_power'] == "🚀": l_score += 1
                    if m9_l:  l_score += 2; l_details.append("⏱ M9 Setup")
                    if m13_l: l_score += 3; l_details.append("⏱ M13 Reversal")

                    is_strong_l  = v_rel > 3.5 and imb > 75
                    btc_is_red   = ctx['btc_trend'] == "🔴"
                    macro_veto_l = btc_is_red and not (is_strong_l or m13_l)

                    l_key = f"{symbol}_{ohlcv[-1][0]}_l"
                    if l_score >= 4 and not macro_veto_l and l_key not in sent_signals:
                        if (is_strong_l or m13_l) and btc_is_red:
                            status = "⚡️ СИЛЬНЕЕ РЫНКА / DeMark M13"
                        else:
                            status = "✅ ТРЕНД"

                        tv_sym = symbol.replace('/', '').replace(':USDT', '.P')
                        msg = (
                            f"🚨 <b>СИЛЬНЫЙ ЛОНГ 4H ({l_score}/10)</b>\n"
                            f"Монета: <b>{symbol}</b> | {status}\n"
                            f"Цена: <code>{price:.6g}</code> | RSI: {rsi:.1f}\n"
                            f"Сигналы: {', '.join(l_details)}\n"
                            f"───────────────────\n"
                            f"⚖️ Дисбаланс: {'🟢' if imb > 0 else '🔴'} <b>{abs(imb):.0f}%</b>\n"
                            f"📊 Рост объёма: <b>x{v_rel:.1f}</b>\n"
                            f"───────────────────\n"
                            f"🌍 <b>МАРКЕТ:</b>\n"
                            f"👑 BTC: {ctx['btc_trend']} {ctx['btc_ch']:.1f}% ({ctx['btc_p']:.0f})\n"
                            f"🚀 Alt-Strength: {ctx['alt_power']} {ctx['alt_ch']:.2f}% (ETH/BTC)\n"
                            f"───────────────────\n"
                            f"🔗 <a href='https://www.tradingview.com/chart/?symbol=MEXC:{tv_sym}'>Открыть в TradingView</a>"
                        )
                        send_msg(msg)
                        sent_signals[l_key] = time.time()
                        logging.info(f"ЛОНГ сигнал: {symbol} | score={l_score} | {l_details}")

                    # ======== ШОРТ ========
                    s_score   = 0
                    s_details = []

                    if div_s:    s_score += 2; s_details.append("🔥 CVD Дивер")
                    if hammer_s: s_score += 1; s_details.append("🏹 Фитиль вверх")
                    if rsi > 65: s_score += 1; s_details.append("📈 RSI")
                    if v_rel > 1.8: s_score += 1; s_details.append(f"📊 Vol x{v_rel:.1f}")
                    if price >= res * 0.985: s_score += 1; s_details.append("🧱 Уровень")
                    if ctx['alt_power'] == "⚓️": s_score += 1
                    if m9_s:  s_score += 2; s_details.append("⏱ M9 Setup")
                    if m13_s: s_score += 3; s_details.append("⏱ M13 Reversal")

                    is_strong_s  = v_rel > 3.5 and imb < -75
                    btc_is_green = ctx['btc_trend'] == "🟢"
                    macro_veto_s = btc_is_green and not (is_strong_s or m13_s)

                    s_key = f"{symbol}_{ohlcv[-1][0]}_s"
                    if s_score >= 4 and not macro_veto_s and s_key not in sent_signals:
                        if (is_strong_s or m13_s) and btc_is_green:
                            status = "⚡️ ПРОТИВ РЫНКА / DeMark M13"
                        else:
                            status = "🔻 ШОРТ"

                        tv_sym = symbol.replace('/', '').replace(':USDT', '.P')
                        msg = (
                            f"🚨 <b>СИЛЬНЫЙ ШОРТ 4H ({s_score}/10)</b>\n"
                            f"Монета: <b>{symbol}</b> | {status}\n"
                            f"Цена: <code>{price:.6g}</code> | RSI: {rsi:.1f}\n"
                            f"Сигналы: {', '.join(s_details)}\n"
                            f"───────────────────\n"
                            f"⚖️ Дисбаланс: {'🟢' if imb > 0 else '🔴'} <b>{abs(imb):.0f}%</b>\n"
                            f"📊 Рост объёма: <b>x{v_rel:.1f}</b>\n"
                            f"───────────────────\n"
                            f"🌍 <b>МАРКЕТ:</b>\n"
                            f"👑 BTC: {ctx['btc_trend']} {ctx['btc_ch']:.1f}% ({ctx['btc_p']:.0f})\n"
                            f"🚀 Alt-Strength: {ctx['alt_power']} {ctx['alt_ch']:.2f}% (ETH/BTC)\n"
                            f"───────────────────\n"
                            f"🔗 <a href='https://www.tradingview.com/chart/?symbol=MEXC:{tv_sym}'>Открыть в TradingView</a>"
                        )
                        send_msg(msg)
                        sent_signals[s_key] = time.time()
                        logging.info(f"ШОРТ сигнал: {symbol} | score={s_score} | {s_details}")

                    time.sleep(0.15)

                except ccxt.RateLimitExceeded:
                    logging.warning(f"Rate limit при обработке {symbol}, пауза 30с")
                    time.sleep(30)
                except ccxt.NetworkError as e:
                    logging.error(f"Сетевая ошибка {symbol}: {e}")
                except Exception as e:
                    logging.error(f"Ошибка обработки {symbol}: {e}")

            now = time.time()
            sent_signals = {k: v for k, v in sent_signals.items() if now - v < 86400}

            active_set = set(symbols)
            stale = [s for s in list(td_states.keys()) if s not in active_set]
            for s in stale:
                del td_states[s]

            logging.info(f"Итерация завершена. Активных сигналов в кэше: {len(sent_signals)}")
            time.sleep(300)

        except ccxt.NetworkError as e:
            logging.error(f"Глобальная сетевая ошибка: {e}")
            time.sleep(60)
        except Exception as e:
            logging.error(f"Критическая ошибка в цикле: {e}")
            time.sleep(60)


threading.Thread(target=analyst_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
