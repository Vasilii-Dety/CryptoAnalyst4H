import ccxt
import requests
import time
import os
import logging
from datetime import datetime
from flask import Flask
import threading
import numpy as np
from collections import defaultdict, deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

@app.route('/')
def home():
    return f"✅ АНАЛИТИК v3.7 (DeMark Exhaustion) АКТИВЕН. Время: {datetime.now().strftime('%H:%M:%S')}"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
exchange = ccxt.mexc({'enableRateLimit': True, 'timeout': 30000, 'options': {'defaultType': 'swap'}})

# --- ГЛОБАЛЬНОЕ ХРАНИЛИЩЕ ДЛЯ DEMARK (TD) ---
# Храним состояние счетчиков для каждого символа и таймфрейма
# {symbol: {tf: {'s_count': 0, 'c_count': 0, 'last_s': None, 'in_c': False}}}
td_states = defaultdict(lambda: defaultdict(lambda: {'s_count': 0, 'c_count': 0, 'last_s': None, 'in_c': False, 'perfect': False}))

def send_msg(text):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        logging.error(f"Ошибка TG: {e}")

# --- ФУНКЦИЯ РАСЧЕТА DEMARK (SIMPLIFIED TD) ---
def update_td_counters(ohlcv, symbol, tf, mode='long'):
    state = td_states[symbol][tf]
    closes = [x[4] for x in ohlcv]
    if len(closes) < 30: return False, False # Нужно минимум 30 свечей истории
    
    current_close = closes[-1]
    last_ts = ohlcv[-1][0]
    
    # 1. TD SETUP (M9) - Сравнение с close 4 бара назад
    m9_buy = False
    m9_sell = False
    
    lookback_4 = closes[-5]
    if current_close < lookback_4: # Down setup (Buy potential)
        if mode == 'long':
            state['s_count'] += 1
            if state['s_count'] == 9:
                m9_buy = True
                state['s_count'] = 0 # Сброс после 9
                state['in_c'] = True # Начинаем обратный отсчет M13
                state['c_count'] = 0
                # Перфекционизм: Low 8 или 9 ниже Low 6 и 7 (для простоты опускаем)
        else: state['s_count'] = 0 # Сброс если не в лонг моде
    elif current_close > lookback_4: # Up setup (Sell potential)
        if mode == 'short':
            state['s_count'] += 1
            if state['s_count'] == 9:
                m9_sell = True
                state['s_count'] = 0
                state['in_c'] = True # Начинаем Countdown
                state['c_count'] = 0
        else: state['s_count'] = 0
    else: state['s_count'] = 0 # Флэт - сброс
    
    # 2. TD COUNTDOWN (M13) - Сравнение с close 2 бара назад после завершенного Setup
    m13_buy = False
    m13_sell = False
    
    if state['in_c']:
        lookback_2 = closes[-3]
        if mode == 'long' and current_close <= lookback_2:
            state['c_count'] += 1
            if state['c_count'] == 13:
                m13_buy = True
                state['in_c'] = False # Полный сброс
                state['c_count'] = 0
        elif mode == 'short' and current_close >= lookback_2:
            state['c_count'] += 1
            if state['c_count'] == 13:
                m13_sell = True
                state['in_c'] = False
                state['c_count'] = 0
        # Если M13 затянулась слишком долго (например, 20 баров) - сброс (упрощено)

    return (m9_buy, m13_buy) if mode == 'long' else (m9_sell, m13_sell)

# ... (остальные функции get_market_context, get_levels, calculate_rsi_wilder, get_cvd_data, check_reversal_patterns остаются прежними из v3.6) ...

def get_market_context():
    try:
        btc = exchange.fetch_ohlcv('BTC/USDT:USDT', '4h', limit=5)
        btc_ch = ((btc[-1][4] - btc[-2][4]) / btc[-2][4]) * 100
        btc_trend = "🟢" if btc_ch > -0.3 else "🔴"
        eth_btc = exchange.fetch_ohlcv('ETH/BTC', '4h', limit=5)
        eth_btc_ch = ((eth_btc[-1][4] - eth_btc[-2][4]) / eth_btc[-2][4]) * 100
        alt_power = "🚀" if eth_btc_ch > 0.1 else "⚓️"
        return {"btc_trend": btc_trend, "btc_ch": btc_ch, "btc_p": btc[-1][4], "alt_power": alt_power, "alt_ch": eth_btc_ch}
    except:
        return {"btc_trend": "⚪️", "btc_ch": 0, "btc_p": 0, "alt_power": "⚪️", "alt_ch": 0}

def get_levels(ohlcv):
    lows = [x[3] for x in ohlcv[-60:]]
    highs = [x[2] for x in ohlcv[-60:]]
    return min(lows), max(highs)

def analyst_loop():
    sent_signals = {}
    logging.info("Аналитик v3.7 (DeMark) запущен.")
    
    while True:
        try:
            ctx = get_market_context()
            exchange.load_markets()
            tickers = exchange.fetch_tickers()
            active_swaps = [s for s, m in exchange.markets.items() if m['active'] and m['type'] == 'swap']
            symbols = [x['s'] for x in sorted([{'s': s, 'v': tickers[s].get('quoteVolume', 0)} for s in active_swaps if s in tickers], key=lambda x: x['v'], reverse=True)[:150]]

            for symbol in symbols:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, '4h', limit=100)
                    if len(ohlcv) < 60: continue
                    
                    price = ohlcv[-1][4]
                    v_history = [x[5] for x in ohlcv[-21:-1]]
                    v_avg = np.mean(v_history)
                    v_rel = ohlcv[-1][5] / v_avg if v_avg > 0 else 1.0
                    sup, res = get_levels(ohlcv)
                    rsi = calculate_rsi_wilder([x[4] for x in ohlcv])
                    cvd = get_cvd_data(ohlcv)
                    
                    if ohlcv[-1][2] != ohlcv[-1][3]:
                        buy_pressure = (price - ohlcv[-1][3]) / (ohlcv[-1][2] - ohlcv[-1][3])
                    else: buy_pressure = 0.5
                    imb = (buy_pressure - 0.5) * 200
                    
                    # --- АНАЛИЗ DEMARK (TD) ---
                    m9_l, m13_l = update_td_counters(ohlcv, symbol, '4h', 'long')
                    
                    # --- СКОРИНГ ЛОНГ (Макс 10 баллов) ---
                    l_score = 0
                    l_details = []
                    div_l, hammer = check_reversal_patterns(ohlcv, cvd, 'long')
                    if div_l: l_score += 2; l_details.append("🔥 CVD Дивер")
                    if hammer: l_score += 1; l_details.append("⚓️ Фитиль")
                    if rsi < 35: l_score += 1; l_details.append("📉 RSI")
                    if v_rel > 1.8: l_score += 1; l_details.append(f"📊 Vol x{v_rel:.1f}")
                    if price <= sup * 1.015: l_score += 1; l_details.append("🧱 Уровень")
                    if ctx['alt_power'] == "🚀": l_score += 1 # Доп балл за силу альтов
                    
                    # Добавляем DeMark (сигналы истощения)
                    if m9_l: l_score += 2; l_details.append("⏱ M9 (Setup)")
                    if m13_l: l_score += 3; l_details.append("⏱ M13 Reversal ( Countdown)") # countdown сильнее

                    # СМАРТ-ВЕТО + АНАЛОГИЧНО ДЛЯ ШОРТ (можно добавить блок самостоятельно)
                    is_strong = (v_rel > 3.5 and imb > 75)
                    # Если есть сильный Демарк сигнал (M13), мы можем рискнуть лонговать даже на 🔴 BTC
                    marco_veto_allowed = (ctx['btc_trend'] == "🔴" and not (is_strong or m13_l))

                    if l_score >= 4 and not marco_veto_allowed and l_key not in sent_signals:
                        l_key = f"{symbol}_{ohlcv[-1][0]}_l_td"
                        status = "⚡️ СИЛЬНЕЕ РЫНКА / DeMark M13" if (is_strong or m13_l) and ctx['btc_trend'] == "🔴" else "✅ ТРЕНД"
                        dm_status = f" | {DOMINATION_DOMINATOR_DOMINATION_DOMINATOR[domination]:.0f}% Buyers" if 'imb' in locals() else ""
                        msg = (f"🚨 <b>СИЛЬНЫЙ ЛОНГ 4H ({l_score}/10)</b>\n"
                               f"Монета: <b>{symbol}</b> | {status}\n"
                               f"Цена: <code>{price}</code> | RSI: {rsi:.1f}\n"
                               f"Сигналы: {', '.join(l_details)}\n"
                               f"───────────────────\n"
                               f"⚖️ Дисбаланс: {'🟢' if imb>0 else '🔴'} <b>{abs(imb):.0f}%</b>\n"
                               f"📊 Рост объема: <b>x{v_rel:.1f}</b>\n"
                               f"───────────────────\n"
                               f"🌍 <b>МАРКЕТ:</b>\n"
                               f"👑 BTC: {ctx['btc_trend']} {ctx['btc_ch']:.1f}% ({ctx['btc_p']:.0f})\n"
                               f"🚀 Alt-Strength: {ctx['alt_power']} {ctx['alt_ch']:.2f}% (ETH/BTC)\n"
                               f"───────────────────\n"
                               f"🔗 <a href='https://www.tradingview.com/chart/?symbol=MEXC:{symbol.replace('/', '').replace(':USDT', '.P')}'>Открыть в TradingView</a>")
                        send_msg(msg)
                        sent_signals[l_key] = time.time()
                    time.sleep(0.1)
                except: continue
            sent_signals = {k: v for k, v in sent_signals.items() if v > (time.time() - 86400)}
            time.sleep(300)
        except: time.sleep(60)

threading.Thread(target=analyst_loop, daemon=True).start()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
