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
    return f"✅ АНАЛИТИК v3.4 (Ultimate) АКТИВЕН. Время: {datetime.now().strftime('%H:%M:%S')}"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

exchange = ccxt.mexc({'enableRateLimit': True, 'timeout': 30000, 'options': {'defaultType': 'swap'}})

def send_msg(text):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        logging.error(f"Ошибка TG: {e}")

def calculate_rsi_wilder(closes, period=14):
    if len(closes) < period + 1: return 50.0
    deltas = np.diff(closes)
    gains = deltas * (deltas > 0)
    losses = -deltas * (deltas < 0)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0: return 100
    return 100 - (100 / (1 + (avg_gain / avg_loss)))

def get_cvd_data(ohlcv):
    cvd = []
    cum = 0.0
    for c in ohlcv:
        delta = c[5] if c[4] >= c[1] else -c[5]
        cum += delta
        cvd.append(cum)
    return cvd

def check_reversal_patterns(ohlcv, cvd, mode='long'):
    if len(ohlcv) < 10: return False, False
    o, h, l, c, v = ohlcv[-1][1:6]
    body = abs(c - o)
    if mode == 'long':
        lower_wick = min(o, c) - l
        is_hammer = lower_wick > (body * 1.5)
        price_min_5 = min(x[3] for x in ohlcv[-6:-1])
        cvd_min_5 = min(cvd[-6:-1])
        divergence = l <= price_min_5 and cvd[-1] > cvd_min_5
        return divergence, is_hammer
    else:
        upper_wick = h - max(o, c)
        is_star = upper_wick > (body * 1.5)
        price_max_5 = max(x[2] for x in ohlcv[-6:-1])
        cvd_max_5 = max(cvd[-6:-1])
        divergence = h >= price_max_5 and cvd[-1] < cvd_max_5
        return divergence, is_star

def analyst_loop():
    sent_signals = {} 
    logging.info("Аналитик v3.4 запущен.")
    
    while True:
        try:
            exchange.load_markets()
            tickers = exchange.fetch_tickers()
            active_swaps = [s for s, m in exchange.markets.items() if m['active'] and m['type'] == 'swap' and m['quote'] == 'USDT']
            
            # Сортировка по объему (Топ 150)
            symbols_to_check = []
            for s in active_swaps:
                if s in tickers:
                    symbols_to_check.append({'s': s, 'v': tickers[s].get('quoteVolume', 0)})
            symbols_to_check.sort(key=lambda x: x['v'], reverse=True)
            symbols = [x['s'] for x in symbols_to_check[:150]]

            for symbol in symbols:
                try:
                    ohlcv_4h = exchange.fetch_ohlcv(symbol, '4h', limit=60)
                    if not ohlcv_4h or len(ohlcv_4h) < 25: continue
                    
                    # Данные текущей свечи
                    ts_4h = ohlcv_4h[-1][0]
                    o_curr, h_curr, l_curr, price, v_curr = ohlcv_4h[-1][1:6]
                    
                    # 1. ОБЪЕМ (Средний за 20 свечей)
                    v_history = [x[5] for x in ohlcv_4h[-21:-1]]
                    v_avg = np.mean(v_history)
                    v_rel = v_curr / v_avg if v_avg > 0 else 1.0
                    
                    # 2. ДИСБАЛАНС (Pressure)
                    if h_curr != l_curr:
                        buy_pressure = (price - l_curr) / (h_curr - l_curr)
                    else:
                        buy_pressure = 0.5
                    imbalance_pct = ((buy_pressure - 0.5) * 2) * 100
                    dom_str = f"🟢 <b>{abs(imbalance_pct):.0f}% Покупатели</b>" if imbalance_pct > 0 else f"🔴 <b>{abs(imbalance_pct):.0f}% Продавцы</b>"
                    
                    # 3. ИНДИКАТОРЫ
                    cvd_full = get_cvd_data(ohlcv_4h)
                    rsi_4h = calculate_rsi_wilder([c[4] for c in ohlcv_4h])
                    candle_percent = ((price - o_curr) / o_curr) * 100
                    
                    # --- ЛОГИКА ЛОНГ (Баллы + Фильтры) ---
                    l_score = 0
                    l_details = []
                    div_l, hammer = check_reversal_patterns(ohlcv_4h, cvd_full, 'long')
                    
                    if div_l: l_score += 2; l_details.append("🔥 CVD Дивергенция")
                    if hammer: l_score += 1; l_details.append("⚓️ Выкуп (Фитиль)")
                    if rsi_4h < 35: l_score += 1; l_details.append("📉 RSI Перепроданность")
                    if v_rel > 1.5: l_score += 1; l_details.append(f"📊 Объем (x{v_rel:.1f})")

                    # ВЕТО ФИЛЬТРЫ
                    veto_long = (rsi_4h >= 50) or (candle_percent < -4.0) # RSI Вето + Anti-Train
                    
                    l_key = f"{symbol}_{ts_4h}_long"
                    if l_score >= 3 and (div_l or hammer) and not veto_long and l_key not in sent_signals:
                        msg = (f"🚨 <b>СИЛЬНЫЙ ЛОНГ 4H ({l_score}/5)</b>\n"
                               f"Монета: <b>{symbol}</b>\n"
                               f"Цена: <code>{price}</code> | RSI: {rsi_4h:.1f}\n"
                               f"Сигналы: {', '.join(l_details)}\n"
                               f"───────────────────\n"
                               f"📈 Рост объема: <b>x{v_rel:.1f}</b>\n"
                               f"⚖️ Дисбаланс: {dom_str}\n"
                               f"───────────────────\n"
                               f"🔗 <a href='https://www.tradingview.com/chart/?symbol=MEXC:{symbol.replace('/', '').replace(':USDT', '.P')}'>Открыть график</a>")
                        send_msg(msg)
                        sent_signals[l_key] = time.time()

                    # --- ЛОГИКА ШОРТ (Баллы + Фильтры) ---
                    s_score = 0
                    s_details = []
                    div_s, star = check_reversal_patterns(ohlcv_4h, cvd_full, 'short')
                    
                    if div_s: s_score += 2; s_details.append("🔥 CVD Дивергенция")
                    if star: s_score += 1; s_details.append("☄️ Давление (Фитиль)")
                    if rsi_4h > 65: s_score += 1; s_details.append("📈 RSI Перекупленность")
                    if v_rel > 1.5: s_score += 1; s_details.append(f"📊 Объем (x{v_rel:.1f})")

                    # ВЕТО ФИЛЬТРЫ
                    veto_short = (rsi_4h <= 50) or (candle_percent > 4.0) # RSI Вето + Anti-Train
                    
                    s_key = f"{symbol}_{ts_4h}_short"
                    if s_score >= 3 and (div_s or star) and not veto_short and s_key not in sent_signals:
                        msg = (f"❄️ <b>СИЛЬНЫЙ ШОРТ 4H ({s_score}/5)</b>\n"
                               f"Монета: <b>{symbol}</b>\n"
                               f"Цена: <code>{price}</code> | RSI: {rsi_4h:.1f}\n"
                               f"Сигналы: {', '.join(s_details)}\n"
                               f"───────────────────\n"
                               f"📉 Рост объема: <b>x{v_rel:.1f}</b>\n"
                               f"⚖️ Дисбаланс: {dom_str}\n"
                               f"───────────────────\n"
                               f"🔗 <a href='https://www.tradingview.com/chart/?symbol=MEXC:{symbol.replace('/', '').replace(':USDT', '.P')}'>Открыть график</a>")
                        send_msg(msg)
                        sent_signals[s_key] = time.time()
                        
                    time.sleep(0.2)
                except: continue
            
            # Очистка старых сигналов
            sent_signals = {k: v for k, v in sent_signals.items() if v > (time.time() - 86400)}
            time.sleep(300) # Пауза между кругами
        except: time.sleep(60)

threading.Thread(target=analyst_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
