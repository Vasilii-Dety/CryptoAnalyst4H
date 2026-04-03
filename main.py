import ccxt
import requests
import time
import os
import logging
from datetime import datetime
from flask import Flask
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

@app.route('/')
def home():
    return f"✅ АНАЛИТИК v3.0 (Anti-Knife + Divergence) АКТИВЕН. Время: {datetime.now().strftime('%H:%M:%S')}"

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
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [max(0, d) for d in deltas]
    losses = [max(0, -d) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
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
    """
    Улучшенная проверка: Дивергенция за 5 свечей + Свечной паттерн (Фитиль)
    """
    if len(ohlcv) < 10: return False, False
    
    # Последняя свеча
    o, h, l, c, v = ohlcv[-1][1:6]
    body = abs(c - o)
    
    # Анализ фитилей (Price Action)
    if mode == 'long':
        lower_wick = min(o, c) - l
        is_hammer = lower_wick > (body * 2) and c > l # Длинный хвост снизу
        
        # Дивергенция за 5 свечей
        price_min_5 = min(x[3] for x in ohlcv[-6:-1])
        cvd_min_5 = min(cvd[-6:-1])
        divergence = l <= price_min_5 and cvd[-1] > cvd_min_5
        return divergence, is_hammer
    else:
        upper_wick = h - max(o, c)
        is_star = upper_wick > (body * 2) and c < h # Длинный хвост сверху
        
        price_max_5 = max(x[2] for x in ohlcv[-6:-1])
        cvd_max_5 = max(cvd[-6:-1])
        divergence = h >= price_max_5 and cvd[-1] < cvd_max_5
        return divergence, is_star

def analyst_loop():
    sent_signals = {} 
    logging.info("Аналитик v3.0 запущен.")
    
    while True:
        try:
            exchange.load_markets()
            tickers = exchange.fetch_tickers()
            
            active_swaps = []
            for symbol, t in tickers.items():
                market = exchange.markets.get(symbol)
                if market and market.get('active') and market.get('type') == 'swap' and market.get('quote') == 'USDT':
                    active_swaps.append({'symbol': symbol, 'vol': t.get('quoteVolume', 0) or 0})
            
            active_swaps.sort(key=lambda x: x['vol'], reverse=True)
            symbols = [x['symbol'] for x in active_swaps][:180]

            for symbol in symbols:
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    price = ticker['last']
                    funding = float(ticker.get('info', {}).get('fundingRate', 0) or 0)
                    
                    ohlcv_4h = exchange.fetch_ohlcv(symbol, '4h', limit=30)
                    if not ohlcv_4h or len(ohlcv_4h) < 20: continue
                    
                    cvd_full = get_cvd_data(ohlcv_4h)
                    rsi_4h = calculate_rsi_wilder([c[4] for c in ohlcv_4h])
                    ts_4h = ohlcv_4h[-1][0]
                    
                    # --- ЛОГИКА ЛОНГ (Балльная система) ---
                    l_score = 0
                    l_details = []
                    
                    div_l, hammer = check_reversal_patterns(ohlcv_4h, cvd_full, 'long')
                    
                    if div_l: l_score += 2; l_details.append("🔥 CVD Дивергенция")
                    if hammer: l_score += 1; l_details.append("⚓️ Выкуп (Фитиль)")
                    if rsi_4h < 35: l_score += 1; l_details.append("📉 RSI Перепроданность")
                    if funding < -0.0005: l_score += 1; l_details.append("💰 Отриц. фандинг")

                    l_key = f"{symbol}_{ts_4h}_long"
                    # Требуем минимум 3 балла, при этом Дивергенция или Фитиль ОБЯЗАТЕЛЬНЫ
                    if l_score >= 3 and (div_l or hammer) and l_key not in sent_signals:
                        msg = (f"🚨 <b>СИЛЬНЫЙ ЛОНГ 4H ({l_score}/5)</b>\n"
                               f"Монета: {symbol}\n"
                               f"Цена: <code>{price}</code>\n"
                               f"RSI (4h): {rsi_4h:.1f}\n"
                               f"Сигналы: {', '.join(l_details)}\n"
                               f"🔗 <a href='https://www.tradingview.com/chart/?symbol=MEXC:{symbol.replace('/', '').replace(':USDT', '.P')}'>График</a>")
                        send_msg(msg)
                        sent_signals[l_key] = time.time()

                    # --- ЛОГИКА ШОРТ ---
                    s_score = 0
                    s_details = []
                    div_s, star = check_reversal_patterns(ohlcv_4h, cvd_full, 'short')
                    
                    if div_s: s_score += 2; s_details.append("🔥 CVD Дивергенция")
                    if star: s_score += 1; s_details.append("☄️ Давление (Фитиль)")
                    if rsi_4h > 65: s_score += 1; s_details.append("📈 RSI Перекупленность")
                    if funding > 0.0005: s_score += 1; s_details.append("💰 Полож. фандинг")

                    s_key = f"{symbol}_{ts_4h}_short"
                    if s_score >= 3 and (div_s or star) and s_key not in sent_signals:
                        msg = (f"❄️ <b>СИЛЬНЫЙ ШОРТ 4H ({s_score}/5)</b>\n"
                               f"Монета: {symbol}\n"
                               f"Цена: <code>{price}</code>\n"
                               f"RSI (4h): {rsi_4h:.1f}\n"
                               f"Сигналы: {', '.join(s_details)}\n"
                               f"🔗 <a href='https://www.tradingview.com/chart/?symbol=MEXC:{symbol.replace('/', '').replace(':USDT', '.P')}'>График</a>")
                        send_msg(msg)
                        sent_signals[s_key] = time.time()

                    time.sleep(0.3)
                except: continue

            sent_signals = {k: v for k, v in sent_signals.items() if v > (time.time() - 86400)}
            time.sleep(180) 
        except: time.sleep(30)

threading.Thread(target=analyst_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
