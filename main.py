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
    return f"✅ АНАЛИТИК (4H Развороты | 180 монет) АКТИВЕН. Время: {datetime.now().strftime('%H:%M:%S')}"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

exchange = ccxt.mexc({'enableRateLimit': True, 'timeout': 30000, 'options': {'defaultType': 'swap'}})

def get_funding_status(val):
    abs_val = abs(val)
    if abs_val < 0.0003: return "🟢"
    if abs_val < 0.001: return "⚠️"
    return "🚨"

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

def calculate_cvd_logic(ohlcv, mode='long'):
    if len(ohlcv) < 6: return False
    cvd = []
    cum = 0.0
    for c in ohlcv:
        delta = c[5] if c[4] >= c[1] else -c[5]
        cum += delta
        cvd.append(cum)
    if mode == 'long':
        return ohlcv[-1][4] > ohlcv[-1][1] and cvd[-1] > 0 and (cvd[-3] > cvd[-2] < cvd[-1])
    else:
        return ohlcv[-1][4] < ohlcv[-1][1] and cvd[-1] < 0 and (cvd[-3] < cvd[-2] > cvd[-1])

def analyst_loop():
    sent_signals = {} 
    logging.info("Аналитик запущен (180 монет, 4H).")
    
    while True:
        try:
            try:
                exchange.load_markets()
                tickers = exchange.fetch_tickers()
            except:
                time.sleep(10)
                continue

            active_swaps = []
            for symbol, ticker_data in tickers.items():
                market = exchange.markets.get(symbol)
                if market and market.get('active') and market.get('type') == 'swap' and market.get('quote') == 'USDT':
                    vol = ticker_data.get('quoteVolume', 0) or 0
                    active_swaps.append({'symbol': symbol, 'vol': vol})
            
            # Топ-180 по объему
            active_swaps.sort(key=lambda x: x['vol'], reverse=True)
            symbols = [x['symbol'] for x in active_swaps][:180]

            for symbol in symbols:
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    price = ticker['last']
                    funding = float(ticker.get('info', {}).get('fundingRate', 0) or 0)
                    f_pct = funding * 100
                    f_status = get_funding_status(funding)
                    tv = f"https://www.tradingview.com/chart/?symbol=MEXC:{symbol.replace('/', '').replace(':USDT', '.P')}"

                    ohlcv_1h = exchange.fetch_ohlcv(symbol, '1h', limit=50)
                    if ohlcv_1h and len(ohlcv_1h) == 50:
                        ma50 = sum([c[4] for c in ohlcv_1h]) / 50
                        ma_text = "🟢 Выше MA50" if price > ma50 else "🔴 Ниже MA50"
                    else:
                        ma_text = "⚪ Нет данных"

                    ohlcv_4h = exchange.fetch_ohlcv(symbol, '4h', limit=30)
                    if not ohlcv_4h or len(ohlcv_4h) < 20: continue
                    ts_4h = ohlcv_4h[-1][0]
                    rsi_4h = calculate_rsi_wilder([c[4] for c in ohlcv_4h])
                    vol_avg_4h = sum(c[5] for c in ohlcv_4h[-6:-1]) / 5

                    l_count = 0
                    if funding < -0.0005: l_count += 1
                    if rsi_4h < 45: l_count += 1
                    if calculate_cvd_logic(ohlcv_4h, 'long'): l_count += 1
                    if ohlcv_4h[-1][5] > vol_avg_4h * 1.3: l_count += 1
                    low_24h = min(c[3] for c in ohlcv_4h[-6:])
                    if (price - low_24h) / low_24h < 0.03: l_count += 1

                    l_key = f"{symbol}_{ts_4h}_long"
                    if l_count >= 3 and rsi_4h < 50 and l_key not in sent_signals:
                        msg = (f"🚨 <b>СИЛЬНЫЙ ЛОНГ 4H ({l_count}/5)</b>\n"
                               f"Монета: {symbol}\n"
                               f"Цена: <code>{price}</code>\n"
                               f"RSI (4h): {rsi_4h:.1f}\n"
                               f"Тренд 1h: {ma_text}\n"
                               f"Фандинг: {f_status} <code>{f_pct:.4f}%</code>\n"
                               f"🔗 <a href='{tv}'>График</a>")
                        send_msg(msg)
                        sent_signals[l_key] = time.time()

                    s_count = 0
                    if funding > 0.0005: s_count += 1
                    if rsi_4h > 65: s_count += 1
                    if calculate_cvd_logic(ohlcv_4h, 'short'): s_count += 1
                    if ohlcv_4h[-1][5] > vol_avg_4h * 1.3: s_count += 1
                    high_24h = max(c[2] for c in ohlcv_4h[-6:])
                    if (high_24h - price) / price < 0.03: s_count += 1

                    s_key = f"{symbol}_{ts_4h}_short"
                    if s_count >= 3 and rsi_4h > 50 and s_key not in sent_signals:
                        msg = (f"❄️ <b>СИЛЬНЫЙ ШОРТ 4H ({s_count}/5)</b>\n"
                               f"Монета: {symbol}\n"
                               f"Цена: <code>{price}</code>\n"
                               f"RSI (4h): {rsi_4h:.1f}\n"
                               f"Тренд 1h: {ma_text}\n"
                               f"Фандинг: {f_status} <code>{f_pct:.4f}%</code>\n"
                               f"🔗 <a href='{tv}'>График</a>")
                        send_msg(msg)
                        sent_signals[s_key] = time.time()

                    time.sleep(0.3)
                except:
                    continue

            now = time.time()
            sent_signals = {k: v for k, v in sent_signals.items() if v > (now - 86400)}
            time.sleep(90) # Аналитику спешить некуда
            
        except Exception as e:
            logging.error(f"Глобальная ошибка Аналитика: {e}")
            time.sleep(30)

threading.Thread(target=analyst_loop, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
