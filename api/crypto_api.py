import logging
import pandas as pd
import ccxt
import threading
import time
import json
from typing import Dict, List, Optional
from .base_api import BaseAPI
from config.settings import MONITORING_CONFIG, TRADING_CONFIG

try:
    import websocket
except ImportError:
    websocket = None

logger = logging.getLogger(__name__)


class UpbitAPI(BaseAPI):
    """업비트 API 구현"""
    
    def __init__(self, api_key: str, api_secret: str):
        super().__init__(api_key, api_secret)
        self.exchange = None
        self.use_websocket = False
        self.ws_manager = None
        self.price_cache = {}  # 실시간 가격 캐시 {symbol: price}
        self.lock = threading.Lock()
        self.callbacks = [] # 실시간 가격 콜백 리스트
    
    def connect(self):
        """업비트 API 연결"""
        try:
            self.exchange = ccxt.upbit({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'createMarketBuyOrderRequiresPrice': False,
                    'defaultType': 'spot',
                },
                'timeout': 10000, # [Request] 타임아웃 10초 설정
            })
            
            # 웹소켓 사용 여부 확인
            self.use_websocket = MONITORING_CONFIG.get("websocket_enabled", False)
            mode_msg = " (WebSocket ✅)" if self.use_websocket else " (REST API ⚠️)"
            
            # [New] 연결 및 권한 검증 (시장 데이터 로드 + 잔액 조회)
            self.exchange.load_markets()
            self.exchange.fetch_balance()
            
            logger.info(f"✅ 업비트 API 연결 및 검증 성공{mode_msg}")
            
        except Exception as e:
            logger.error(f"업비트 API 연결 오류: {e}")
            self.exchange = None
            raise e # 메인에서 예외를 처리할 수 있도록 전파
    
    def disconnect(self):
        """업비트 API 연결 종료"""
        if self.ws_manager:
            try:
                self.ws_manager.terminate()
                logger.info("업비트 WebSocket 연결 종료")
            except Exception as e:
                logger.error(f"WebSocket 종료 오류: {e}")
        
        self.exchange = None
        logger.info("업비트 API 연결 종료")

    def add_price_callback(self, callback):
        """실시간 가격 업데이트 콜백 등록"""
        self.callbacks.append(callback)

    def subscribe_websocket(self, symbols: List[str]):
        """웹소켓 구독 시작 (실시간 시세 수신)"""
        if not self.use_websocket or not symbols:
            return

        try:
            import pyupbit
            
            # 기존 연결 종료
            if self.ws_manager:
                self.ws_manager.terminate()
            
            # 심볼 변환 (BTC/KRW -> KRW-BTC)
            upbit_codes = [s.replace('/', '-') for s in symbols]
            self.code_map = {s.replace('/', '-'): s for s in symbols}
            
            # WebSocketManager 시작 (별도 프로세스/스레드로 동작)
            self.ws_manager = pyupbit.WebSocketManager("ticker", upbit_codes)
            
            # 데이터 수집 스레드 시작
            thread = threading.Thread(target=self._ws_worker)
            thread.daemon = True
            thread.start()
            
            logger.info(f"📡 WebSocket 구독 시작: {len(symbols)}개 종목")
            
        except ImportError:
            logger.error("❌ pyupbit 라이브러리가 없어 WebSocket을 사용할 수 없습니다.")
            self.use_websocket = False
        except Exception as e:
            logger.error(f"❌ WebSocket 시작 오류: {e}")
            self.use_websocket = False

    def _ws_worker(self):
        """웹소켓 데이터 처리 워커"""
        while self.use_websocket:
            try:
                if self.ws_manager is None:
                    time.sleep(1)
                    continue

                data = self.ws_manager.get()
                if data and 'code' in data and 'trade_price' in data:
                    code = data['code']
                    price = float(data['trade_price'])
                    
                    symbol = None
                    # 캐시 업데이트
                    if code in self.code_map:
                        symbol = self.code_map[code]
                        with self.lock:
                            self.price_cache[symbol] = price
                    
                    # 콜백 실행 (RiskManager 등 실시간 처리용)
                    if symbol:
                        for callback in self.callbacks:
                            try:
                                callback(symbol, price)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")

            except Exception as e:
                logger.error(f"⚠️ WebSocket 연결 끊김 또는 오류: {e}")
                
                # 재연결 시도
                if self.use_websocket:
                    logger.info("🔄 WebSocket 재연결 시도 중...")
                    time.sleep(3)
                    try:
                        if self.ws_manager:
                            self.ws_manager.terminate()
                        
                        import pyupbit
                        codes = list(self.code_map.keys())
                        self.ws_manager = pyupbit.WebSocketManager("ticker", codes)
                        logger.info("✅ WebSocket 재연결 성공")
                    except Exception as reconnect_e:
                        logger.error(f"❌ WebSocket 재연결 실패: {reconnect_e}")
                        time.sleep(5)
    
    def get_balance(self) -> Dict:
        """계좌 잔액 조회"""
        try:
            balance = self.exchange.fetch_balance()
            return {
                "total": balance.get("total", {}),
                "free": balance.get("free", {}),
                "used": balance.get("used", {}),
            }
        except Exception as e:
            logger.error(f"잔액 조회 오류: {e}")
            return {}
    
    def get_price(self, symbol: str) -> float:
        """현재가 조회"""
        # 1. 웹소켓 캐시 확인
        if self.use_websocket:
            with self.lock:
                if symbol in self.price_cache:
                    return self.price_cache[symbol]
            # [최적화] 웹소켓 사용 시 REST API Fallback 차단 (429 에러 방지)
            # 데이터가 아직 수신되지 않았으면 0.0 반환 -> 메인 루프에서 스킵됨
            return 0.0
        
        # 2. 웹소켓 미사용 시에만 REST API 호출
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"{symbol} 현재가 조회 오류: {e}")
            return 0.0
    
    def get_ticker(self, symbol: str) -> Dict:
        """티커 정보 조회 (호가 포함)"""
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"{symbol} 티커 조회 오류: {e}")
            return {}

    def get_ohlcv(self, symbol: str, timeframe: str = "1d", limit: int = 200, count: int = None, min_required_data: int = 200) -> pd.DataFrame:
        """OHLCV 데이터 조회 (데이터 개수 검증 로직 추가)"""
        # [Request 1] limit 파라미터 지원 (count와 호환)
        if count is None:
            count = limit
            
        # [Rate Limit] API 호출 간격 강제 (429 에러 방지)
        time.sleep(0.1)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Upbit는 요청당 최대 200개 제한 -> 200개 초과 시 반복 조회(Pagination)
                limit = 200
                if count <= limit:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=count)
                else:
                    ohlcv = []
                    remaining = count
                    end_date = None
                    
                    while remaining > 0:
                        fetch_limit = min(remaining, limit)
                        params = {}
                        if end_date:
                            params['to'] = end_date
                        
                        current_ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=fetch_limit, params=params)
                        if not current_ohlcv:
                            break
                            
                        ohlcv = current_ohlcv + ohlcv
                        remaining -= len(current_ohlcv)
                        
                        # 다음 요청을 위해 가장 오래된 데이터의 시간 설정
                        first_timestamp = current_ohlcv[0][0]
                        end_date = self.exchange.iso8601(first_timestamp)
                        # Pagination 사이에도 딜레이 추가
                        time.sleep(0.2)

                if not ohlcv:
                    return pd.DataFrame()

                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                # 중복 제거 (Pagination 경계)
                df.drop_duplicates(subset=['timestamp'], inplace=True)
                df.sort_values('timestamp', inplace=True)
                df.set_index('timestamp', inplace=True)
                
                # [데이터 검증] 최소 요구량 확인
                if len(df) < min_required_data:
                    logger.warning(f"⚠️ {symbol} 데이터 부족 (요청: {count}, 최소: {min_required_data}, 수신: {len(df)}). 스킵합니다.")
                    return pd.DataFrame()

                # [데이터 검증] 요청한 개수보다 부족하더라도, 확보된 데이터로 최대한 진행하도록 수정
                # (ML 학습 시 2000개를 요청하는데, 1500개만 있어도 학습은 가능해야 함)
                if len(df) < count:
                    logger.warning(f"⚠️ {symbol} 데이터 일부 부족 (요청: {count}, 수신: {len(df)}). 확보된 데이터로 진행합니다.")

                # [실시간 업데이트] 마지막 캔들의 현재가를 웹소켓 데이터로 최신화 (라이브 매매용)
                if self.use_websocket and symbol in self.price_cache and not df.empty:
                    current_price = self.price_cache[symbol]
                    df.iloc[-1, df.columns.get_loc('close')] = current_price
                    # 고가/저가 갱신
                    if current_price > df.iloc[-1]['high']:
                        df.iloc[-1, df.columns.get_loc('high')] = current_price
                    if current_price < df.iloc[-1]['low']:
                        df.iloc[-1, df.columns.get_loc('low')] = current_price
                
                return df

            except (ccxt.RateLimitExceeded, ccxt.DDoSProtection) as e:
                wait_time = (attempt + 1) * 2.0  # 2초, 4초, 6초 대기
                logger.warning(f"⚠️ {symbol} API 요청 제한(429). {wait_time}초 대기 후 재시도... ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"{symbol} OHLCV 데이터 조회 오류: {e}")
                return pd.DataFrame()
        
        return pd.DataFrame()
    
    def get_positions(self) -> List[Dict]:
        """보유 포지션 조회 (평단가 포함)"""
        try:
            balance = self.exchange.fetch_balance()
            positions = []
            # Upbit의 경우 info 필드에 원본 데이터(평단가 포함)가 있음
            if 'info' in balance:
                for item in balance['info']:
                    currency = item['currency']
                    if currency == 'KRW':
                        continue
                    
                    qty = float(item['balance']) + float(item['locked'])
                    avg_price = float(item['avg_buy_price'])
                    
                    if qty > 0:
                        symbol = f"{currency}/KRW"  # KRW 마켓 가정
                        positions.append({
                            'symbol': symbol,
                            'quantity': qty,
                            'entry_price': avg_price
                        })
            return positions
        except Exception as e:
            logger.error(f"포지션 조회 오류: {e}")
            return []

    def get_tick_size(self, price: float) -> float:
        """가격대별 호가 단위(Tick Size) 조회"""
        if price >= 2000000: return 1000
        elif price >= 1000000: return 500
        elif price >= 500000: return 100
        elif price >= 100000: return 50
        elif price >= 10000: return 10
        elif price >= 1000: return 1
        elif price >= 100: return 0.1
        elif price >= 10: return 0.01
        elif price >= 1: return 0.001
        else: return 0.0001

    def adjust_price_unit(self, price: float) -> float:
        """업비트 호가 단위(Tick Size) 보정"""
        tick = self.get_tick_size(price)
        
        # 호가 단위에 맞춰 버림 처리 (매수/매도 공통 안전하게)
        return float(int(price / tick) * tick)

    def buy(self, symbol: str, quantity: float, price: Optional[float] = None, **kwargs) -> Dict:
        """매수 주문 (재시도 로직 포함)"""
        # [요청사항] price가 없으면 '공격적 지정가' 주문 로직 수행
        if price is None:
            return self._buy_aggressive(symbol, quantity)

        if price:
            # [요청사항 1] 호가 단위 보정
            price = self.adjust_price_unit(price)

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                if price:
                    order = self.exchange.create_limit_buy_order(symbol, quantity, price)
                else:
                    order = self.exchange.create_market_buy_order(symbol, quantity)
                
                # [로그 상세화] 주문 ID 및 타임스탬프 기록
                order_id = order.get('id', 'unknown')
                order_ts = order.get('timestamp', int(time.time()*1000))
                logger.info(f"매수 주문 성공: {symbol} {quantity} (ID: {order_id}, Time: {order_ts})")
                return order
            
            except (ccxt.NetworkError, ccxt.RateLimitExceeded, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable) as e:
                if attempt < max_retries:
                    error_type = type(e).__name__
                    logger.warning(f"🚀 [RETRY] {symbol} 매수 주문 재시도 중... (사유: {error_type}) ({attempt+1}/{max_retries})")
                    time.sleep(0.5)
                    
                    # 안전장치: 방금 던진 주문이 들어갔는지 확인
                    try:
                        open_orders = self.get_open_orders(symbol)
                        now_ms = int(time.time() * 1000)
                        for o in open_orders:
                            # 매수, 수량 일치, 최근 10초 내 생성
                            if o['side'] == 'buy' and abs(float(o['amount']) - quantity) < 0.00000001:
                                if price and abs(float(o['price']) - price) > 0.00000001:
                                    continue
                                if (now_ms - o['timestamp']) < 10000:
                                    logger.info(f"♻️ 재시도 전 기존 주문 확인됨: {o['id']}")
                                    return o
                    except Exception as check_e:
                        logger.warning(f"중복 주문 확인 중 오류: {check_e}")
                    
                    continue
                else:
                    logger.error(f"매수 주문 최종 실패: {e} | Symbol: {symbol}, Price: {price}, Qty: {quantity}")
                    return {}
            except Exception as e:
                # [로그 상세화] 실패 시 시도 값 기록
                logger.error(f"매수 주문 오류: {e} | Symbol: {symbol}, Price: {price}, Qty: {quantity}")
                return {}
        return {}
    
    def _buy_aggressive(self, symbol: str, quantity: float) -> Dict:
        """공격적 지정가 매수 (추격형)"""
        slippage_ticks = TRADING_CONFIG["crypto"].get("slippage_ticks", 2)
        wait_sec = TRADING_CONFIG["crypto"].get("order_wait_seconds", 5)
        
        for attempt in range(3): # 최대 3회 추격
            try:
                ticker = self.get_ticker(symbol)
                ask_price = float(ticker['ask'])
                tick_size = self.get_tick_size(ask_price)
                
                # 현재가 + N틱 (공격적)
                target_price = ask_price + (tick_size * slippage_ticks)
                target_price = self.adjust_price_unit(target_price)
                
                logger.info(f"📉 [SLIPPAGE_PROTECTION] 매수: 현재가({ask_price:,.0f}) 대비 +{slippage_ticks}틱({target_price:,.0f})으로 지정가 제출 ({attempt+1}/3)")
                
                order = self.exchange.create_limit_buy_order(symbol, quantity, target_price)
                
                # 체결 대기
                time.sleep(wait_sec)
                
                # 상태 확인
                order_info = self.exchange.fetch_order(order['id'], symbol)
                if order_info['status'] == 'closed':
                    logger.info(f"✅ 공격적 매수 체결 완료: {symbol}")
                    return order_info
                
                # 미체결 시 취소 후 재시도 (추격)
                logger.info(f"⏳ 미체결로 인한 주문 취소 및 갱신 (Follow-up)...")
                self.exchange.cancel_order(order['id'], symbol)
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"공격적 매수 중 오류: {e}")
                time.sleep(1)
        
        # [변경] 매수는 3회 실패 시 포기 (시장가 강제 집행 안함)
        logger.error(f"❌ 공격적 매수 최종 실패 (3회 시도 후 포기): {symbol}")
        self.cancel_all_orders(symbol)
        return {}

    def sell(self, symbol: str, quantity: float, price: Optional[float] = None, is_stop_loss: bool = False) -> Dict:
        """매도 주문 (재시도 로직 포함)"""
        # [요청사항] 급격한 손절은 시장가로 처리
        if is_stop_loss:
            logger.warning(f"🚨 [STOP_LOSS] 급격한 손절 상황! 시장가 매도 실행: {symbol}")
            return self._sell_market_safe(symbol, quantity)

        # [요청사항] price가 없으면 '공격적 지정가' 주문 로직 수행
        if price is None:
            return self._sell_aggressive(symbol, quantity)

        # [요청사항 1] 호가 단위 보정
        if price:
            price = self.adjust_price_unit(price)
            
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                # [요청사항 2] 매도 전 미체결 주문 취소 -> 대기 -> 잔액 재조회 -> 수량 보정
                # 재시도 시에도 자산 잠김을 풀기 위해 매번 실행
                try:
                    open_orders = self.get_open_orders(symbol)
                    if open_orders:
                        logger.info(f"🚨 {symbol} 매도 진입 전 미체결 주문 {len(open_orders)}건 강제 취소")
                        for o in open_orders:
                            self.cancel_order(o['id'], symbol)
                        time.sleep(0.5)
                    
                    # 잔액 재조회 (미체결 취소 후 실제 가용 잔액 확인)
                    balance = self.exchange.fetch_balance()
                    currency = symbol.split('/')[0]
                    available = float(balance.get(currency, {}).get('free', 0))
                    
                    # 요청 수량보다 가용 수량이 적으면 가용 수량으로 조정
                    if available < quantity:
                        if attempt == 0:
                            logger.warning(f"⚠️ 매도 수량 조정: 요청 {quantity} -> 가용 {available}")
                        quantity = available

                except Exception as e:
                    logger.warning(f"매도 전처리(취소/잔액조회) 중 오류: {e}")

                # [요청사항 2] 수량 정밀도 조정
                quantity = float(self.exchange.amount_to_precision(symbol, quantity))

                if quantity <= 0:
                    logger.error(f"❌ 매도 가능 수량 없음 (0): {symbol}")
                    return {}

                if price:
                    order = self.exchange.create_limit_sell_order(symbol, quantity, price)
                else:
                    order = self.exchange.create_market_sell_order(symbol, quantity)
                
                # [로그 상세화] 주문 ID 및 타임스탬프 기록
                order_id = order.get('id', 'unknown')
                order_ts = order.get('timestamp', int(time.time()*1000))
                logger.info(f"매도 주문 성공: {symbol} {quantity} (ID: {order_id}, Time: {order_ts})")
                return order

            except (ccxt.NetworkError, ccxt.RateLimitExceeded, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable) as e:
                if attempt < max_retries:
                    error_type = type(e).__name__
                    logger.warning(f"🚀 [RETRY] {symbol} 매도 주문 재시도 중... (사유: {error_type}) ({attempt+1}/{max_retries})")
                    time.sleep(0.5)
                    # 매도는 다음 루프의 전처리(미체결 취소)가 자산 잠김을 해결함
                    continue
                else:
                    logger.error(f"매도 주문 최종 실패: {e} | Symbol: {symbol}, Price: {price}, Qty: {quantity}")
                    return {}
            except Exception as e:
                # [요청사항 4] 상세 로깅
                logger.error(f"매도 주문 오류: {e} | Symbol: {symbol}, Price: {price}, Qty: {quantity}")
                return {}
        return {}
    
    def _sell_aggressive(self, symbol: str, quantity: float) -> Dict:
        """공격적 지정가 매도 (추격형)"""
        slippage_ticks = TRADING_CONFIG["crypto"].get("slippage_ticks", 2)
        wait_sec = TRADING_CONFIG["crypto"].get("order_wait_seconds", 5)
        
        # 매도 전처리 (미체결 취소 등)
        self.cancel_all_orders(symbol)
        
        for attempt in range(3):
            try:
                # 잔액 및 수량 재확인
                balance = self.exchange.fetch_balance()
                currency = symbol.split('/')[0]
                available = float(balance.get(currency, {}).get('free', 0))
                if available < quantity:
                    quantity = available
                if quantity <= 0: return {}

                ticker = self.get_ticker(symbol)
                bid_price = float(ticker['bid'])
                tick_size = self.get_tick_size(bid_price)
                
                # 현재가 - N틱 (공격적)
                target_price = bid_price - (tick_size * slippage_ticks)
                target_price = self.adjust_price_unit(target_price)
                
                logger.info(f"📉 [SLIPPAGE_PROTECTION] 매도: 현재가({bid_price:,.0f}) 대비 -{slippage_ticks}틱({target_price:,.0f})으로 지정가 제출 ({attempt+1}/3)")
                
                order = self.exchange.create_limit_sell_order(symbol, quantity, target_price)
                
                time.sleep(wait_sec)
                
                order_info = self.exchange.fetch_order(order['id'], symbol)
                if order_info['status'] == 'closed':
                    logger.info(f"✅ 공격적 매도 체결 완료: {symbol}")
                    return order_info
                
                logger.info(f"⏳ 미체결로 인한 주문 취소 및 갱신 (Follow-up)...")
                self.exchange.cancel_order(order['id'], symbol)
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"공격적 매도 중 오류: {e}")
                time.sleep(1)
        
        # [Last Resort] 3회 시도 실패 시 시장가 강제 집행
        logger.warning(f"📉 [LAST_RESORT] 지정가 체결 실패로 인해 시장가 강제 집행 (종목: {symbol})")
        # _sell_market_safe 내부에서 cancel_all_orders 수행함
        
        try:
            # 기준가 조회 (슬리피지 계산용)
            ticker = self.get_ticker(symbol)
            ref_price = float(ticker['bid']) if ticker.get('bid') else 0.0
            
            order = self._sell_market_safe(symbol, quantity)
            if order:
                self._log_execution_details(symbol, order, ref_price, "SELL")
            return order
        except Exception as e:
            logger.error(f"❌ 시장가 강제 매도 실패: {e}")
            return {}

    def _sell_market_safe(self, symbol: str, quantity: float) -> Dict:
        """안전한 시장가 매도 (미체결 취소 포함)"""
        try:
            self.cancel_all_orders(symbol)
            time.sleep(0.2)
            
            # 잔액 재조회
            balance = self.exchange.fetch_balance()
            currency = symbol.split('/')[0]
            available = float(balance.get(currency, {}).get('free', 0))
            if available < quantity:
                quantity = available
            
            if quantity > 0:
                return self.exchange.create_market_sell_order(symbol, quantity)
        except Exception as e:
            logger.error(f"시장가 매도 오류: {e}")
        return {}

    def _log_execution_details(self, symbol: str, order: Dict, ref_price: float, side: str):
        """체결 세부 정보 및 슬리피지 로깅"""
        try:
            # 체결가 확인 (average가 없으면 fetch_order 시도)
            avg_price = order.get('average')
            if avg_price is None:
                time.sleep(0.2) # 체결 대기
                updated_order = self.exchange.fetch_order(order['id'], symbol)
                avg_price = updated_order.get('average')
            
            if avg_price:
                avg_price = float(avg_price)
                # 슬리피지 계산: (체결가 - 기준가) / 기준가 * 100
                if ref_price > 0:
                    diff_pct = ((avg_price - ref_price) / ref_price) * 100
                    
                    logger.warning(f"📊 [EXECUTION] {side} 시장가 체결 완료")
                    logger.warning(f"   - 종목: {symbol}")
                    logger.warning(f"   - 기준가: {ref_price:,.0f}원 -> 체결가: {avg_price:,.0f}원")
                    logger.warning(f"   - 차이(슬리피지): {diff_pct:+.2f}%")
            else:
                logger.warning(f"📊 [EXECUTION] {side} 시장가 체결 완료 (체결가 확인 불가)")
        except Exception as e:
            logger.error(f"체결 정보 로깅 중 오류: {e}")

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """주문 취소"""
        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info(f"주문 취소 성공: {order_id}")
            return True
        except Exception as e:
            logger.error(f"주문 취소 오류: {e}")
            return False
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """미체결 주문 조회"""
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            return orders
        except Exception as e:
            logger.error(f"미체결 주문 조회 오류: {e}")
            return []
            
    def cancel_all_orders(self, symbol: Optional[str] = None, side: Optional[str] = None) -> int:
        """특정 종목(또는 전체)의 미체결 주문 일괄 취소 (side: 'buy' or 'sell')"""
        try:
            orders = self.get_open_orders(symbol)
            if not orders:
                return 0
            
            count = 0
            for order in orders:
                # side 필터
                if side and order['side'] != side:
                    continue
                
                if self.cancel_order(order['id'], order['symbol']):
                    count += 1
                time.sleep(0.05) # API 호출 제한 고려
            
            if count > 0:
                target = symbol if symbol else "전체"
                type_str = side if side else "모든"
                logger.info(f"🛡️ {target} 미체결 {type_str} 주문 {count}건 취소 완료")
                # [요청사항 1] 취소 후 잔액 업데이트 대기 시간 삽입
                time.sleep(0.5)
            return count
        except Exception as e:
            logger.error(f"일괄 취소 중 오류: {e}")
            return 0


class BinanceAPI(BaseAPI):
    """바이낸스 API 구현"""
    
    def __init__(self, api_key: str, api_secret: str):
        super().__init__(api_key, api_secret)
        self.exchange = None
        # [New] WebSocket 관련
        self.ws_app = None
        self.wst = None
        self.price_cache = {}
        self.is_ws_ready = False
        self.ws_symbols = []
        self.symbol_map = {}
        self.lock = threading.Lock()
        self.callbacks = []
        self.use_websocket = False
        self.is_future = False
        self.last_ws_update = 0
        self.error_callbacks = []
        self.leverage_cache = {} # [New] 레버리지 캐시
    
    def connect(self):
        """바이낸스 API 연결"""
        try:
            # [요청사항 1] 현물/선물 분기 (엄격한 적용)
            self.is_future = TRADING_CONFIG["binance"].get("futures_enabled", False)
            default_type = 'future' if self.is_future else 'spot'
            
            # [New] API 키 로깅 (보안을 위해 마스킹 처리)
            masked_key = self.api_key[:4] + "*" * 10 + self.api_key[-4:] if self.api_key and len(self.api_key) > 8 else "INVALID"
            logger.info(f"🔑 바이낸스 API 키 로드: {masked_key}")

            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': default_type, # .env 설정에 따라 강제
                    'adjustForTimeDifference': True, # [Request] 시간 동기화 자동 보정
                    'recvWindow': 10000, # [Request] 네트워크 지연 허용 (10초)
                },
                'timeout': 10000, # [Request] 타임아웃 10초 설정
            })
            
            # [New] 연결 및 권한 검증 (시장 데이터 로드 + 잔액 조회로 키 유효성 테스트)
            self.exchange.load_markets() 
            self.exchange.fetch_balance() # -2015 에러 등 권한 문제 즉시 확인용

            mode_str = "선물" if self.is_future else "현물"
            logger.info(f"✅ 바이낸스 API ({mode_str}) 연결 및 검증 성공")
        except Exception as e:
            logger.error(f"바이낸스 API 연결 오류: {e}")
            self.exchange = None
            raise e # 메인에서 예외를 처리할 수 있도록 전파
    
    def disconnect(self):
        """바이낸스 API 연결 종료"""
        self.use_websocket = False
        if self.ws_app:
            self.ws_app.close()
        self.exchange = None
        logger.info("바이낸스 API 연결 종료")

    def set_leverage(self, symbol: str, leverage: int):
        """[New] 레버리지 설정"""
        try:
            # 캐시 확인 (불필요한 API 호출 방지)
            prev_lev = self.leverage_cache.get(symbol)
            if prev_lev == leverage:
                return

            # ccxt unified method
            self.exchange.set_leverage(leverage, symbol)
            self.leverage_cache[symbol] = leverage
            
            if prev_lev:
                direction = "상향" if leverage > prev_lev else "하향"
                logger.info(f"⚖️ [DYNAMIC LEVERAGE] 변동성 감지: {symbol} 레버리지를 {prev_lev}배에서 {leverage}배로 {direction} 조정합니다.")
            else:
                logger.info(f"⚙️ [BINANCE] {symbol} 레버리지 초기 설정: {leverage}x")
        except Exception as e:
            logger.warning(f"{symbol} 레버리지 설정 실패: {e}")

    def set_position_mode(self, hedge_mode: bool = False):
        """[New] 포지션 모드 설정 (Hedge Mode vs One-way Mode)"""
        try:
            # binance specific
            self.exchange.set_position_mode(hedge_mode)
            mode = "Hedge" if hedge_mode else "One-way"
            logger.info(f"⚙️ [BINANCE] 포지션 모드 설정: {mode}")
        except Exception as e:
            logger.debug(f"포지션 모드 설정 실패/스킵: {e}")

    def get_liquidation_risk(self, symbol: str) -> Dict:
        """[New] 청산 위험도 조회 (청산가 거리 모니터링)"""
        try:
            positions = self.exchange.fetch_positions([symbol])
            for p in positions:
                if p['symbol'] == symbol:
                    liq_price = float(p.get('liquidationPrice') or 0)
                    mark_price = float(p.get('markPrice') or 0)
                    if liq_price > 0 and mark_price > 0:
                        # 청산가와의 거리 비율 (Distance to Liquidation)
                        distance_pct = abs(mark_price - liq_price) / mark_price
                        return {'distance_pct': distance_pct, 'liquidation_price': liq_price}
        except Exception as e:
            logger.error(f"{symbol} 청산 리스크 조회 오류: {e}")
        return {}

    def _ensure_market_settings(self, symbol: str):
        """[요청사항 1, 2] 격리 마진 및 레버리지 설정 (하드캡 적용)"""
        # [요청사항 4] 선물 전용 로직 보호 (현물 모드 시 실행 차단)
        if self.exchange.options.get('defaultType') != 'future':
            return

        try:
            # 1. 격리 마진 강제 (ISOLATED)
            try:
                self.exchange.set_margin_mode('ISOLATED', symbol)
            except Exception:
                pass # 이미 설정된 경우 무시

            # 2. 레버리지 설정 및 하드캡 적용
            config_lev = TRADING_CONFIG["binance"].get("leverage", 1)
            target_lev = config_lev
            
            # [요청사항 2] 5배 초과 시 3배로 강제 하향 조정
            if config_lev > 5:
                logger.warning(f"⚠️ [SAFETY] 설정된 레버리지({config_lev}x)가 5배를 초과하여 3배로 강제 하향 조정합니다.")
                target_lev = 3
            
            self.exchange.set_leverage(target_lev, symbol)
        except Exception as e:
            logger.warning(f"{symbol} 마진/레버리지 설정 중 오류: {e}")
    
    def get_balance(self, currency: str = "USDT") -> Dict:
        """계좌 잔액 조회 (재시도 로직 포함)"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                balance = self.exchange.fetch_balance()
                return {
                    "total": balance.get("total", {}),
                    "free": balance.get("free", {}),
                    "used": balance.get("used", {}),
                }
            except Exception as e:
                # 마지막 시도였다면 에러 로그 후 종료
                if attempt == max_retries - 1:
                    logger.error(f"❌ [BINANCE] 잔액 조회 최종 실패: {e}")
                    return {}
                
                # 지수 백오프: 0.2s -> 0.4s -> 0.8s
                wait_time = 0.2 * (2 ** attempt)
                logger.warning(f"⚠️ [BINANCE] 잔액 조회 실패, 재시도 중... ({attempt + 1}/{max_retries}) | 대기: {wait_time:.1f}s | 오류: {e}")
                time.sleep(wait_time)
        return {}
    
    def get_price(self, symbol: str) -> float:
        """현재가 조회"""
        # 1. 웹소켓 캐시 확인
        if self.is_ws_ready:
            with self.lock:
                if symbol in self.price_cache:
                    return self.price_cache[symbol]
        
        # 2. REST API Fallback
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"{symbol} 현재가 조회 오류: {e}")
            return 0.0
    
    def get_ticker(self, symbol: str) -> Dict:
        """티커 정보 조회 (호가 포함)"""
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"{symbol} 티커 조회 오류: {e}")
            return {}
            
    def get_tick_size(self, symbol: str) -> float:
        """심볼별 호가 단위(Tick Size) 조회"""
        try:
            market = self.exchange.market(symbol)
            # ccxt precisionMode에 따라 다를 수 있으나 binance는 보통 decimal places
            if 'precision' in market and 'price' in market['precision']:
                precision = market['precision']['price']
                return 1 / (10 ** precision)
            return 0.00000001
        except:
            return 0.00000001

    def adjust_price_unit(self, symbol: str, price: float) -> float:
        """바이낸스 호가 단위 보정 (price_to_precision 사용)"""
        try:
            # ccxt가 문자열로 반환하므로 float 변환
            return float(self.exchange.price_to_precision(symbol, price))
        except:
            return price

    def get_ohlcv(self, symbol: str, timeframe: str = "1d", limit: int = 200) -> pd.DataFrame:
        """OHLCV 데이터 조회"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"{symbol} OHLCV 데이터 조회 오류: {e}")
            return pd.DataFrame()
    
    def get_positions(self) -> List[Dict]:
        """보유 포지션 조회 (평단가 포함)"""
        try:
            # [요청사항 1] 현물/선물 모드에 따라 조회 방식 분기
            is_future = self.exchange.options.get('defaultType') == 'future'
            
            if is_future:
                # [선물] fetch_positions 사용
                raw_positions = self.exchange.fetch_positions()
                positions = []
                for p in raw_positions:
                    qty = float(p['contracts'])
                    if qty > 0:
                        positions.append({
                            'symbol': p['symbol'],
                            'quantity': qty,
                            'entry_price': float(p['entryPrice'])
                        })
                return positions
            else:
                # [현물] fetch_balance 사용 (Spot Balance)
                balance = self.exchange.fetch_balance()
                positions = []
                if 'total' in balance:
                    for currency, qty in balance['total'].items():
                        if currency == 'USDT': continue
                        if qty > 0:
                            symbol = f"{currency}/USDT"
                            positions.append({
                                'symbol': symbol,
                                'quantity': float(qty),
                                'entry_price': 0.0 # 현물은 평단가 API 미제공
                            })
            return positions
        except Exception as e:
            logger.error(f"포지션 조회 오류: {e}")
            return []

    def buy(self, symbol: str, quantity: float, price: Optional[float] = None, leverage: int = 1, **kwargs) -> Dict:
        """매수 주문 (재시도 및 공격적 지정가 포함)"""
        # [요청사항 4] 주문 전 클린 슬레이트 (미체결 취소 -> 대기 -> 설정 확인)
        self.cancel_all_orders(symbol)
        time.sleep(0.5)
        self._ensure_market_settings(symbol)

        # [New] 동적 레버리지 적용
        if self.is_future and leverage > 1:
            self.set_leverage(symbol, leverage)

        if price is None:
            return self._buy_aggressive(symbol, quantity)

        if price:
            price = self.adjust_price_unit(symbol, price)
        
        # 수량 정밀도 보정
        quantity = float(self.exchange.amount_to_precision(symbol, quantity))

        max_retries = 2
        for attempt in range(max_retries + 1):
            # 일반 지정가/시장가 주문 (price가 명시된 경우)
            try:
                if price:
                    order = self.exchange.create_limit_buy_order(symbol, quantity, price)
                else:
                    order = self.exchange.create_market_buy_order(symbol, quantity)
                
                order_id = order.get('id', 'unknown')
                logger.info(f"매수 주문 성공: {symbol} {quantity} (ID: {order_id})")
                self._place_stop_loss_order(symbol, quantity, price or 0) # 시장가일 경우 가격 확인 필요하나 일단 호출
                return order
            except Exception as e:
                # [요청사항 2] 지수 백오프 및 에러 구분
                if attempt < max_retries:
                    wait_time = 0.2 * (2 ** attempt)
                    err_type = "네트워크 오류" if isinstance(e, ccxt.NetworkError) else "API/권한 오류"
                    logger.warning(f"🚀 [RETRY] {symbol} 매수 재시도 ({attempt+1}/{max_retries}) 사유: {err_type} ({e}) | 대기: {wait_time:.1f}s")
                    time.sleep(wait_time)
                    continue
                logger.error(f"매수 주문 오류: {e}")
                return {}
        return {}

    def _buy_aggressive(self, symbol: str, quantity: float) -> Dict:
        """공격적 지정가 매수 (바이낸스용)"""
        slippage_ticks = TRADING_CONFIG["binance"].get("slippage_ticks", 2)
        wait_sec = TRADING_CONFIG["binance"].get("order_wait_seconds", 5)
        
        # 수량 정밀도 보정
        quantity = float(self.exchange.amount_to_precision(symbol, quantity))

        for attempt in range(3):
            try:
                ticker = self.get_ticker(symbol)
                ask_price = float(ticker['ask'])
                tick_size = self.get_tick_size(symbol)
                
                target_price = ask_price + (tick_size * slippage_ticks)
                target_price = self.adjust_price_unit(symbol, target_price)
                
                logger.info(f"📉 [BINANCE] 공격적 매수: {ask_price} -> {target_price} ({attempt+1}/3)")
                order = self.exchange.create_limit_buy_order(symbol, quantity, target_price)
                
                time.sleep(wait_sec)
                order_info = self.exchange.fetch_order(order['id'], symbol)
                if order_info['status'] == 'closed':
                    logger.info(f"✅ [BINANCE] 공격적 매수 체결 완료: {symbol}")
                    self._place_stop_loss_order(symbol, quantity, target_price)
                    return order_info
                
                self.exchange.cancel_order(order['id'], symbol)
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"공격적 매수 오류: {e}")
                time.sleep(1)
        
        # [요청사항 3] 매수: 3회 시도 후 포기 (추격 매수 금지)
        logger.error(f"❌ [BINANCE] 공격적 매수 최종 실패 (포기): {symbol}")
        self.cancel_all_orders(symbol)
        return {}

    def create_oco_order(self, symbol: str, quantity: float, buy_price: float, take_profit_pct: float, stop_loss_pct: float) -> Dict:
        """OCO 주문 생성 (현물 전용: 익절/손절 동시 설정)"""
        # 현물 모드인지 확인
        if self.exchange.options.get('defaultType') != 'spot':
            return {}

        try:
            # 1. 정밀도 보정 (바이낸스 규격 준수)
            qty = float(self.exchange.amount_to_precision(symbol, quantity))
            
            # 2. 가격 계산
            # 익절가 (Limit Maker)
            tp_price = buy_price * (1 + take_profit_pct)
            
            # 손절 트리거 (Stop Price)
            sl_trigger = buy_price * (1 - stop_loss_pct)
            
            # [New] 최소 간격 보정 (1% Rule) - 거절 방지
            min_gap = buy_price * 0.01
            current_gap = tp_price - sl_trigger
            
            if current_gap < min_gap:
                logger.warning(f"⚠️ [OCO] 익절/손절 간격 부족({current_gap:.2f} < {min_gap:.2f}). 1% 간격으로 자동 보정합니다.")
                mid_price = (tp_price + sl_trigger) / 2
                tp_price = mid_price + (min_gap / 2)
                sl_trigger = mid_price - (min_gap / 2)

            # 정밀도 보정 (BinanceAPI.adjust_price_unit 사용)
            tp_price = self.adjust_price_unit(symbol, tp_price)
            sl_trigger = self.adjust_price_unit(symbol, sl_trigger)
            
            # 손절 리밋 (Stop Limit Price) - 트리거보다 0.5% 낮게 설정하여 급락 시 체결 확률 확보
            sl_limit = sl_trigger * 0.995
            sl_limit = self.adjust_price_unit(symbol, sl_limit)

            # 3. OCO 주문 전송
            logger.info(f"🛡️ [OCO] 주문 시도: {symbol} {qty}개 | 익절: {tp_price} | 손절: {sl_trigger}(Limit {sl_limit})")
            
            order = self.exchange.create_order(
                symbol,
                'oco',
                'sell',
                qty,
                tp_price,
                params={
                    'stopPrice': sl_trigger,
                    'stopLimitPrice': sl_limit,
                    'stopLimitTimeInForce': 'GTC' # 취소 전까지 유효
                }
            )
            logger.info(f"✅ [OCO] 주문 등록 성공: {symbol}")
            return order
        except Exception as e:
            logger.error(f"❌ [OCO] 주문 실패: {e}")
            return {}

    def _place_stop_loss_order(self, symbol: str, quantity: float, entry_price: float):
        """[요청사항 2] 진입 직후 STOP_MARKET 주문 등록"""
        # [요청사항 4] 선물 전용 로직 보호 (현물 모드 시 실행 차단)
        if self.exchange.options.get('defaultType') != 'future':
            return

        try:
            # 시장가 체결 등으로 entry_price가 0이면 현재가 조회
            if entry_price <= 0:
                entry_price = self.get_price(symbol)
            
            # -3% ~ -5% (기본 3%)
            stop_loss_pct = 0.03 
            stop_price = entry_price * (1 - stop_loss_pct)
            stop_price = self.adjust_price_unit(symbol, stop_price)
            
            params = {'stopPrice': stop_price, 'reduceOnly': True}
            self.exchange.create_order(symbol, 'STOP_MARKET', 'sell', quantity, params=params)
            logger.info(f"🛡️ [SAFETY] STOP_MARKET 등록 완료: {symbol} @ {stop_price} (-{stop_loss_pct*100}%)")
        except Exception as e:
            logger.error(f"STOP_MARKET 주문 등록 실패: {e}")

    def sell(self, symbol: str, quantity: float, price: Optional[float] = None, is_stop_loss: bool = False) -> Dict:
        """매도 주문 (Last Resort 포함)"""
        # [요청사항 4] 주문 전 클린 슬레이트
        self.cancel_all_orders(symbol)
        time.sleep(0.5)
        # 매도는 청산이므로 레버리지 설정 불필요하나 안전을 위해 체크 가능

        if is_stop_loss:
            return self._sell_market_safe(symbol, quantity)
        
        if price is None:
            return self._sell_aggressive(symbol, quantity)

        if price:
            price = self.adjust_price_unit(symbol, price)
        
        quantity = float(self.exchange.amount_to_precision(symbol, quantity))

        try:
            if price:
                order = self.exchange.create_limit_sell_order(symbol, quantity, price)
            else:
                order = self.exchange.create_market_sell_order(symbol, quantity)
            logger.info(f"매도 주문 성공: {symbol} {quantity}")
            return order
        except Exception as e:
            logger.error(f"매도 주문 오류: {e}")
            return {}

    def _sell_aggressive(self, symbol: str, quantity: float) -> Dict:
        """공격적 지정가 매도 (바이낸스용)"""
        slippage_ticks = TRADING_CONFIG["binance"].get("slippage_ticks", 2)
        wait_sec = TRADING_CONFIG["binance"].get("order_wait_seconds", 5)
        
        quantity = float(self.exchange.amount_to_precision(symbol, quantity))

        for attempt in range(3):
            try:
                # 잔액 재확인
                balance = self.exchange.fetch_balance()
                currency = symbol.split('/')[0]
                available = float(balance.get(currency, {}).get('free', 0))
                if available < quantity: quantity = available
                if quantity <= 0: return {}

                ticker = self.get_ticker(symbol)
                bid_price = float(ticker['bid'])
                tick_size = self.get_tick_size(symbol)
                
                target_price = bid_price - (tick_size * slippage_ticks)
                target_price = self.adjust_price_unit(symbol, target_price)
                
                logger.info(f"📉 [BINANCE] 공격적 매도: {bid_price} -> {target_price} ({attempt+1}/3)")
                order = self.exchange.create_limit_sell_order(symbol, quantity, target_price)
                
                time.sleep(wait_sec)
                order_info = self.exchange.fetch_order(order['id'], symbol)
                if order_info['status'] == 'closed':
                    logger.info(f"✅ [BINANCE] 공격적 매도 체결 완료: {symbol}")
                    return order_info
                
                self.exchange.cancel_order(order['id'], symbol)
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"공격적 매도 오류: {e}")
                time.sleep(1)
        
        # [요청사항 3] 매도: 3회 시도 후 시장가 강제 청산 (Last Resort)
        logger.warning(f"📉 [LAST_RESORT] 바이낸스 시장가 강제 매도: {symbol}")
        return self._sell_market_safe(symbol, quantity)

    def _sell_market_safe(self, symbol: str, quantity: float) -> Dict:
        """안전한 시장가 매도"""
        try:
            self.cancel_all_orders(symbol)
            time.sleep(0.2)
            quantity = float(self.exchange.amount_to_precision(symbol, quantity))
            if quantity > 0:
                return self.exchange.create_market_sell_order(symbol, quantity)
                order = self.exchange.create_market_sell_order(symbol, quantity)
                logger.info(f"✅ [BINANCE] 시장가 매도 주문 완료: {symbol} {quantity}")
                return order
        except Exception as e:
            logger.error(f"시장가 매도 오류: {e}")
        return {}
    
    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """주문 취소"""
        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info(f"주문 취소 성공: {order_id}")
            return True
        except Exception as e:
            logger.error(f"주문 취소 오류: {e}")
            return False
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """미체결 주문 조회"""
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            return orders
        except Exception as e:
            logger.error(f"미체결 주문 조회 오류: {e}")
            return []

    def cancel_all_orders(self, symbol: Optional[str] = None, side: Optional[str] = None) -> int:
        """미체결 주문 일괄 취소"""
        try:
            orders = self.get_open_orders(symbol)
            count = 0
            for order in orders:
                if side and order['side'] != side: continue
                try:
                    self.exchange.cancel_order(order['id'], order['symbol'])
                    count += 1
                except: pass
            if count > 0:
                logger.info(f"🛡️ [BINANCE] {symbol or '전체'} 미체결 주문 {count}건 취소")
                time.sleep(0.5)
            return count
        except Exception as e:
            logger.error(f"일괄 취소 오류: {e}")
            return 0

    def add_price_callback(self, callback):
        self.callbacks.append(callback)

    def subscribe_websocket(self, symbols: List[str]):
        """웹소켓 구독 시작 (자동 재연결 및 비동기 수집)"""
        if not websocket:
            logger.warning("⚠️ websocket-client 미설치로 바이낸스 실시간 시세 불가")
            return

        self.ws_symbols = [s.replace('/', '').lower() for s in symbols]
        self.symbol_map = {s.replace('/', '').lower(): s for s in symbols}
        self.use_websocket = True
        
        # 기존 연결 종료
        if self.ws_app:
            self.ws_app.close()
            
        # 별도 스레드에서 실행 (비동기 병렬 처리)
        self.wst = threading.Thread(target=self._ws_run_loop)
        self.wst.daemon = True
        self.wst.start()

    def reconnect_websocket(self):
        """웹소켓 강제 재연결"""
        if not self.use_websocket:
            return
        
        logger.info("🔄 [BINANCE] WebSocket 재연결 실행...")
        # 현재 구독 중인 심볼 목록 복원
        current_symbols = list(self.symbol_map.values())
        if current_symbols:
            self.subscribe_websocket(current_symbols)

    def add_error_callback(self, callback):
        """에러 발생 시 호출할 콜백 등록"""
        self.error_callbacks.append(callback)

    def _notify_error(self, message):
        """등록된 에러 콜백 호출"""
        for cb in self.error_callbacks:
            try: cb(message)
            except: pass

    def check_server_time(self) -> bool:
        """서버 시간과 로컬 시간 차이 확인 (5초 이상 시 경고)"""
        try:
            server_time = self.exchange.fetch_time()
            local_time = int(time.time() * 1000)
            diff = server_time - local_time
            
            if abs(diff) > 5000:
                logger.warning(f"⚠️ [BINANCE] 서버/로컬 시간 차이 과다: {diff}ms (허용: 5000ms)")
                return False
            return True
        except Exception as e:
            logger.error(f"시간 동기화 체크 오류: {e}")
            return False

    def health_check(self):
        """API 연결 상태 점검 및 자동 재연결"""
        try:
            # 1. REST API 연결 확인 (시간 조회로 대체)
            if not self.check_server_time():
                logger.warning("⚠️ [BINANCE] 연결 불안정 또는 시간 오차 감지. REST API 재연결 시도...")
                self.connect()
        except Exception as e:
            logger.error(f"❌ [BINANCE] 헬스 체크 실패: {e} -> 재연결 시도")
            try:
                self.connect()
            except Exception as re_e:
                logger.error(f"재연결 실패: {re_e}")

    def _ws_run_loop(self):
        """웹소켓 실행 루프 (Auto-Reconnect)"""
        while self.use_websocket:
            try:
                # 스트림 URL 생성
                streams = "/".join([f"{s}@ticker" for s in self.ws_symbols])
                base = "wss://fstream.binance.com" if self.is_future else "wss://stream.binance.com:9443"
                url = f"{base}/stream?streams={streams}"
                
                logger.info(f"📡 [BINANCE] WebSocket 연결 시도 ({len(self.ws_symbols)}종목)...")
                
                self.ws_app = websocket.WebSocketApp(
                    url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                
                # 블로킹 호출 (연결 유지)
                self.ws_app.run_forever(ping_interval=60, ping_timeout=10)
                
            except Exception as e:
                logger.error(f"❌ [BINANCE] WebSocket 오류: {e}")
                self._notify_error(f"WebSocket 런타임 오류: {e}")
                time.sleep(5)
            
            if self.use_websocket:
                logger.warning("⚠️ [BINANCE] WebSocket 연결 끊김. 5초 후 재연결...")
                self._notify_error("WebSocket 연결 끊김. 5초 후 재연결 시도...")
                self.is_ws_ready = False # [요청사항 1] 초기값 대기 상태로 전환
                time.sleep(5)

    def _on_open(self, ws):
        logger.info("✅ [BINANCE] WebSocket 연결 수립")

    def _on_message(self, ws, message):
        self.last_ws_update = time.time()
        try:
            data = json.loads(message)
            if 'data' in data:
                ticker = data['data']
                symbol_raw = ticker['s'].lower()
                price = float(ticker['c'])
                
                std_symbol = self.symbol_map.get(symbol_raw)
                if std_symbol:
                    with self.lock:
                        self.price_cache[std_symbol] = price
                    
                    # [요청사항 1] 첫 데이터 수신 시 Ready
                    if not self.is_ws_ready:
                        self.is_ws_ready = True
                        logger.info("✅ [BINANCE] 실시간 시세 수신 시작 (Ready)")
                        
                    for cb in self.callbacks:
                        try: cb(std_symbol, price)
                        except: pass
        except Exception as e:
            logger.error(f"WS Message Error: {e}")

    def _on_error(self, ws, error):
        logger.error(f"❌ [BINANCE] WS Error: {error}")
        self._notify_error(f"WS 프로토콜 오류: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info("🔒 [BINANCE] WebSocket 연결 종료")
