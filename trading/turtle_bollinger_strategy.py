import logging
import time
import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange, BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator
from typing import Optional
from trading.strategy import TradingStrategy, Signal
from config.settings import TRADING_CONFIG

logger = logging.getLogger(__name__)

class TurtleBollingerStrategy(TradingStrategy):
    """
    터틀 트레이딩 + 볼린저 밴드 결합 전략
    
    진입 조건:
    1. 10일 최고가 돌파 (터틀 단기)
    2. 볼린저 밴드 하단 터치 후 반등
    3. RSI 과매도 (<30)
    
    장점: 하이킨 아시보다 훨씬 명확하고 진입 기회 많음
    """
    
    def __init__(self, lookback_window: int = 200):
        super().__init__(lookback_window)
        self.breakout_period = 7   # 터틀 단기 (10 -> 7캔들로 단축)
        self.exit_period = 30       # 터틀 청산 (30일 -> 30캔들)
        self.last_log_time = {}     # 로그 출력 제한용

    def generate_signal(self, symbol: str, data: pd.DataFrame, 
                       current_capital: float = 0.0, 
                       strategy_override: str = None) -> Optional[Signal]:
        """매수/매도 신호 생성"""
        
        # [Request 2] 안정적인 지표 계산을 위한 최소 데이터 확인 (EMA50, BB20 고려)
        if len(data) < 50:
            logger.debug(f"[{symbol}] 데이터 부족")
            return Signal(symbol=symbol, action="HOLD", confidence=0.0, reason="데이터 부족")

        try:
            # 현재 가격
            current_price = data['close'].iloc[-1]
            
            # 1. 터틀 트레이딩 돌파 계산
            high_breakout = data['high'].rolling(self.breakout_period).max().iloc[-2]  # 전일까지의 최고가
            low_breakout = data['low'].rolling(self.breakout_period).min().iloc[-2]
            
            # 2. 볼린저 밴드
            bb_indicator = BollingerBands(close=data['close'], window=20, window_dev=2)
            bb_upper = bb_indicator.bollinger_hband().iloc[-1]
            bb_lower = bb_indicator.bollinger_lband().iloc[-1]
            bb_middle = bb_indicator.bollinger_mavg().iloc[-1]
            
            # 3. RSI
            rsi_indicator = RSIIndicator(close=data['close'], window=14)
            rsi = rsi_indicator.rsi().iloc[-1]
            
            # 4. ADX (추세 강도)
            adx_indicator = ADXIndicator(high=data['high'], low=data['low'], close=data['close'], window=14)
            adx = adx_indicator.adx().iloc[-1]
            
            # 5. ATR (변동성)
            atr_indicator = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=14)
            atr = atr_indicator.average_true_range().iloc[-1]
            
            # 6. 거래량 확인
            volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            
            # [Safety] 평균 거래량이 0인 경우 방어
            if avg_volume is None or avg_volume == 0:
                avg_volume = 1.0
                
            # [변경] 거래량 가중치 상향 (1.2 -> 1.5배) : 진짜 수급이 들어올 때만 진입
            volume_surge = volume > avg_volume * 1.5
            
            # 디버깅 로그
            breakout_pct = ((current_price/high_breakout-1)*100) if high_breakout > 0 else 0.0
            vol_ratio = (volume/avg_volume) if avg_volume > 0 else 0.0
            
            logger.debug(f"""
[{symbol}] 시장 상황:
  - 가격: {current_price:,.0f}
  - 돌파선: {high_breakout:,.0f} (현재가 {breakout_pct:+.1f}%)
  - BB: 하단 {bb_lower:,.0f} / 중간 {bb_middle:,.0f} / 상단 {bb_upper:,.0f}
  - RSI: {rsi:.1f}, ADX: {adx:.1f}
  - 거래량: {volume:,.0f} (평균 대비 {vol_ratio:.1f}x)
            """)
            
            # ========== 매수 조건 평가 ==========
            signal_action = "HOLD"
            reason = ""
            confidence = 0.0
            suggested_stop_loss = None
            
            # 조건 1: 터틀 돌파 (가장 강력)
            if current_price > high_breakout and volume_surge:
                signal_action = "BUY"
                confidence = 0.90
                reason = f"10일 고가({high_breakout:,.0f}) 돌파 + 거래량 급증"
                logger.info(f"🔥 [{symbol}] 조건1: 터틀 돌파!")
            
            # 조건 2: 볼린저 밴드 하단 반등
            elif current_price <= bb_lower * 1.02 and rsi < 45: # 범위 1%->2%, RSI 40->45 완화
                # 하단 터치 + RSI 과매도
                signal_action = "BUY"
                confidence = 0.75
                reason = f"볼린저 하단 터치 + RSI {rsi:.1f}"
                logger.info(f"💎 [{symbol}] 조건2: 볼린저 하단 반등")
            
            # 조건 3: RSI 과매도 (기준 완화: 25 -> 30)
            elif rsi < 30:
                signal_action = "BUY"
                confidence = 0.70
                reason = f"RSI 과매도({rsi:.1f})"
                logger.info(f"📉 [{symbol}] 조건3: RSI 과매도")
            
            # 조건 4: 중간선 돌파 + 추세 강화 (ADX 기준 완화: 20 -> 15)
            elif current_price > bb_middle and adx > 15 and volume_surge:
                prev_close = data['close'].iloc[-2]
                if prev_close <= bb_middle:  # 방금 돌파
                    signal_action = "BUY"
                    confidence = 0.65
                    reason = f"중간선 돌파 + ADX {adx:.1f} + 거래량"
                    logger.info(f"📈 [{symbol}] 조건4: 중간선 돌파")
            
            # 조건 5: 약한 신호 (테스트용 - 실전에서는 제거 고려)
            elif rsi < 45 and current_price < bb_middle and volume_surge:
                signal_action = "BUY"
                confidence = 0.50
                reason = f"약세장 저점 매수 (RSI {rsi:.1f})"
                logger.info(f"🎯 [{symbol}] 조건5: 저점 매수")
            
            # 로그 출력 (진입 실패 시)
            if signal_action == "HOLD":
                # [변경] 구체적인 미달 사유 로깅
                fail_reasons = []
                vol_ratio_log = (volume/avg_volume) if avg_volume > 0 else 0.0
                if not volume_surge: fail_reasons.append(f"거래량부족({vol_ratio_log:.1f}x)")
                if rsi >= 45: fail_reasons.append(f"RSI높음({rsi:.1f})")
                if adx <= 15: fail_reasons.append(f"추세약함(ADX {adx:.1f})")
                if current_price <= high_breakout: fail_reasons.append(f"돌파실패({current_price:,.0f}<{high_breakout:,.0f})")
                
                # 너무 자주 찍히지 않게 INFO 대신 DEBUG 사용하되, 내용은 구체적으로
                if volume_surge: # 거래량은 터졌는데 다른게 부족한 경우만 INFO로 격상
                    logger.info(f"[{symbol}] ❌ 진입 실패: {', '.join(fail_reasons)}")
                
                # [New] 1분마다 상태 강제 출력 (진입 장벽 확인용)
                current_time = time.time()
                if current_time - self.last_log_time.get(symbol, 0) > 60:
                    logger.info(f"[{symbol}] 💤 진입 대기: {', '.join(fail_reasons)}")
                    self.last_log_time[symbol] = current_time
            
            # 손절가 계산
            if signal_action == "BUY":
                # 방법 1: ATR 기반 (2.5배)
                atr_stop = current_price - (atr * 2.5)
                
                # 방법 2: 최근 저점 기반
                recent_low = data['low'].tail(10).min()
                low_stop = recent_low * 0.98
                
                # 방법 3: 볼린저 하단 기준
                bb_stop = bb_lower * 0.97
                
                # 가장 높은 손절가 선택 (타이트한 손절)
                suggested_stop_loss = max(atr_stop, low_stop, bb_stop)
                
                stop_pct = ((current_price - suggested_stop_loss) / current_price) * 100 if current_price > 0 else 0.0
                logger.info(f"  → 진입: {current_price:,.0f}, 손절: {suggested_stop_loss:,.0f} (-{stop_pct:.1f}%)")

            return Signal(
                symbol=symbol,
                action=signal_action,
                confidence=confidence,
                reason=reason,
                suggested_stop_loss=suggested_stop_loss,
                suggested_quantity=0.0  # [안전장치] 명시적 0.0 할당
            )

        except Exception as e:
            logger.error(f"[{symbol}] 전략 오류: {e}", exc_info=True)
            return Signal(symbol=symbol, action="HOLD", confidence=0.0, reason=f"오류: {e}")
