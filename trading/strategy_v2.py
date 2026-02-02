import logging
import pandas as pd
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator
from typing import Optional
from trading.strategy import TradingStrategy, Signal
from config.settings import TRADING_CONFIG

logger = logging.getLogger(__name__)

class HeikinAshiStrategy(TradingStrategy):
    """
    하이킨아시(Heikin-Ashi) 추세 추종 전략 (개선 버전)
    - 진입 조건 완화: 현실적인 시장 상황 반영
    - 추가 필터: RSI, ADX로 신뢰도 향상
    """
    
    def __init__(self, lookback_window: int = 60):
        super().__init__(lookback_window)

    def calculate_heikin_ashi(self, df: pd.DataFrame) -> pd.DataFrame:
        """하이킨아시 캔들 계산"""
        try:
            import pandas_ta as ta
            ha_df = ta.ha(df['open'], df['high'], df['low'], df['close'])
            return ha_df
        except ImportError:
            # pandas_ta가 없으면 수동 계산
            ha_df = pd.DataFrame(index=df.index)
            
            ha_df['HA_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            
            ha_open = [(df['open'].iloc[0] + df['close'].iloc[0]) / 2]
            ha_closes = ha_df['HA_close'].values
            for i in range(1, len(df)):
                ha_open.append((ha_open[i-1] + ha_closes[i-1]) / 2)
            ha_df['HA_open'] = ha_open
            
            ha_df['HA_high'] = ha_df[['HA_open', 'HA_close']].join(df['high']).max(axis=1)
            ha_df['HA_low'] = ha_df[['HA_open', 'HA_close']].join(df['low']).min(axis=1)
            
            return ha_df

    def generate_signal(self, symbol: str, data: pd.DataFrame, current_capital: float = 0.0, strategy_override: str = None) -> Optional[Signal]:
        """
        하이킨아시 기반 매수/매도 신호 생성 (완화된 조건)
        
        매수 조건 (3가지 중 하나만 충족하면 진입):
        1. 연속 2회 양봉 (아래 꼬리 5% 이내 허용)
        2. RSI 과매도 + 현재 양봉
        3. 강한 추세 (ADX 25+) + 현재 양봉
        """
        if len(data) < self.lookback_window:
            logger.debug(f"[{symbol}] 데이터 부족 ({len(data)} < {self.lookback_window})")
            return Signal(symbol=symbol, action="HOLD", confidence=0.0, reason="데이터 부족")

        try:
            # 1. 하이킨아시 캔들 계산
            ha_df = self.calculate_heikin_ashi(data)
            
            if ha_df is None or len(ha_df) < 3:
                return Signal(symbol=symbol, action="HOLD", confidence=0.0, reason="HA 계산 불가")

            # 2. RSI, ADX 계산 (추가 필터)
            rsi_indicator = RSIIndicator(close=data['close'], window=14)
            rsi = rsi_indicator.rsi().iloc[-1]
            
            adx_indicator = ADXIndicator(high=data['high'], low=data['low'], close=data['close'], window=14)
            adx = adx_indicator.adx().iloc[-1]

            # 3. 최근 캔들 분석
            current_ha = ha_df.iloc[-1]
            prev_ha = ha_df.iloc[-2]
            
            # 디버깅 로그
            logger.debug(f"[{symbol}] RSI: {rsi:.1f}, ADX: {adx:.1f}")
            
            # 양봉 판정 함수 (완화: 아래 꼬리 5% 이내 허용)
            def is_green_candle(candle):
                is_green = candle['HA_close'] > candle['HA_open']
                
                # 아래 꼬리 길이 계산
                body_size = abs(candle['HA_close'] - candle['HA_open'])
                lower_shadow = candle['HA_open'] - candle['HA_low']
                
                # 아래 꼬리가 몸통의 5% 이내 (완화된 조건)
                if body_size > 0:
                    shadow_ratio = lower_shadow / body_size
                    has_small_shadow = shadow_ratio <= 0.05
                else:
                    has_small_shadow = lower_shadow <= (candle['HA_open'] * 0.001)
                
                logger.debug(f"  - Green: {is_green}, Shadow: {lower_shadow:.2f}, Ratio: {shadow_ratio:.2%}")
                
                return is_green and has_small_shadow

            signal_action = "HOLD"
            reason = ""
            confidence = 0.0
            suggested_stop_loss = None
            
            # ========== 매수 조건 1: 연속 양봉 (기본) ==========
            if is_green_candle(prev_ha) and is_green_candle(current_ha):
                signal_action = "BUY"
                confidence = 0.80
                reason = "하이킨아시 연속 양봉 (추세 시작)"
                logger.info(f"🔔 [{symbol}] 조건1 충족: 연속 양봉")
            
            # ========== 매수 조건 2: RSI 과매도 + 양봉 (역추세) ==========
            elif rsi < 35 and is_green_candle(current_ha):
                signal_action = "BUY"
                confidence = 0.75
                reason = f"RSI 과매도({rsi:.1f}) + 양봉 반등"
                logger.info(f"🔔 [{symbol}] 조건2 충족: RSI 과매도 반등")
            
            # ========== 매수 조건 3: 강한 추세 + 양봉 (추세 추종) ==========
            elif adx > 25 and is_green_candle(current_ha):
                signal_action = "BUY"
                confidence = 0.85
                reason = f"강한 추세(ADX {adx:.1f}) + 양봉"
                logger.info(f"🔔 [{symbol}] 조건3 충족: 강한 추세")
            
            # ========== 추가 조건: 단순 양봉 (가장 완화) ==========
            elif is_green_candle(current_ha) and rsi < 60:
                signal_action = "BUY"
                confidence = 0.60
                reason = f"양봉 발생 (RSI {rsi:.1f})"
                logger.info(f"🔔 [{symbol}] 조건4 충족: 기본 양봉")
            
            else:
                logger.debug(f"[{symbol}] 진입 조건 미충족")
            
            # 손절가 계산 (진입 시그널 발생 시)
            if signal_action == "BUY":
                # ATR 기반 손절
                atr_window = TRADING_CONFIG["crypto"].get("atr_window", 20)
                atr_indicator = AverageTrueRange(data['high'], data['low'], data['close'], window=atr_window)
                atr = atr_indicator.average_true_range().iloc[-1]
                
                current_price = data['close'].iloc[-1]
                recent_low = data['low'].tail(5).min()  # 최근 5개 캔들 저점
                
                # 2가지 손절가 중 선택
                atr_stop = current_price - (atr * 2.5)
                recent_low_stop = recent_low * 0.98  # 최근 저점 -2%
                
                suggested_stop_loss = max(atr_stop, recent_low_stop)  # 더 높은 가격 (타이트한 손절)
                
                logger.info(f"  → 진입가: {current_price:,.0f}, 손절가: {suggested_stop_loss:,.0f} ({((current_price-suggested_stop_loss)/current_price*100):.1f}%)")

            return Signal(
                symbol=symbol,
                action=signal_action,
                confidence=confidence,
                reason=reason,
                suggested_stop_loss=suggested_stop_loss
            )

        except Exception as e:
            logger.error(f"HeikinAshi 전략 오류: {e}", exc_info=True)
            return Signal(symbol=symbol, action="HOLD", confidence=0.0, reason=f"오류: {e}")
