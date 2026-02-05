import logging
import pandas as pd
import ta
from trading.strategy import TradingStrategy, Signal

logger = logging.getLogger(__name__)

class AgileStrategy(TradingStrategy):
    """
    기민한 스캘핑/단타 전략 (Agile Strategy)
    시장 상황에 빠르게 반응하여 수수료를 상회하는 짧은 수익을 누적하는 것을 목표로 합니다.
    
    [특징]
    - 타임프레임: 1분(1m), 3분(3m), 5분(5m), 15분(15m) 권장
    - 진입: 과매도 반등(RSI+BB) 또는 단기 추세 시작(EMA)
    - 청산: 과매수 도달 시 즉시 청산 (줄 때 먹기)
    """
    
    def __init__(self, lookback_window: int = 60):
        super().__init__(lookback_window)

    def generate_signal(self, symbol: str, data: pd.DataFrame, current_capital: float = 0.0, strategy_override: str = None, **kwargs) -> Signal:
        # 데이터 검증 (최소 30개 캔들 필요)
        if len(data) < 30:
            return None
            
        try:
            # 1. 지표 계산
            close = data['close']
            high = data['high']
            low = data['low']
            
            # RSI (14) - 민감도 높음
            rsi = ta.momentum.rsi(close, window=14).iloc[-1]
            
            # Bollinger Bands (20, 2) - 변동성 및 이탈 확인
            bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
            lower_band = bb.bollinger_lband().iloc[-1]
            upper_band = bb.bollinger_hband().iloc[-1]
            middle_band = bb.bollinger_mavg().iloc[-1]
            
            # Stochastic (14, 3, 3) - 빠른 반전 신호
            stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
            stoch_k = stoch.stoch().iloc[-1]
            
            # EMA (Fast=9, Slow=21) - 단기 추세 교차
            ema_fast = ta.trend.ema_indicator(close, window=9).iloc[-1]
            ema_slow = ta.trend.ema_indicator(close, window=21).iloc[-1]
            
            # ATR (14) - 변동성 (손절가 계산 및 리스크 관리용)
            atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range().iloc[-1]
            
            current_price = close.iloc[-1]
            
            # ---------------------------------------------------------
            # 2. 매수 조건 (진입)
            # ---------------------------------------------------------
            
            # [조건 A] 역추세 스캘핑: 과매도 상태에서 밴드 하단 지지 (반등 노림)
            # RSI < 35 OR 스토캐스틱 < 20 AND 가격이 하단 밴드 근처
            is_oversold = rsi < 35 or stoch_k < 20
            is_dip = current_price <= lower_band * 1.005 # 하단 밴드 0.5% 이내 접근
            
            if is_oversold and is_dip:
                return Signal(
                    symbol=symbol,
                    action="BUY",
                    confidence=0.85,
                    reason=f"⚡ 과매도 반등 (RSI:{rsi:.1f}, BB하단)",
                    suggested_stop_loss=current_price * 0.99, # 타이트한 손절 (-1%)
                    atr_value=atr
                )
            
            # [조건 B] 추세 스캘핑: 상승 추세에서 눌림목 (중앙선 지지)
            # EMA 정배열(상승세) AND 가격이 볼린저 중앙선 지지
            is_trend_up = ema_fast > ema_slow
            is_support = (middle_band * 0.995 <= current_price <= middle_band * 1.005)
            
            if is_trend_up and is_support:
                if rsi < 60: # 아직 과열되지 않음
                    return Signal(
                        symbol=symbol,
                        action="BUY",
                        confidence=0.75,
                        reason=f"📈 상승추세 눌림목 (EMA정배열 + BB중앙)",
                        suggested_stop_loss=current_price * 0.99,
                        atr_value=atr
                    )

            # ---------------------------------------------------------
            # 3. 매도 조건 (청산 - 가볍게 먹고 나오기)
            # ---------------------------------------------------------
            
            # [조건 A] 과매수 도달 (욕심 부리지 않고 청산)
            # RSI > 70 OR 스토캐스틱 > 80
            is_overbought = rsi > 70 or stoch_k > 80
            
            # [조건 B] 볼린저 상단 터치 (단기 고점)
            is_peak = current_price >= upper_band * 0.995
            
            if is_overbought or is_peak:
                return Signal(
                    symbol=symbol,
                    action="SELL",
                    confidence=0.8,
                    reason=f"💰 단기 고점 도달 (RSI:{rsi:.1f}, BB상단)",
                    atr_value=atr
                )
                
            # [조건 C] 단기 추세 이탈 (EMA 데드크로스)
            if ema_fast < ema_slow:
                 return Signal(
                    symbol=symbol,
                    action="SELL",
                    confidence=0.7,
                    reason="📉 단기 추세 이탈 (EMA 데드크로스)",
                    atr_value=atr
                )

            return Signal(symbol, "HOLD", 0.5, "관망", atr_value=atr)
            
        except Exception as e:
            logger.error(f"{symbol} AgileStrategy 오류: {e}")
            return None
