import logging
import pandas as pd
import ta
from trading.strategy import TradingStrategy, Signal

logger = logging.getLogger(__name__)

class VolumeTrendStrategy(TradingStrategy):
    """
    거래량 급등 + 이평선 추세 추종 전략 (Volume Trend Strategy)
    
    [전략 개요]
    - 타임프레임: 5분봉 (권장)
        - Long 진입: 거래량 > 200이평 * 10 (1000%) AND 이평선 정배열 (20 > 60 > 120)
        - Short 진입: 거래량 > 200이평 * 5 (500%) AND 이평선 역배열 (20 < 60 < 120)
    - Long 진입: 거래량 > 200이평 * 3 (300%) AND 이평선 정배열 (20 > 60 > 120)
    - Short 진입: 거래량 > 200이평 * 2 (200%) AND 이평선 역배열 (20 < 60 < 120)
    """
    
    def __init__(self, lookback_window: int = 200):
        super().__init__(lookback_window)

    def generate_signal(self, symbol: str, data: pd.DataFrame, current_capital: float = 0.0, strategy_override: str = None, **kwargs) -> Signal:
        # 데이터 검증 (200이평 계산을 위해 최소 200개 필요)
        if len(data) < 200:
            return None
            
        try:
            close = data['close']
            volume = data['volume']
            
            # 1. 지표 계산
            # 거래량 200 SMA
            vol_sma_200 = volume.rolling(window=200).mean().iloc[-1]
            current_vol = volume.iloc[-1]
            
            # 이동평균선 (20, 60, 120) - 정배열/역배열 판단용
            sma_20 = ta.trend.sma_indicator(close, window=20).iloc[-1]
            sma_60 = ta.trend.sma_indicator(close, window=60).iloc[-1]
            sma_120 = ta.trend.sma_indicator(close, window=120).iloc[-1]
            
            # ATR (손절/익절용)
            atr = ta.volatility.AverageTrueRange(data['high'], data['low'], close, window=14).average_true_range().iloc[-1]
            
            current_price = close.iloc[-1]
            
            # 거래량 비율 (0 나누기 방지)
            vol_ratio = current_vol / vol_sma_200 if vol_sma_200 > 0 else 0
            
            # 2. 진입 조건 확인
            
            # [Long 조건]
            # 1. 거래량 300% (3배) 이상 (메이저 코인 특성 반영 완화)
            # 2. 정배열 (20 > 60 > 120)
            is_vol_long = vol_ratio >= 3.0
            is_trend_up = sma_20 > sma_60 > sma_120
            
            if is_vol_long and is_trend_up:
                return Signal(
                    symbol=symbol,
                    action="BUY",
                    confidence=0.9,
                    reason=f"🚀 거래량 폭발({vol_ratio:.1f}배) + 정배열",
                    suggested_stop_loss=current_price - (atr * 3.0), # 변동성이 크므로 여유있게 3ATR
                    atr_value=atr
                )
                
            # [Short 조건]
            # 1. 거래량 200% (2배) 이상 (완화)
            # 2. 역배열 (20 < 60 < 120)
            is_vol_short = vol_ratio >= 2.0
            is_trend_down = sma_20 < sma_60 < sma_120
            
            if is_vol_short and is_trend_down:
                return Signal(
                    symbol=symbol,
                    action="SELL",
                    confidence=0.9,
                    reason=f"📉 거래량 급증({vol_ratio:.1f}배) + 역배열",
                    suggested_stop_loss=current_price + (atr * 3.0),
                    atr_value=atr
                )
                
            return Signal(symbol, "HOLD", 0.5, "관망", atr_value=atr)
            
        except Exception as e:
            logger.error(f"{symbol} VolumeTrendStrategy 오류: {e}")
            return None
