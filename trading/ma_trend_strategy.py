import logging
import pandas as pd
import ta
from trading.strategy import TradingStrategy, Signal

logger = logging.getLogger(__name__)

class MATrendStrategy(TradingStrategy):
    """
    바이낸스 선물 전용 이평선 추세 추종 전략 (MA Trend Strategy)
    
    [전략 개요]
    1. 타임프레임: 5분봉 (Trigger) + 4시간봉 (Trend Filter)
    2. 진입 필터 (4h):
       - 정배열 (SMA 10 > 20 > 50): LONG ONLY
       - 역배열 (SMA 50 > 20 > 10): SHORT ONLY
    3. 진입 타점 (5m):
       - 정배열/역배열 진입 시 (Golden/Dead Cross Alignment)
       - 또는 정배열 상태에서 20 SMA 터치 시 (Pullback)
    4. 청산:
       - 반대 방향 이평선 크로스 발생 시
    """
    
    def __init__(self, lookback_window: int = 200):
        super().__init__(lookback_window)

    def generate_signal(self, symbol: str, data: pd.DataFrame, current_capital: float = 0.0, strategy_override: str = None, **kwargs) -> Signal:
        # 데이터 검증 (4시간봉 SMA 50 계산을 위해 5분봉 기준 최소 2400개 필요하나, 
        # 리샘플링 오차 고려하여 3000개 권장. 부족하면 가능한 만큼만 계산)
        if len(data) < 200:
            return Signal(symbol, "HOLD", 0.0, "데이터 부족")

        try:
            current_price = data['close'].iloc[-1]
            
            # 1. 4시간봉 리샘플링 및 SMA 계산 (Trend Filter)
            # 데이터 인덱스가 datetime인지 확인
            df_source = data.copy()
            if not isinstance(df_source.index, pd.DatetimeIndex):
                if 'timestamp' in df_source.columns:
                    df_source.set_index('timestamp', inplace=True)
                    df_source.index = pd.to_datetime(df_source.index)
            
            # 4시간봉 변환 (5분봉 -> 4시간봉)
            df_4h = df_source.resample('4h').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
            }).dropna()
            
            trend_direction = "NEUTRAL"
            
            if len(df_4h) >= 50:
                sma10_4h = ta.trend.sma_indicator(df_4h['close'], window=10).iloc[-1]
                sma20_4h = ta.trend.sma_indicator(df_4h['close'], window=20).iloc[-1]
                sma50_4h = ta.trend.sma_indicator(df_4h['close'], window=50).iloc[-1]
                
                if sma10_4h > sma20_4h > sma50_4h:
                    trend_direction = "LONG"
                elif sma50_4h > sma20_4h > sma10_4h:
                    trend_direction = "SHORT"
            else:
                # 데이터 부족 시 보수적으로 관망 (또는 5분봉 EMA 200으로 대체 가능)
                return Signal(symbol, "HOLD", 0.0, "4시간봉 데이터 부족 (Trend Filter 불가)")

            # 2. 5분봉 SMA 계산 (Trigger)
            sma10_5m = ta.trend.sma_indicator(data['close'], window=10)
            sma20_5m = ta.trend.sma_indicator(data['close'], window=20)
            sma50_5m = ta.trend.sma_indicator(data['close'], window=50)
            
            curr_sma10 = sma10_5m.iloc[-1]
            curr_sma20 = sma20_5m.iloc[-1]
            curr_sma50 = sma50_5m.iloc[-1]
            
            prev_sma10 = sma10_5m.iloc[-2]
            prev_sma20 = sma20_5m.iloc[-2]
            prev_sma50 = sma50_5m.iloc[-2]
            
            # ATR 계산 (레버리지 및 손절용)
            atr = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close'], window=14).average_true_range().iloc[-1]
            
            # 3. 진입 로직
            signal_action = "HOLD"
            reason = ""
            confidence = 0.0
            
            # LONG 진입 조건
            if trend_direction == "LONG":
                # 조건 A: 5분봉 정배열 진입 (Alignment Start)
                is_aligned = (curr_sma10 > curr_sma20 > curr_sma50)
                was_aligned = (prev_sma10 > prev_sma20 > prev_sma50)
                
                # 조건 B: 정배열 상태에서 20 SMA 터치 (Pullback)
                # 저가가 20 SMA를 터치하고 종가는 그 위에 있음 (지지도 확인)
                is_touch_20 = (data['low'].iloc[-1] <= curr_sma20) and (data['close'].iloc[-1] > curr_sma20)
                
                if (is_aligned and not was_aligned) or (is_aligned and is_touch_20):
                    signal_action = "BUY"
                    confidence = 0.85
                    reason = f"4H 상승장 + 5M 정배열/눌림 (Price: {current_price:.0f})"

            # SHORT 진입 조건
            elif trend_direction == "SHORT":
                # 조건 A: 5분봉 역배열 진입
                is_aligned = (curr_sma50 > curr_sma20 > curr_sma10)
                was_aligned = (prev_sma50 > prev_sma20 > prev_sma10)
                
                # 조건 B: 역배열 상태에서 20 SMA 터치 (Pullback)
                # 고가가 20 SMA를 터치하고 종가는 그 아래 있음 (저항 확인)
                is_touch_20 = (data['high'].iloc[-1] >= curr_sma20) and (data['close'].iloc[-1] < curr_sma20)
                
                if (is_aligned and not was_aligned) or (is_aligned and is_touch_20):
                    signal_action = "SELL"
                    confidence = 0.85
                    reason = f"4H 하락장 + 5M 역배열/반등 (Price: {current_price:.0f})"

            # 4. 청산 로직 (반대 크로스) - 포지션 보유 중일 때 유효
            # 여기서는 진입 신호와 별개로 반대 방향 신호를 생성하여 봇이 스위칭하거나 청산하게 함
            if trend_direction == "LONG" and curr_sma10 < curr_sma20:
                 # 상승 추세 중 단기 이평 데드크로스 -> 청산 신호
                 # (봇 로직상 SELL 신호는 롱 포지션 청산으로 동작)
                 if signal_action == "HOLD": # 진입 신호가 없을 때만 청산 신호 발생
                     return Signal(symbol, "SELL", 0.7, "단기 이평 데드크로스 (청산)")

            if trend_direction == "SHORT" and curr_sma10 > curr_sma20:
                 # 하락 추세 중 단기 이평 골든크로스 -> 청산 신호
                 if signal_action == "HOLD":
                     return Signal(symbol, "BUY", 0.7, "단기 이평 골든크로스 (청산)")

            # 레버리지 계산 (ATR 기반)
            # 변동성이 낮으면 레버리지 높게, 높으면 낮게 (기본 1% 변동성 타겟)
            target_volatility = 0.01 # 1%
            current_volatility = atr / current_price if current_price > 0 else 0.01
            suggested_leverage = max(1, min(int(target_volatility / current_volatility), 10))
            
            if signal_action != "HOLD":
                return Signal(
                    symbol=symbol,
                    action=signal_action,
                    confidence=confidence,
                    reason=reason,
                    atr_value=atr,
                    suggested_leverage=suggested_leverage,
                    suggested_stop_loss=current_price - (atr * 2) if signal_action == "BUY" else current_price + (atr * 2)
                )
                
            return Signal(symbol, "HOLD", 0.0, f"Trend: {trend_direction}")
            
        except Exception as e:
            logger.error(f"MATrendStrategy 오류: {e}")
            return Signal(symbol, "HOLD", 0.0, f"오류: {e}")