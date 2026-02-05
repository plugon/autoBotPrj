import logging
import pandas as pd
import ta
from trading.strategy import TradingStrategy, Signal

logger = logging.getLogger(__name__)

class EarlyBirdStrategy(TradingStrategy):
    """
    이평선 정배열 완성 직전 선취매 (Early Bird) 전략
    
    [전략 개요]
    1. 이평선 상태: SMA(10) > SMA(20) > SMA(50) (정배열)
    2. 성장성(기울기):
       - SMA(10), SMA(20) 기울기 양수
       - SMA(50) 기울기가 음수에서 0으로 수렴하거나, 하락세 둔화 (2차 미분 > 0)
    3. 진입(Entry):
       - 위 조건 충족 시, 가격이 SMA(20) 부근으로 눌림목(Retracement) 발생 시 Long
    4. 청산(Exit):
       - SMA(10) < SMA(20) 데드크로스 시 즉시 손절
       - SMA(50) 상향 반전 시 ATR 기반 트레일링 스탑 가동 (RiskManager 위임)
    """
    
    def __init__(self, lookback_window: int = 200):
        super().__init__(lookback_window)

    def generate_signal(self, symbol: str, data: pd.DataFrame, current_capital: float = 0.0, strategy_override: str = None, **kwargs) -> Signal:
        # 데이터 검증 (SMA 50 및 기울기 계산을 위해 최소 60개 필요)
        if len(data) < 60:
            return Signal(symbol, "HOLD", 0.0, "데이터 부족")

        try:
            close = data['close']
            low = data['low']
            current_price = close.iloc[-1]
            
            # [New] 포지션 정보 (동적 청산용)
            entry_price = kwargs.get('entry_price', 0.0)
            position_quantity = kwargs.get('position_quantity', 0.0)
            
            # 1. 이평선 계산
            sma10 = ta.trend.sma_indicator(close, window=10)
            sma20 = ta.trend.sma_indicator(close, window=20)
            sma50 = ta.trend.sma_indicator(close, window=50)
            
            curr_sma10 = sma10.iloc[-1]
            curr_sma20 = sma20.iloc[-1]
            curr_sma50 = sma50.iloc[-1]
            
            # 2. 기울기 계산 (최근 3개 봉 기준 변화량)
            slope_window = 3
            slope_sma10 = (curr_sma10 - sma10.iloc[-1 - slope_window]) / slope_window
            slope_sma20 = (curr_sma20 - sma20.iloc[-1 - slope_window]) / slope_window
            
            # SMA50 기울기 및 변화량 (2차 미분: 가속도)
            prev_sma50 = sma50.iloc[-1 - slope_window]
            prev_prev_sma50 = sma50.iloc[-1 - 2*slope_window]
            
            slope_sma50 = (curr_sma50 - prev_sma50) / slope_window
            prev_slope_sma50 = (prev_sma50 - prev_prev_sma50) / slope_window
            
            # 기울기 변화량 (양수면 하락세 둔화 또는 상승세 가속)
            slope_change_sma50 = slope_sma50 - prev_slope_sma50 
            
            # ATR 계산 (손절/익절용)
            atr = ta.volatility.AverageTrueRange(data['high'], data['low'], close, window=14).average_true_range().iloc[-1]
            
            # 3. 동적 청산 시스템 (Exit Logic)
            if position_quantity > 0 and entry_price > 0:
                # 수익률 계산 (Zero Division 방지)
                profit_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
                
                # 3-1. 이평선 기반 트레일링 스탑 (Trailing Stop)
                # 수익이 +1.0% 이상일 때, 가격이 5분봉 20 SMA 아래로 내려가면 청산
                if profit_pct >= 0.01:
                    if current_price < curr_sma20:
                        return Signal(symbol, "SELL", 1.0, f"트레일링 스탑 발동 (수익 {profit_pct*100:.2f}% > 1.0% & 가격 < SMA20)")

                # 3-2. 추세 종료 감지 (Exit Signal)
                # 데드크로스 청산: SMA(10) < SMA(20)
                if curr_sma10 < curr_sma20:
                    return Signal(symbol, "SELL", 0.9, "SMA 10/20 데드크로스 (추세 종료)")
                
                # 최종 방어선: 가격 < SMA(50)
                if current_price < curr_sma50:
                    return Signal(symbol, "SELL", 1.0, "최종 방어선 이탈 (가격 < SMA50)")

                # 3-3. 과열 구간 분할 익절 (Take Profit)
                # 현재가와 SMA(50) 거리 > 최근 100봉 평균 이격의 2배
                dist_sma50 = abs(close - sma50)
                avg_dist_100 = dist_sma50.rolling(window=100).mean().iloc[-1]
                
                # Zero Division 방지
                if avg_dist_100 > 0:
                    current_dist = abs(current_price - curr_sma50)
                    if current_dist > (avg_dist_100 * 2.0):
                        # 보유 물량의 50% 익절
                        partial_qty = position_quantity * 0.5
                        return Signal(
                            symbol=symbol, 
                            action="SELL", 
                            confidence=0.8, 
                            reason=f"과열 구간 도달 (이격 {current_dist:.0f} > 평균 {avg_dist_100:.0f}*2)",
                            suggested_quantity=partial_qty
                        )

                # 3-4. 본절가 보호 (Breakeven)
                # 수익률 +2.0% 도달 시, 현재가가 진입가 아래로 내려가면 청산 (이미 손실 구간이면 즉시 청산)
                # Note: 캔들 고가가 2% 이상 찍었으나 현재 수익률이 0.2% 미만으로 떨어졌다면 본절 위협으로 간주
                if profit_pct < 0.002 and data['high'].iloc[-1] >= entry_price * 1.02:
                     return Signal(symbol, "SELL", 1.0, "본절가 위협 (최고점 2% 달성 후 하락)")

            # 4. 진입 조건 확인
            
            # 조건 A: 정배열 상태 (SMA 10 > 20 > 50)
            is_aligned = curr_sma10 > curr_sma20 > curr_sma50
            
            # 조건 B: 단기 이평선 성장성 (기울기 양수)
            is_growing = slope_sma10 > 0 and slope_sma20 > 0
            
            # 조건 C: 장기 이평선(SMA50) 턴어라운드 감지
            # 1. 기울기가 양수 (이미 상승 전환) OR
            # 2. 기울기가 음수지만 0에 가까움 (평단 수렴, 0.05% 이내) OR
            # 3. 하락세가 둔화됨 (기울기가 증가함, 즉 2차 미분 > 0)
            is_sma50_turning = (slope_sma50 > 0) or \
                               (slope_sma50 <= 0 and (abs(slope_sma50) < (curr_sma50 * 0.0005) or slope_change_sma50 > 0))
            
            # 조건 D: 눌림목 (Pullback)
            # 가격이 SMA20 부근까지 내려옴 (SMA20의 0.5% 이내 접근)
            # 저가가 SMA20 근처까지 내려왔고, 종가는 지지받음
            dist_to_sma20 = (low.iloc[-1] - curr_sma20) / curr_sma20
            # SMA20 위 0.5% ~ 아래 0.5% 구간 터치
            is_pullback = (dist_to_sma20 <= 0.005) and (close.iloc[-1] >= curr_sma20 * 0.995)
            
            if is_aligned and is_growing and is_sma50_turning and is_pullback:
                # SMA50이 이미 상승 반전했다면 더 강한 신호
                confidence = 0.9 if slope_sma50 > 0 else 0.8
                
                # SMA50이 상승 중이면 트레일링 스탑을 타이트하게 잡음 (2 ATR), 아니면 여유 있게 (3 ATR)
                stop_loss_buffer = 2.0 if slope_sma50 > 0 else 3.0
                
                return Signal(
                    symbol=symbol,
                    action="BUY",
                    confidence=confidence,
                    reason=f"Early Bird: 정배열+SMA50턴+눌림 (Slope50: {slope_sma50:.2f})",
                    suggested_stop_loss=current_price - (atr * stop_loss_buffer),
                    atr_value=atr
                )

            return Signal(symbol, "HOLD", 0.5, "관망", atr_value=atr)

        except Exception as e:
            logger.error(f"EarlyBirdStrategy 오류: {e}")
            return Signal(symbol, "HOLD", 0.0, f"오류: {e}")
