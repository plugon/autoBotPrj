import logging
from typing import Dict, Optional
from dataclasses import dataclass
import pandas as pd
import ta
from ta.trend import ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from config.settings import TRADING_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """거래 신호"""
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0-1 신뢰도
    reason: str
    suggested_quantity: float = 0.0
    suggested_stop_loss: Optional[float] = None  # ATR 기반 추천 손절가
    atr_value: Optional[float] = None  # ATR(N) 값


class TradingStrategy:
    """거래 전략 기본 클래스"""
    
    def __init__(self, lookback_window: int = 60):
        self.lookback_window = lookback_window
    
    def generate_signal(self, symbol: str, data: pd.DataFrame, current_capital: float = 0.0, strategy_override: str = None) -> Optional[Signal]:
        """거래 신호 생성"""
        raise NotImplementedError


class MLStrategy(TradingStrategy):
    """머신러닝 기반 거래 전략"""
    
    def __init__(self, ml_predictor, lookback_window: int = 60, 
                 confidence_threshold: float = 0.6):
        super().__init__(lookback_window)
        self.ml_predictor = ml_predictor
        self.confidence_threshold = confidence_threshold
    
    def generate_signal(self, symbol: str, data: pd.DataFrame, current_capital: float = 0.0, strategy_override: str = None) -> Optional[Signal]:
        """
        머신러닝 모델의 예측을 기반으로 신호 생성
        
        Args:
            symbol: 종목코드
            data: OHLCV 데이터
            current_capital: 현재 가용 자본 (터틀 트레이딩 유닛 계산용)
        
        Returns:
            Signal 객체
        """
        # [Request 1] 최소 데이터 개수 검증
        if len(data) < 20:
            return None

        if len(data) < self.lookback_window:
            return Signal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                reason="데이터 부족"
            )
        
        try:
            current_price = data['close'].iloc[-1]
            
            # ATR 기반 손절가 계산
            atr_window = TRADING_CONFIG["crypto"].get("atr_window", 20)
            atr_multiplier = TRADING_CONFIG["crypto"].get("atr_multiplier", 2.0)
            
            atr_stop_loss = None
            atr = 0.0
            if len(data) >= atr_window:
                atr_indicator = AverageTrueRange(data['high'], data['low'], data['close'], window=atr_window)
                atr = atr_indicator.average_true_range().iloc[-1]
                # 2N 손절가 계산
                atr_stop_loss = current_price - (atr * atr_multiplier)
            else:
                logger.warning(f"{symbol} ATR 계산 데이터 부족. 손절가 미설정.")

            # 터틀 트레이딩 유닛 계산 (1% 리스크)
            suggested_qty = 0.0
            if current_capital > 0:
                # [수정] ATR 0일 경우 임시 변동성(2%) 적용 (Division by Zero 방지)
                calc_atr = atr
                if calc_atr <= 0:
                    calc_atr = current_price * 0.02
                
                stop_dist = calc_atr * atr_multiplier
                
                if stop_dist > 0:
                    # Qty = (Capital * 0.01) / (ATR * Multiplier)
                    risk_amt = current_capital * 0.01
                    suggested_qty = risk_amt / stop_dist

            # 방향성 예측
            direction = self.ml_predictor.predict_direction(
                data, current_price
            )
            
            # 신뢰도 계산 (변동성 기반)
            recent_returns = data['close'].pct_change().tail(20)
            volatility = recent_returns.std()
            confidence = 1 - min(volatility * 2, 1.0)  # 변동성이 낮을수록 신뢰도 높음
            
            if confidence < self.confidence_threshold:
                return Signal(
                    symbol=symbol,
                    action="HOLD",
                    confidence=confidence,
                    reason=f"신뢰도 부족 ({confidence:.2f})"
                )
            
            if direction == "UP":
                return Signal(
                    symbol=symbol,
                    action="BUY",
                    confidence=confidence,
                    reason=f"상승 추세 예상 (신뢰도: {confidence:.2f})",
                    suggested_quantity=suggested_qty if suggested_qty > 0 else 0.0,
                    suggested_stop_loss=atr_stop_loss,
                    atr_value=atr
                )
            elif direction == "DOWN":
                return Signal(
                    symbol=symbol,
                    action="SELL",
                    confidence=confidence,
                    reason=f"하강 추세 예상 (신뢰도: {confidence:.2f})",
                    suggested_quantity=0.0
                )
            else:
                return Signal(
                    symbol=symbol,
                    action="HOLD",
                    confidence=confidence,
                    reason="보합 추세",
                )
        
        except Exception as e:
            logger.error(f"{symbol} 신호 생성 오류: {e}")
            return Signal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                reason=f"오류: {e}"
            )


class TechnicalStrategy(TradingStrategy):
    """기술적 지표 기반 거래 전략"""
    
    def __init__(self, lookback_window: int = 60, confidence_threshold: float = 0.5):
        super().__init__(lookback_window)
        self.confidence_threshold = confidence_threshold
    
    def _check_dynamic_breakout(self, data: pd.DataFrame) -> tuple:
        """
        코인 전용 동적 돌파 전략 (1h/15m 공용)
        1. EMA 50 위에서만 매수 (중기 상승장)
        2. 볼린저 밴드 상단 돌파 시 진입 (강한 모멘텀)
        3. ADX 20 이상 (추세 확인)
        """
        close = data['close']
        # ta 라이브러리 함수 활용
        ema50 = ta.trend.ema_indicator(close, window=50)
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        adx = ta.trend.adx(data['high'], data['low'], close, window=14)
        
        curr_price = close.iloc[-1]
        bb_high = bb.bollinger_hband().iloc[-1]
        curr_ema = ema50.iloc[-1]
        curr_adx = adx.iloc[-1]
        
        # 조건 1: 중기 이평선 위 (상승 추세)
        is_uptrend = curr_price > curr_ema
        # 조건 2: 볼린저 밴드 상단 돌파
        is_breakout = curr_price > bb_high
        # 조건 3: 추세 강도 존재
        is_strong = curr_adx > 20
        
        if is_uptrend and is_breakout and is_strong:
            return True, 0.85, f"Breakout! (Price:{curr_price:,.0f} > BB_High:{bb_high:,.0f})"
        
        return False, 0.0, ""

    def _check_volatility_breakout(self, data: pd.DataFrame) -> tuple:
        """
        변동성 돌파 전략 (Volatility Breakout)
        1. 일봉 기준 Range 계산 (전일 고가 - 전일 저가)
        2. 매수 기준: 당일 시가 + (Range * K)
        3. 필터: 현재가가 EMA 50 위에 있을 때 (중기 상승 추세)
        """
        try:
            # 1. 일봉 데이터로 리샘플링 (Upbit 기준 09:00 KST = 00:00 UTC)
            # ccxt 데이터는 UTC 기준이므로 '1D' 리샘플링 시 00:00 UTC 기준이 됨
            df_daily = data.resample('1D').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
            }).dropna()
            
            # 전일 데이터와 당일 데이터가 모두 필요함 (최소 2일치)
            if len(df_daily) < 2:
                return False, 0.0, "일봉 데이터 부족"

            prev_day = df_daily.iloc[-2]
            today = df_daily.iloc[-1]
            
            # 2. Range 및 돌파 기준 가격 계산
            prev_range = prev_day['high'] - prev_day['low']
            k = TRADING_CONFIG["crypto"].get("k_value", 0.5)
            breakout_price = today['open'] + (prev_range * k)
            
            current_price = data['close'].iloc[-1]
            
            # 3. EMA 50 필터 (중기 상승 추세 확인)
            ema50 = data['close'].ewm(span=50, adjust=False).mean().iloc[-1]
            is_uptrend = current_price > ema50
            
            if current_price > breakout_price and is_uptrend:
                return True, 0.9, f"변동성 돌파 (현재가 {current_price:,.0f} > {breakout_price:,.0f}, K={k})"
                
            return False, 0.0, ""
            
        except Exception as e:
            logger.error(f"변동성 돌파 계산 오류: {e}")
            return False, 0.0, ""

    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> float:
        """RSI 계산"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # 마지막 값 추출 (Series 연산 대신 값 연산으로 변경하여 타입 에러 방지)
        loss_val = loss.iloc[-1]
        gain_val = gain.iloc[-1]
        
        if loss_val == 0:
            return 100.0 if gain_val > 0 else 50.0
            
        rs = gain_val / loss_val
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, data: pd.DataFrame) -> tuple:
        """MACD 계산"""
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        return macd.iloc[-1], signal.iloc[-1], histogram.iloc[-1]
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std_dev: int = 2) -> tuple:
        """볼린저 밴드 계산 (변동성 분석)"""
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        middle_band = sma
        
        current_price = data['close'].iloc[-1]
        return upper_band.iloc[-1], middle_band.iloc[-1], lower_band.iloc[-1], current_price
    
    def calculate_obv(self, data: pd.DataFrame) -> float:
        """OBV (거래량 기반 지표) 계산"""
        obv = (data['volume'] * (data['close'].diff().fillna(0) > 0).astype(int) - 
               data['volume'] * (data['close'].diff().fillna(0) < 0).astype(int)).cumsum()
        
        # OBV 추세 (최근 5일 vs 이전 5일)
        recent_obv = obv.iloc[-5:].mean()
        previous_obv = obv.iloc[-10:-5].mean()
        obv_trend = (recent_obv - previous_obv) / previous_obv if previous_obv != 0 else 0
        
        return obv_trend
    
    def calculate_trendline(self, data: pd.DataFrame, period: int = 20) -> str:
        """추세선 분석 (상승/하강/보합)"""
        recent_closes = data['close'].tail(period).values
        
        # 간단한 선형 회귀로 추세 판단
        x = list(range(len(recent_closes)))
        y = recent_closes
        
        # 기울기 계산
        n = len(x)
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0
        
        # 기울기 기반 추세 판단
        if slope > 0.001:
            return "UPTREND"
        elif slope < -0.001:
            return "DOWNTREND"
        else:
            return "SIDEWAYS"
    
    def calculate_volume_spike(self, data: pd.DataFrame, period: int = 20) -> float:
        """거래량 급등 비율 계산 (현재 거래량 / 평균 거래량)"""
        if len(data) < period + 1:
            return 0.0
            
        # 현재 캔들을 제외한 이전 평균 거래량
        avg_volume = data['volume'].iloc[-(period+1):-1].mean()
        current_volume = data['volume'].iloc[-1]
        
        if avg_volume == 0:
            return 0.0
            
        return current_volume / avg_volume

    def generate_signal(self, symbol: str, data: pd.DataFrame, current_capital: float = 0.0, strategy_override: str = None) -> Optional[Signal]:
        """기술적 지표 기반 신호 생성 (RSI, MACD, 볼린저밴드, OBV, 추세선)"""
        # [Request 1] 최소 데이터 개수 검증
        if len(data) < 20:
            return None

        if len(data) < self.lookback_window:
            return Signal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                reason="데이터 부족"
            )
        
        try:
            # 0. 추세 필터 (ADX) 및 변동성(ATR) 계산
            # ---------------------------------------------------------
            adx_threshold = TRADING_CONFIG["crypto"].get("adx_threshold", 20.0)
            atr_multiplier = TRADING_CONFIG["crypto"].get("atr_multiplier", 2.0)
            atr_window = TRADING_CONFIG["crypto"].get("atr_window", 20)
            volatility_threshold = TRADING_CONFIG["crypto"].get("volatility_threshold", 0.5)

            adx_indicator = ADXIndicator(data['high'], data['low'], data['close'], window=14)
            adx_series = adx_indicator.adx()
            adx = adx_series.iloc[-1]
            prev_adx = adx_series.iloc[-2] if len(adx_series) > 1 else 0
            
            atr_indicator = AverageTrueRange(data['high'], data['low'], data['close'], window=atr_window)
            atr = atr_indicator.average_true_range().iloc[-1]
            current_price = data['close'].iloc[-1]

            # [필터 1] 변동성 필터 (Volatility Filter)
            # ATR이 가격 대비 너무 낮으면(예: 0.5% 미만) 횡보장으로 판단하여 매매를 쉼.
            # 이는 수수료만 나가는 잦은 매매를 방지함.
            volatility_pct = (atr / current_price) * 100
            if volatility_pct < volatility_threshold:
                return Signal(symbol=symbol, action="HOLD", confidence=0.0, 
                              reason=f"변동성 낮음 (ATR {volatility_pct:.2f}% < {volatility_threshold}%)")

            # 설정된 진입 전략 가져오기 (기본값: combined)
            entry_strategy = strategy_override if strategy_override else TRADING_CONFIG["crypto"].get("entry_strategy", "combined").lower()

            # [필터 3] EMA 50 중기 추세 필터 (변경: EMA100 -> EMA50)
            # 하락장(가격 < EMA 50)에서는 매수 신호 생성을 원천 차단
            is_downtrend = False
            # [수정] 전략 타입과 무관하게 추세 필터 적용 (하락장 매수 방지)
            if len(data) >= 50:
                ema50 = data['close'].ewm(span=50, adjust=False).mean().iloc[-1]
                if current_price < ema50:
                    is_downtrend = True

            rsi = self.calculate_rsi(data)
            macd, signal_line, histogram = self.calculate_macd(data)
            upper, middle, lower, current_price = self.calculate_bollinger_bands(data)
            obv_trend = self.calculate_obv(data)
            trendline = self.calculate_trendline(data)
            vol_ratio = self.calculate_volume_spike(data)
            
            signals = []
            
            # [필터 2] 추세 강도 필터 (ADX)
            # ADX > 20 (완화) AND ADX 상승 중 (모멘텀 확인)
            if entry_strategy != "rsi":
                if adx < 20 or adx < prev_adx:
                    signals.append(("HOLD", 0.0, f"추세 약함/정체 (ADX: {adx:.1f})"))
            
            # 1. RSI 전략 (또는 combined)
            if entry_strategy in ["combined", "rsi"]:
                # [개선] 추세에 따른 동적 임계값 적용 (하락장에서도 과매도 심화 시 진입 허용)
                # 상승장(SMA60 위): RSI 35 미만 매수 (눌림목)
                # 하락장(SMA60 아래): RSI 25 미만 매수 (과매도 반등)
                buy_threshold = 25 if is_downtrend else 35
                
                if rsi < buy_threshold:
                    signals.append(("BUY", 0.8, f"RSI 과매도({rsi:.1f})"))
                elif rsi > 70: # 75 -> 70으로 매도 기준 완화 (빠른 익절)
                    signals.append(("SELL", 0.8, f"RSI 과매수({rsi:.1f})"))
            
            # 2. MACD 전략 (또는 combined)
            if entry_strategy in ["combined", "macd"]:
                if histogram > 0 and macd > signal_line and not is_downtrend:
                    signals.append(("BUY", 0.7, "MACD 골든크로스"))
                elif histogram < 0 and macd < signal_line:
                    signals.append(("SELL", 0.7, "MACD 데드크로스"))
            
            # 3. 볼린저 밴드 전략 (또는 combined)
            if entry_strategy in ["combined", "bollinger"]:
                if current_price < lower and not is_downtrend:
                    signals.append(("BUY", 0.8, "볼린저밴드 하단 돌파"))
                elif current_price > upper:
                    signals.append(("SELL", 0.8, "볼린저밴드 상단 돌파"))
            
            # 4. 추세 추종 (OBV + 추세선) - combined에 포함
            if entry_strategy == "combined":
                # OBV 신호
                if obv_trend > 0.02 and not is_downtrend:
                    signals.append(("BUY", 0.5, "OBV 상승"))
                elif obv_trend < -0.02:
                    signals.append(("SELL", 0.5, "OBV 하락"))
                
                # 추세선 신호
                if trendline == "UPTREND" and not is_downtrend:
                    signals.append(("BUY", 0.6, "상승 추세선"))
                elif trendline == "DOWNTREND":
                    signals.append(("SELL", 0.6, "하강 추세선"))
            
            # 거래량 급등 신호 (단타 강화)
            # breakout 전략이거나 combined일 때 적용
            if entry_strategy in ["combined", "breakout"]:
                if vol_ratio >= 3.0:  # 3배 이상 폭발적 증가
                    if data['close'].iloc[-1] > data['open'].iloc[-1]:
                        if not is_downtrend:
                            signals.append(("BUY", 0.9, f"거래량 폭발 ({vol_ratio:.1f}배)"))
                    else:
                        signals.append(("SELL", 0.9, f"투매 발생 ({vol_ratio:.1f}배)"))
                elif vol_ratio >= 2.0 and data['close'].iloc[-1] > data['open'].iloc[-1]:
                    # 2배 이상 증가하며 양봉인 경우
                    if not is_downtrend:
                        signals.append(("BUY", 0.7, f"거래량 급등 ({vol_ratio:.1f}배)"))

            # 5. 볼린저 밴드 돌파 (Trend Following) - [New]
            # 역추세(RSI)와 달리 상승 추세가 확정될 때 진입하여 큰 파동을 먹는 전략
            if entry_strategy == "bb_breakout":
                # 이전 캔들 정보 확인을 위해 시리즈 계산 필요
                sma = data['close'].rolling(window=20).mean()
                std = data['close'].rolling(window=20).std()
                upper_series = sma + (std * 2)
                middle_series = sma
                lower_series = sma - (std * 2)
                
                # [필터] 밴드폭(Bandwidth) 계산: (상단-하단)/중앙
                # 밴드폭이 이전보다 커지고 있어야 함 (확장 국면)
                bandwidth = (upper_series - lower_series) / middle_series
                current_bw = bandwidth.iloc[-1]
                prev_bw = bandwidth.iloc[-2]
                
                # [필터] 거래량 급등 확인
                vol_ratio = self.calculate_volume_spike(data)
                
                # [매수] 가격이 상단 밴드를 돌파할 때 (강한 상승세 시작)
                # 조건: 상단 돌파 + 거래량 1.2배 이상 + 밴드폭 확장
                # [수정] 하락 추세(SMA60 아래)가 아닐 때만 진입 (Trend Filter)
                if data['close'].iloc[-2] <= upper_series.iloc[-2] and current_price > upper_series.iloc[-1]:
                    if vol_ratio >= 1.2 and current_bw > prev_bw and not is_downtrend:
                        signals.append(("BUY", 0.9, f"BB 돌파+거래량({vol_ratio:.1f}배)+확장"))
                
                # [매도] 가격이 중앙선(20일선) 아래로 내려올 때 (추세 종료)
                elif current_price < middle_series.iloc[-1]:
                    signals.append(("SELL", 0.8, "BB 중앙선 이탈 (추세 종료)"))

            # 6. 추세 추종 눌림목 (Trend Pullback) - [New]
            # 장기 이평선(EMA200) 위에 있을 때, 단기 과매도(RSI 눌림)에서 매수
            if entry_strategy == "pullback":
                # [MTF 필터] 상위 타임프레임(1h) 추세 확인
                # 15분봉 기준 400개 캔들이면 약 100시간. 1시간봉 100개 생성 가능.
                is_higher_tf_uptrend = True
                try:
                    # 데이터가 충분할 때만 리샘플링 수행
                    if len(data) >= 60:
                        # 1시간봉으로 리샘플링 (1h)
                        df_1h = data.resample('1h').agg({
                            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                        }).dropna()
                        
                        if len(df_1h) >= 20:
                            # 1시간봉 기준 EMA20 (단기 추세)
                            ema20_1h = df_1h['close'].ewm(span=20, adjust=False).mean().iloc[-1]
                            if df_1h['close'].iloc[-1] < ema20_1h:
                                is_higher_tf_uptrend = False
                except Exception as e:
                    logger.warning(f"MTF 분석 오류: {e}")

                # EMA 100 계산 (장기 추세선, 200 -> 100 완화)
                ema100_long = data['close'].ewm(span=100, adjust=False).mean().iloc[-1]
                
                # RSI 시리즈 계산 (이전 값 비교용)
                rsi_indicator = RSIIndicator(close=data['close'], window=14)
                rsi_series = rsi_indicator.rsi()
                current_rsi = rsi_series.iloc[-1]
                prev_rsi = rsi_series.iloc[-2]
                
                # [매수] 가격 > EMA100 (상승장) AND 상위차트 상승(MTF) AND RSI < 50 (눌림목 상향) AND RSI 반등
                if current_price > ema100_long and is_higher_tf_uptrend:
                    if current_rsi < 50 and current_rsi > prev_rsi:
                        signals.append(("BUY", 0.85, f"상승추세 눌림목 (RSI {current_rsi:.1f}⤴)"))
                
                # [매도] RSI 과매수 구간 도달 시 (단기 고점)
                elif current_rsi > 70:
                    signals.append(("SELL", 0.8, f"RSI 과매수 ({current_rsi:.1f})"))

            # 7. 동적 돌파 전략 (Dynamic Breakout) - [New]
            if entry_strategy == "dynamic_breakout":
                is_signal, conf, reason = self._check_dynamic_breakout(data)
                if is_signal:
                    signals.append(("BUY", conf, reason))

            # 8. 변동성 돌파 전략 (Volatility Breakout) - [New]
            if entry_strategy == "breakout":
                is_signal, conf, reason = self._check_volatility_breakout(data)
                if is_signal:
                    signals.append(("BUY", conf, reason))
            
            # 9. RSI + 볼린저 밴드 조합 전략 (역추세 매매) - [New]
            if entry_strategy == "rsi_bollinger":
                # 매수 조건: RSI < 30 (과매도) AND 가격 < 볼린저 밴드 하단 (강한 반등 예상 구간)
                if rsi < 30 and current_price < lower:
                    # 하락 추세가 너무 강할 때(ADX > 50)는 위험할 수 있으나, 기본 로직대로 진입
                    signals.append(("BUY", 0.85, f"RSI({rsi:.1f}) 과매도 + BB 하단 돌파"))
                
                # 매도 조건: RSI > 70 (과매수) OR 가격 > 볼린저 밴드 상단
                elif rsi > 70 or current_price > upper:
                    signals.append(("SELL", 0.8, f"RSI({rsi:.1f}) 과매수 또는 BB 상단 돌파"))

            # 홀드 신호가 있으면 다른 신호 무시
            if any(action == "HOLD" for action, _, _ in signals):
                return Signal(symbol=symbol, action="HOLD", confidence=0.0, reason=f"필터 조건 미충족 (ADX: {adx:.1f})")

            if not signals:
                return Signal(
                    symbol=symbol,
                    action="HOLD",
                    confidence=0.5,
                    reason="중립 신호 (추세: {})".format(trendline)
                )
            
            # 신호 집계
            buy_count = sum(1 for action, _, _ in signals if action == "BUY")
            sell_count = sum(1 for action, _, _ in signals if action == "SELL")
            avg_confidence = sum(conf for _, conf, _ in signals) / len(signals)
            
            # 상세 신호 이유
            reasons = " + ".join([reason for _, _, reason in signals[:2]])
            
            # [신뢰도 필터] 설정된 임계값보다 신뢰도가 낮으면 진입 보류
            if avg_confidence < self.confidence_threshold:
                return Signal(
                    symbol=symbol, action="HOLD", confidence=avg_confidence,
                    reason=f"신뢰도 부족 ({avg_confidence:.2f} < {self.confidence_threshold})"
                )

            # [자금 관리] ATR 기반 가변 손절가 계산 (2N Rule: 현재가 - 2 * ATR)
            atr_stop_loss = current_price - (atr * atr_multiplier)
            
            # [개선] 직전 저점 기반 손절가 (Recent Lows) - 최근 3개 캔들 최저점
            recent_low_stop_loss = data['low'].tail(3).min()

            # 터틀 트레이딩 유닛 계산 (1% 리스크)
            suggested_qty = 0.0
            if current_capital > 0:
                # [수정] ATR 0일 경우 임시 변동성(2%) 적용 (Division by Zero 방지)
                calc_atr = atr
                if calc_atr <= 0:
                    calc_atr = current_price * 0.02
                    logger.debug(f"{symbol} ATR 0 -> 임시 변동성(2%) 적용")

                # [Safety] ATR이 너무 작으면(0.1% 미만) 최소값 보정
                min_atr = current_price * 0.001
                safe_atr = max(calc_atr, min_atr)
                
                stop_dist = safe_atr * atr_multiplier
                
                if stop_dist > 0:
                    risk_amt = current_capital * 0.01
                    suggested_qty = risk_amt / stop_dist

            # 손절가 결정: 직전 저점이 있으면 우선 사용 (유연한 대응), 없으면 ATR 사용
            # 단, 직전 저점이 현재가와 너무 가까우면(0.5% 미만) ATR 사용
            final_stop_loss = recent_low_stop_loss if recent_low_stop_loss < current_price * 0.995 else atr_stop_loss

            if buy_count > sell_count:
                return Signal(
                    symbol=symbol,
                    action="BUY",
                    confidence=avg_confidence,
                    reason=f"매수 신호: {reasons} (ADX:{adx:.1f})",
                    suggested_quantity=suggested_qty if suggested_qty > 0 else 0.0,
                    suggested_stop_loss=final_stop_loss,
                    atr_value=atr
                )
            else:
                return Signal(
                    symbol=symbol,
                    action="SELL",
                    confidence=avg_confidence,
                    reason=f"매도 신호: {reasons}"
                )
        
        except Exception as e:
            logger.error(f"{symbol} 신호 생성 오류: {e}")
            return Signal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                reason=f"오류: {e}"
            )
