"""
백테스팅 모듈 - Walk Forward Analysis
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from models.ml_model import MLPredictor
from trading.strategy import MLStrategy, TechnicalStrategy, Signal


class WalkForwardAnalyzer:
    """전진분석(Walk-Forward Analysis) 백테스팅"""
    
    def __init__(self, data: pd.DataFrame, train_period: int = 252, 
                 test_period: int = 63, slippage: float = 0.0005, fee: float = 0.0005,
                 stop_loss: float = None, take_profit: float = None, 
                 trailing_stop: float = None, confidence_threshold: float = 0.5):
        """
        Args:
            data: OHLCV 데이터
            train_period: 학습 기간 (거래일)
            test_period: 테스트 기간 (거래일)
            slippage: 슬리피지 (기본 0.05%) - 체결 오차 반영
            fee: 거래 수수료 (기본 0.05%)
            stop_loss: 손절 기준 (예: 0.02 = 2%)
            take_profit: 익절 기준 (예: 0.04 = 4%)
            trailing_stop: 트레일링 스탑 기준 (예: 0.015 = 1.5%)
            confidence_threshold: 전략 신뢰도 임계값 (기본 0.5)
        """
        self.data = data
        self.train_period = train_period
        self.test_period = test_period
        self.slippage = slippage
        self.fee = fee
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop
        self.confidence_threshold = confidence_threshold
        self.results = []
    
    def run(self, strategy_type: str = "technical", model_type: str = "lstm") -> pd.DataFrame:
        """전진분석 실행"""
        step_size = self.test_period
        lookback = 400  # [수정] MTF 분석을 위해 룩백 기간 상향 (60 -> 400)
        
        # 마지막 구간까지 포함하기 위해 range 범위 수정 (+1)
        for i in range(0, len(self.data) - self.train_period - self.test_period + 1, step_size):
            # 학습 구간
            train_data = self.data.iloc[i:i+self.train_period]
            
            # 테스트 구간 (Lookback Window 포함)
            # _backtest_period에서 과거 60개 데이터를 참조하므로 이를 포함해서 잘라야 함
            test_start_idx = i + self.train_period
            slice_start = max(0, test_start_idx - lookback)
            slice_end = test_start_idx + self.test_period
            
            test_data = self.data.iloc[slice_start:slice_end]
            
            # 데이터가 충분하지 않으면 스킵
            if len(test_data) <= lookback:
                continue
            
            # 전략 초기화
            if strategy_type == "ml":
                model = MLPredictor(lookback_window=60, model_type=model_type)
                model.train(train_data, epochs=30)
                # MLStrategy 래퍼 사용 (가정) 또는 직접 예측
                strategy = MLStrategy(model, lookback_window=lookback)
            elif isinstance(strategy_type, object) and not isinstance(strategy_type, str):
                # 전략 객체가 직접 전달된 경우
                strategy = strategy_type
            else:
                # 문자열인 경우 기본 TechnicalStrategy 생성
                strategy = TechnicalStrategy(lookback_window=lookback, confidence_threshold=self.confidence_threshold)
            
            # 테스트
            test_returns = self._backtest_period(strategy, test_data, lookback)
            
            # 로깅용 실제 테스트 구간 (룩백 제외)
            actual_test_data = self.data.iloc[test_start_idx:slice_end]
            
            self.results.append({
                'start_idx': i,
                'train_period': f"{train_data.index[0]} - {train_data.index[-1]}",
                'test_period': f"{actual_test_data.index[0]} - {actual_test_data.index[-1]}",
                'total_return': test_returns['total_return'],
                'win_rate': test_returns['win_rate'],
                'max_drawdown': test_returns['max_drawdown'],
                'trade_count': test_returns['trade_count'], # [New] 거래 횟수 추가
            })
        
        return pd.DataFrame(self.results)
    
    def _backtest_period(self, strategy, test_data: pd.DataFrame, lookback: int = 400) -> Dict:
        """테스트 기간 백테스팅"""
        position = 0  # 1: 롱, 0: 플랫, -1: 숏
        entry_price = 0
        entry_cost = 0 # 진입 비용 (수수료 포함)
        quantity = 0   # 보유 수량
        current_stop_loss_price = 0.0 # [New] 현재 포지션의 동적 손절가
        highest_price = 0.0 # [New] 진입 후 최고가 (트레일링 스탑용)
        trades = []
        
        # 백테스팅용 가상 자본 (1억원)
        current_capital = 100000000

        for i in range(lookback, len(test_data)):
            # 과거 데이터 윈도우 준비
            recent_data = test_data.iloc[i-lookback:i]
            # 현재 캔들 (미래 참조 방지를 위해 i번째 캔들의 open이나 close 사용 시 주의)
            # 여기서는 종가 기준으로 신호 발생 가정
            current_price = test_data['close'].iloc[i]
            current_high = test_data['high'].iloc[i]
            current_low = test_data['low'].iloc[i]
            symbol = "TEST_SYM"
            
            # 1. Long 보유 중일 때 SL/TP 체크
            if position == 1:
                # 최고가 갱신 (트레일링 스탑용)
                if current_high > highest_price:
                    highest_price = current_high

                # [New] Break-even (본절 보존) 로직
                # 수익률이 4% 이상 도달하면 손절가를 본절+0.5%로 상향
                if current_high >= entry_price * 1.04:
                    break_even_price = entry_price * 1.005
                    # 현재 설정된 손절가보다 본절가가 높으면 업데이트
                    current_sl = current_stop_loss_price if current_stop_loss_price > 0 else (entry_price * (1 - self.stop_loss) if self.stop_loss else 0)
                    
                    if current_sl < break_even_price:
                        current_stop_loss_price = break_even_price
                        # 동적 손절가가 설정되었으므로 이후 로직에서 이 값이 사용됨

                # Stop Loss (손절)
                # [수정] 전략에서 제안한 동적 손절가가 있으면 우선 사용, 없으면 고정 비율 사용
                sl_price = 0.0
                if current_stop_loss_price > 0:
                    sl_price = current_stop_loss_price
                elif self.stop_loss:
                    sl_price = entry_price * (1 - self.stop_loss)

                if sl_price > 0 and current_low <= sl_price:
                    position = 0
                    # 손절가에 슬리피지 적용하여 청산
                    exit_price = sl_price * (1 - self.slippage)
                    exit_value = (exit_price * quantity) * (1 - self.fee)
                    pnl = exit_value - entry_cost
                    current_capital += pnl # 자본금 갱신
                    trades.append(pnl)
                    continue  # 청산했으므로 이번 캔들의 신호는 무시

                # Take Profit (익절)
                elif self.take_profit and current_high >= entry_price * (1 + self.take_profit):
                    position = 0
                    # 익절가에 슬리피지 적용하여 청산
                    exit_price = entry_price * (1 + self.take_profit) * (1 - self.slippage)
                    exit_value = (exit_price * quantity) * (1 - self.fee)
                    pnl = exit_value - entry_cost
                    current_capital += pnl
                    trades.append(pnl)
                    continue

                # [New] Trailing Stop (트레일링 스탑)
                # 최고가 대비 일정 비율 하락 시 청산
                if self.trailing_stop and current_low <= highest_price * (1 - self.trailing_stop):
                    position = 0
                    exit_price = highest_price * (1 - self.trailing_stop) * (1 - self.slippage)
                    exit_value = (exit_price * quantity) * (1 - self.fee)
                    pnl = exit_value - entry_cost
                    current_capital += pnl
                    trades.append(pnl)
                    continue

            # 2. Short 보유 중일 때 SL/TP 체크 (공매도)
            elif position == -1:
                # Short는 가격이 내려가야 이익 (최저가 갱신 추적 필요하나 여기선 생략)
                
                # Stop Loss (가격 상승 시 손절)
                sl_price = 0.0
                if self.stop_loss:
                    sl_price = entry_price * (1 + self.stop_loss)
                
                if sl_price > 0 and current_high >= sl_price:
                    position = 0
                    exit_price = sl_price * (1 + self.slippage)
                    # Short 수익: (진입가 - 청산가) * 수량
                    exit_value = (entry_price - exit_price) * quantity
                    # 수수료 차감 (진입금액 + 청산금액) * fee
                    fee_cost = (entry_price * quantity + exit_price * quantity) * self.fee
                    pnl = exit_value - fee_cost
                    current_capital += pnl
                    trades.append(pnl)
                    continue

            # 전략에서 신호 생성
            signal = strategy.generate_signal(symbol, recent_data, current_capital=current_capital)
            
            # 매수 신호 (BUY)
            if signal and signal.action == "BUY":
                # Short 포지션 청산
                if position == -1:
                    position = 0
                    exit_price = current_price * (1 + self.slippage)
                    pnl = ((entry_price - exit_price) * quantity) - ((entry_price + exit_price) * quantity * self.fee)
                    current_capital += pnl
                    trades.append(pnl)
                
                # Long 진입 (포지션 없을 때)
                if position == 0:
                    position = 1
                    highest_price = current_price # 최고가 초기화
                    # 매수 체결가: 현재가보다 슬리피지만큼 비싸게 체결된다고 가정
                    entry_price = current_price * (1 + self.slippage)
                    
                    # 수량 결정 (전략 제안 수량 사용)
                    if signal.suggested_quantity > 0:
                        quantity = signal.suggested_quantity
                    else:
                        # 제안 수량이 없으면 자본의 20% 투입 가정
                        quantity = (current_capital * 0.2) / entry_price

                    # [Safety] 자본금 초과 매수 방지 (레버리지 방지: 현물 100% 제한)
                    max_quantity = current_capital / (entry_price * (1 + self.fee))
                    if quantity > max_quantity:
                        quantity = max_quantity

                    # 실제 비용: 체결가 + 수수료
                    entry_cost = (entry_price * quantity) * (1 + self.fee)
                    
                    # [New] 동적 손절가 설정 (전략에서 제안한 값 사용)
                    if signal.suggested_stop_loss:
                        current_stop_loss_price = signal.suggested_stop_loss
                    else:
                        current_stop_loss_price = 0.0
            
            # 매도 신호 (SELL)
            elif signal and signal.action == "SELL":
                # Long 포지션 청산
                if position == 1:
                    position = 0
                    # 매도 체결가: 현재가보다 슬리피지만큼 싸게 체결된다고 가정
                    exit_price = current_price * (1 - self.slippage)
                    # 실제 수령액: 체결가 - 수수료
                    exit_value = (exit_price * quantity) * (1 - self.fee)
                    
                    # 순손익 계산
                    pnl = exit_value - entry_cost
                    current_capital += pnl
                    trades.append(pnl)
                
                # Short 진입 (포지션 없을 때 & 양방향 매매)
                if position == 0:
                    position = -1
                    entry_price = current_price * (1 - self.slippage) # 싸게 팔림
                    
                    # 수량 결정
                    if signal.suggested_quantity > 0:
                        quantity = signal.suggested_quantity
                    else:
                        quantity = (current_capital * 0.2) / entry_price
                    
                    # 비용 계산 (Short는 증거금 필요)
                    entry_cost = (entry_price * quantity) * self.fee # 진입 수수료만 기록
        
        # 통계 계산
        if not trades:
            return {
                'total_return': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'trade_count': 0,
            }
        
        total_return = sum(trades)
        win_count = sum(1 for t in trades if t > 0)
        win_rate = win_count / len(trades) if trades else 0
        
        # 최대낙폭
        # 자산 곡선 생성 (초기 자본 0 가정 시 누적 손익)
        cumulative_returns = np.array([0] + np.cumsum(trades).tolist())
        running_max = np.maximum.accumulate(cumulative_returns)
        max_drawdown = np.min(cumulative_returns - running_max) if len(cumulative_returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'trade_count': len(trades),
        }
