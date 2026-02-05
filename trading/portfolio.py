import logging
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


class Portfolio:
    """포트폴리오 관리"""
    
    def __init__(self, initial_capital: float, max_position_size: float = 0.3):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.positions: Dict[str, float] = {}  # {symbol: quantity}
        self.entry_prices: Dict[str, float] = {}  # {symbol: entry_price}
        self.trade_history: List[Dict] = []
        self.atr_values: Dict[str, float] = {} # [New] 진입 시점 ATR 저장 {symbol: atr}
        self.pyramiding_info: Dict[str, Dict] = {} # {symbol: {'count': 0, 'last_entry_price': 0.0}}
        self.metadata: Dict = {}  # 전략 정보 등 메타데이터 저장
        self.daily_history: List[Dict] = [] # [{'date': 'YYYY-MM-DD', 'total_value': float}]
        self.peak_equity = initial_capital # [New] 실시간 고점 추적용
    
    def add_position(self, symbol: str, quantity: float, entry_price: float, fee_rate: float = 0.0, atr_value: float = 0.0, reason: str = "") -> bool:
        """포지션 추가"""
        try:
            cost = quantity * entry_price
            transaction_fee = cost * fee_rate
            total_cost = cost + transaction_fee
            
            # 포지션 크기 확인
            if total_cost > self.current_capital * self.max_position_size:
                logger.warning(
                    f"포지션 크기 초과: {symbol} "
                    f"({cost} > {self.current_capital * self.max_position_size})"
                )
                return False
            
            # 포지션 추가
            if symbol not in self.positions:
                self.positions[symbol] = quantity
                self.entry_prices[symbol] = entry_price
                if atr_value > 0:
                    self.atr_values[symbol] = atr_value
            else:
                # 기존 포지션에 추가
                total_quantity = self.positions[symbol] + quantity
                avg_price = (
                    (self.positions[symbol] * self.entry_prices[symbol] + 
                     quantity * entry_price) / total_quantity
                )
                self.positions[symbol] = total_quantity
                self.entry_prices[symbol] = avg_price
                # [New] 추가 매수 시 ATR 업데이트 (최신 값 반영)
                if atr_value > 0:
                    self.atr_values[symbol] = atr_value
            
            # 자본 차감
            self.current_capital -= total_cost
            
            logger.info(
                f"포지션 추가: {symbol} {quantity}주 @ {entry_price} (사유: {reason})"
            )
            return True
        
        except Exception as e:
            logger.error(f"포지션 추가 오류: {e}")
            return False
    
    def sync_position(self, symbol: str, quantity: float, entry_price: float, atr_value: float = 0.0):
        """외부 포지션 동기화 (자본 차감 없이 포지션만 등록)"""
        self.positions[symbol] = quantity
        self.entry_prices[symbol] = entry_price
        if atr_value > 0:
            self.atr_values[symbol] = atr_value
        # 동기화 시 피라미딩 정보가 없으면 초기화
        if symbol not in self.pyramiding_info:
            self.pyramiding_info[symbol] = {'count': 0, 'last_entry_price': entry_price}

    def update_pyramiding_state(self, symbol: str, price: float, is_reset: bool = False):
        """피라미딩 상태 업데이트"""
        if is_reset or symbol not in self.pyramiding_info:
            self.pyramiding_info[symbol] = {'count': 0, 'last_entry_price': price}
        else:
            self.pyramiding_info[symbol]['count'] += 1
            self.pyramiding_info[symbol]['last_entry_price'] = price

    def remove_position(self, symbol: str):
        """포지션 강제 제거 (외부 매도 시 사용)"""
        if symbol in self.positions:
            del self.positions[symbol]
        if symbol in self.entry_prices:
            del self.entry_prices[symbol]
        if symbol in self.atr_values:
            del self.atr_values[symbol]
        if symbol in self.pyramiding_info:
            del self.pyramiding_info[symbol]

    def close_position(self, symbol: str, quantity: float, 
                      exit_price: float, fee_rate: float = 0.0, reason: str = "") -> bool:
        """포지션 종료"""
        try:
            if symbol not in self.positions or self.positions[symbol] < quantity:
                logger.warning(f"포지션 부족: {symbol}")
                return False
            
            # 수익/손실 계산 (수수료 반영)
            entry_price_val = self.entry_prices[symbol]
            transaction_fee = (quantity * exit_price) * fee_rate
            pnl = ((exit_price - entry_price_val) * quantity) - transaction_fee
            
            # 자본에 추가
            self.current_capital += (quantity * exit_price) - transaction_fee
            
            # 포지션 감소
            self.positions[symbol] -= quantity
            
            # 포지션이 0이면 제거
            if self.positions[symbol] == 0:
                del self.positions[symbol]
                del self.entry_prices[symbol]
                if symbol in self.atr_values:
                    del self.atr_values[symbol]
                if symbol in self.pyramiding_info:
                    del self.pyramiding_info[symbol]
            
            # 수익률 계산 (0 나누기 방지)
            pnl_percent = (pnl / (entry_price_val * quantity)) * 100 if entry_price_val > 0 else 0.0
            
            # 거래 기록
            self.trade_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'type': 'sell',
                'quantity': quantity,
                'entry_price': entry_price_val,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'reason': reason # [New] 사유 저장
            })
            
            logger.info(
                f"포지션 종료: {symbol} {quantity}주 @ {exit_price} "
                f"PnL: {pnl:.2f} ({pnl_percent:.2f}%) | 사유: {reason}"
            )
            return True
        
        except Exception as e:
            logger.error(f"포지션 종료 오류: {e}")
            return False
    
    def get_position_value(self, symbol: str, current_price: float) -> float:
        """포지션 현재 가치"""
        if symbol not in self.positions:
            return 0.0
        return self.positions[symbol] * current_price
    
    def get_total_value(self, prices: Dict[str, float], use_entry_price_fallback: bool = False) -> float:
        """포트폴리오 총 가치"""
        total = self.current_capital
        for symbol, quantity in self.positions.items():
            price = prices.get(symbol, 0.0)
            if price <= 0 and use_entry_price_fallback:
                price = self.entry_prices.get(symbol, 0.0)
            
            if price > 0:
                total += quantity * price
        return total
    
    def get_unrealized_pnl(self, symbol: str, current_price: float) -> float:
        """미실현 손익"""
        if symbol not in self.positions:
            return 0.0
        return (current_price - self.entry_prices[symbol]) * self.positions[symbol]
    
    def get_unrealized_pnl_percent(self, symbol: str, current_price: float) -> float:
        """미실현 손익률"""
        if symbol not in self.positions:
            return 0.0
        pnl = self.get_unrealized_pnl(symbol, current_price)
        cost = self.entry_prices[symbol] * self.positions[symbol]
        return (pnl / cost) * 100 if cost > 0 else 0.0
    
    def get_portfolio_summary(self, prices: Dict[str, float]) -> pd.DataFrame:
        """포트폴리오 요약"""
        summary = []
        
        for symbol, quantity in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                entry_price = self.entry_prices[symbol]
                unrealized_pnl = self.get_unrealized_pnl(symbol, current_price)
                unrealized_pnl_percent = self.get_unrealized_pnl_percent(
                    symbol, current_price
                )
                position_value = self.get_position_value(symbol, current_price)
                
                summary.append({
                    'symbol': symbol,
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'position_value': position_value,
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_percent': unrealized_pnl_percent,
                })
        
        return pd.DataFrame(summary)
    
    def update_daily_status(self, prices: Dict[str, float]):
        """일별 자산 상태 업데이트 (일 1회 기록)"""
        total_value = self.get_total_value(prices, use_entry_price_fallback=True)
        today = datetime.now().strftime("%Y-%m-%d")
        
        # 이미 오늘 데이터가 있으면 업데이트
        for record in self.daily_history:
            if record['date'] == today:
                record['total_value'] = total_value
                return

        # 없으면 추가
        self.daily_history.append({
            'date': today,
            'total_value': total_value
        })

    def calculate_mdd(self, current_total_value: float = None) -> float:
        """MDD(Maximum Drawdown) 계산"""
        values = [d['total_value'] for d in self.daily_history]
        
        # 현재 평가금액도 포함하여 계산 (실시간 반영)
        if current_total_value is not None:
            values.append(current_total_value)
            
        if not values:
            return 0.0
            
        peak = np.maximum.accumulate(values)
        # 0으로 나누기 방지
        with np.errstate(divide='ignore', invalid='ignore'):
            drawdown = (peak - values) / peak
            drawdown = np.nan_to_num(drawdown) # NaN/Inf 처리
        return np.max(drawdown) * 100 # % 단위

    def analyze_asset_allocation(self, prices: Dict[str, float]) -> Dict:
        """자산 배분 현황 분석 (현금 vs 종목별 비중)"""
        total_value = self.get_total_value(prices, use_entry_price_fallback=True)
        
        allocation = {
            'cash': self.current_capital,
            'assets': {},
            'ratios': {
                'cash': (self.current_capital / total_value * 100) if total_value > 0 else 0.0
            }
        }
        
        for symbol, quantity in self.positions.items():
            price = prices.get(symbol, 0)
            if price <= 0:
                price = self.entry_prices.get(symbol, 0)
                
            val = quantity * price
            allocation['assets'][symbol] = val
            allocation['ratios'][symbol] = (val / total_value * 100) if total_value > 0 else 0.0
            
        return allocation

    def get_statistics(self, prices: Dict[str, float], use_entry_price_fallback: bool = False) -> Dict:
        """포트폴리오 통계"""
        total_value = self.get_total_value(prices, use_entry_price_fallback)
        
        # [New] 고점 갱신
        if total_value > self.peak_equity:
            self.peak_equity = total_value
            
        total_realized_pnl = sum(
            trade['pnl'] for trade in self.trade_history
        )
        mdd = self.calculate_mdd(current_total_value=total_value)
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_value': total_value,
            'total_profit_loss': total_value - self.initial_capital,
            'total_profit_loss_percent': (
                (total_value - self.initial_capital) / self.initial_capital * 100 
                if self.initial_capital > 0 else 0.0
            ),
            'realized_pnl': total_realized_pnl,
            'num_positions': len(self.positions),
            'num_trades': len(self.trade_history),
            'mdd': mdd,
            'peak_equity': self.peak_equity,
        }

    def get_daily_realized_pnl(self, target_date: datetime = None) -> float:
        """특정 날짜(기본값: 오늘)의 실현 손익 합계"""
        if target_date is None:
            target_date = datetime.now()
        
        target_date_str = target_date.strftime("%Y-%m-%d")
        daily_pnl = 0.0
        
        for trade in self.trade_history:
            if trade['type'] == 'sell':
                # trade['timestamp']는 datetime 객체임 (load_state에서 변환됨)
                trade_ts = trade['timestamp']
                if trade_ts.strftime("%Y-%m-%d") == target_date_str:
                    daily_pnl += trade.get('pnl', 0.0)
                    
        return daily_pnl

    def save_state(self, filepath: str):
        """포트폴리오 상태 저장"""
        try:
            # 디렉토리 생성
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            state = {
                "initial_capital": self.initial_capital,
                "current_capital": self.current_capital,
                "positions": self.positions,
                "entry_prices": self.entry_prices,
                "atr_values": self.atr_values, # [New] 저장
                "pyramiding_info": self.pyramiding_info,
                "metadata": self.metadata,
                "daily_history": self.daily_history,
                "peak_equity": self.peak_equity,
                "trade_history": [
                    {**t, 'timestamp': t['timestamp'].isoformat()} 
                    for t in self.trade_history
                ]
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=4, ensure_ascii=False)
            # logger.debug(f"포트폴리오 저장 완료: {filepath}")
        except Exception as e:
            logger.error(f"포트폴리오 저장 오류: {e}")

    def load_state(self, filepath: str):
        """포트폴리오 상태 로드"""
        if not os.path.exists(filepath):
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
                self.initial_capital = state["initial_capital"]
                self.current_capital = state["current_capital"]
                self.positions = state["positions"]
                self.entry_prices = state["entry_prices"]
                self.atr_values = state.get("atr_values", {}) # [New] 로드
                self.pyramiding_info = state.get("pyramiding_info", {})
                self.metadata = state.get("metadata", {})
                self.daily_history = state.get("daily_history", [])
                self.peak_equity = state.get("peak_equity", self.initial_capital)
                self.trade_history = []
                for t in state["trade_history"]:
                    t['timestamp'] = datetime.fromisoformat(t['timestamp'])
                    self.trade_history.append(t)
            logger.info(f"포트폴리오 로드 완료: {filepath}")
        except Exception as e:
            logger.error(f"포트폴리오 로드 오류: {e}")
