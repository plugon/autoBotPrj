import logging
import pandas as pd
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
        self.pyramiding_info: Dict[str, Dict] = {} # {symbol: {'count': 0, 'last_entry_price': 0.0}}
        self.metadata: Dict = {}  # 전략 정보 등 메타데이터 저장
    
    def add_position(self, symbol: str, quantity: float, entry_price: float, fee_rate: float = 0.0) -> bool:
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
            else:
                # 기존 포지션에 추가
                total_quantity = self.positions[symbol] + quantity
                avg_price = (
                    (self.positions[symbol] * self.entry_prices[symbol] + 
                     quantity * entry_price) / total_quantity
                )
                self.positions[symbol] = total_quantity
                self.entry_prices[symbol] = avg_price
            
            # 자본 차감
            self.current_capital -= total_cost
            
            logger.info(
                f"포지션 추가: {symbol} {quantity}주 @ {entry_price}"
            )
            return True
        
        except Exception as e:
            logger.error(f"포지션 추가 오류: {e}")
            return False
    
    def sync_position(self, symbol: str, quantity: float, entry_price: float):
        """외부 포지션 동기화 (자본 차감 없이 포지션만 등록)"""
        self.positions[symbol] = quantity
        self.entry_prices[symbol] = entry_price
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
        if symbol in self.pyramiding_info:
            del self.pyramiding_info[symbol]

    def close_position(self, symbol: str, quantity: float, 
                      exit_price: float, fee_rate: float = 0.0) -> bool:
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
                'pnl_percent': pnl_percent
            })
            
            logger.info(
                f"포지션 종료: {symbol} {quantity}주 @ {exit_price} "
                f"PnL: {pnl:.2f} ({pnl_percent:.2f}%)"
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
    
    def get_total_value(self, prices: Dict[str, float]) -> float:
        """포트폴리오 총 가치"""
        total = self.current_capital
        for symbol, quantity in self.positions.items():
            if symbol in prices:
                total += quantity * prices[symbol]
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
    
    def get_statistics(self, prices: Dict[str, float]) -> Dict:
        """포트폴리오 통계"""
        total_value = self.get_total_value(prices)
        total_realized_pnl = sum(
            trade['pnl'] for trade in self.trade_history
        )
        
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
        }

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
                "pyramiding_info": self.pyramiding_info,
                "metadata": self.metadata,
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
                self.pyramiding_info = state.get("pyramiding_info", {})
                self.metadata = state.get("metadata", {})
                self.trade_history = []
                for t in state["trade_history"]:
                    t['timestamp'] = datetime.fromisoformat(t['timestamp'])
                    self.trade_history.append(t)
            logger.info(f"포트폴리오 로드 완료: {filepath}")
        except Exception as e:
            logger.error(f"포트폴리오 로드 오류: {e}")
