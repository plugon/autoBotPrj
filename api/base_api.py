import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class BaseAPI(ABC):
    """API 기본 클래스"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = None
    
    @abstractmethod
    def connect(self):
        """API 연결"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """API 연결 종료"""
        pass
    
    @abstractmethod
    def get_balance(self) -> Dict:
        """계좌 잔액 조회"""
        pass
    
    @abstractmethod
    def get_price(self, symbol: str) -> float:
        """현재가 조회"""
        pass
    
    @abstractmethod
    def get_ohlcv(self, symbol: str, timeframe: str = "1d") -> pd.DataFrame:
        """OHLCV 데이터 조회"""
        pass
    
    @abstractmethod
    def buy(self, symbol: str, quantity: float, price: Optional[float] = None, **kwargs) -> Dict:
        """매수 주문"""
        pass
    
    @abstractmethod
    def sell(self, symbol: str, quantity: float, price: Optional[float] = None) -> Dict:
        """매도 주문"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """주문 취소"""
        pass
    
    @abstractmethod
    def get_open_orders(self) -> List[Dict]:
        """미체결 주문 조회"""
        pass

    def get_positions(self) -> List[Dict]:
        """보유 포지션 조회"""
        return []
