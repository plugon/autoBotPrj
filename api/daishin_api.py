import logging
import json
from typing import Dict, List, Optional
import requests
from .base_api import BaseAPI
import pandas as pd

logger = logging.getLogger(__name__)


class DaishinAPI(BaseAPI):
    """대신증권 API 구현"""
    
    def __init__(self, api_key: str, api_secret: str, account_number: str = ""):
        super().__init__(api_key, api_secret)
        self.base_url = "https://openapi.daishin.co.kr"
        self.account_number = account_number
        self.access_token = None
    
    def connect(self):
        """대신증권 API 연결"""
        try:
            # 토큰 발급
            auth_url = f"{self.base_url}/oauth/authorize"
            payload = {
                "appkey": self.api_key,
                "appsecret": self.api_secret,
                "grant_type": "client_credentials",
            }
            response = requests.post(auth_url, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get("access_token")
                logger.info("대신증권 API 연결 성공")
            else:
                logger.error(f"대신증권 API 연결 실패: {response.status_code}")
        except Exception as e:
            logger.error(f"대신증권 API 연결 오류: {e}")
    
    def disconnect(self):
        """대신증권 API 연결 종료"""
        self.access_token = None
        logger.info("대신증권 API 연결 종료")
    
    def _get_headers(self):
        """요청 헤더"""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
    
    def get_balance(self) -> Dict:
        """계좌 잔액 조회"""
        try:
            url = f"{self.base_url}/v1/account/balance"
            params = {"account_number": self.account_number}
            response = requests.get(url, params=params, headers=self._get_headers())
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "total_balance": data.get("total_balance", 0),
                    "available_cash": data.get("available_cash", 0),
                    "positions": data.get("positions", [])
                }
            else:
                logger.error(f"잔액 조회 실패: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"잔액 조회 오류: {e}")
            return {}
    
    def get_price(self, symbol: str) -> float:
        """현재가 조회"""
        try:
            url = f"{self.base_url}/v1/market/price"
            params = {"code": symbol}
            response = requests.get(url, params=params, headers=self._get_headers())
            
            if response.status_code == 200:
                data = response.json()
                return float(data.get("price", 0))
            else:
                logger.warning(f"{symbol} 현재가 조회 실패")
                return 0.0
        except Exception as e:
            logger.error(f"현재가 조회 오류: {e}")
            return 0.0
    
    def get_ohlcv(self, symbol: str, timeframe: str = "1d") -> pd.DataFrame:
        """OHLCV 데이터 조회"""
        try:
            url = f"{self.base_url}/v1/market/ohlcv"
            params = {
                "code": symbol,
                "period": "day" if timeframe == "1d" else "week",
                "limit": 200
            }
            response = requests.get(url, params=params, headers=self._get_headers())
            
            if response.status_code == 200:
                data = response.json()
                records = data.get("data", [])
                
                df = pd.DataFrame(records)
                if not df.empty:
                    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    return df
                return pd.DataFrame()
            else:
                logger.warning(f"{symbol} OHLCV 데이터 조회 실패")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"OHLCV 데이터 조회 오류: {e}")
            return pd.DataFrame()
    
    def buy(self, symbol: str, quantity: float, price: Optional[float] = None) -> Dict:
        """매수 주문"""
        try:
            url = f"{self.base_url}/v1/order/buy"
            payload = {
                "account_number": self.account_number,
                "code": symbol,
                "quantity": int(quantity),
                "price": int(price) if price else 0,
                "order_type": "limit" if price else "market",
            }
            response = requests.post(url, json=payload, headers=self._get_headers())
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"매수 주문 성공: {symbol} {quantity}주")
                return result
            else:
                logger.error(f"매수 주문 실패: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"매수 주문 오류: {e}")
            return {}
    
    def sell(self, symbol: str, quantity: float, price: Optional[float] = None) -> Dict:
        """매도 주문"""
        try:
            url = f"{self.base_url}/v1/order/sell"
            payload = {
                "account_number": self.account_number,
                "code": symbol,
                "quantity": int(quantity),
                "price": int(price) if price else 0,
                "order_type": "limit" if price else "market",
            }
            response = requests.post(url, json=payload, headers=self._get_headers())
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"매도 주문 성공: {symbol} {quantity}주")
                return result
            else:
                logger.error(f"매도 주문 실패: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"매도 주문 오류: {e}")
            return {}
    
    def cancel_order(self, order_id: str) -> bool:
        """주문 취소"""
        try:
            url = f"{self.base_url}/v1/order/cancel"
            payload = {"order_id": order_id}
            response = requests.post(url, json=payload, headers=self._get_headers())
            
            if response.status_code == 200:
                logger.info(f"주문 취소 성공: {order_id}")
                return True
            else:
                logger.error(f"주문 취소 실패: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"주문 취소 오류: {e}")
            return False
    
    def get_open_orders(self) -> List[Dict]:
        """미체결 주문 조회"""
        try:
            url = f"{self.base_url}/v1/order/open"
            params = {"account_number": self.account_number}
            response = requests.get(url, params=params, headers=self._get_headers())
            
            if response.status_code == 200:
                data = response.json()
                return data.get("orders", [])
            else:
                logger.warning("미체결 주문 조회 실패")
                return []
        except Exception as e:
            logger.error(f"미체결 주문 조회 오류: {e}")
            return []

    def get_positions(self) -> List[Dict]:
        """보유 포지션 조회"""
        try:
            balance_info = self.get_balance()
            positions = []
            for item in balance_info.get("positions", []):
                positions.append({
                    "symbol": item.get("code", ""),
                    "quantity": float(item.get("quantity", 0)),
                    "entry_price": float(item.get("price", 0))
                })
            return positions
        except Exception as e:
            logger.error(f"포지션 조회 오류: {e}")
            return []
