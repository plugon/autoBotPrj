import logging
import json
from typing import Dict, List, Optional
import requests
from .base_api import BaseAPI
import pandas as pd

logger = logging.getLogger(__name__)


class KiwoomAPI(BaseAPI):
    """키움증권 API 구현 (Open API)"""
    
    def __init__(self, api_key: str, api_secret: str, account_number: str = ""):
        super().__init__(api_key, api_secret)
        self.base_url = "https://openapi.kiwoom.com/h0stEx"
        self.account_number = account_number
        self.access_token = None
        self.token_expires_at = None
    
    def connect(self):
        """키움증권 API 연결"""
        try:
            # OAuth 토큰 발급
            auth_url = f"{self.base_url}/oauth/authorize"
            payload = {
                "client_id": self.api_key,
                "client_secret": self.api_secret,
                "grant_type": "client_credentials",
            }
            response = requests.post(auth_url, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get("access_token")
                logger.info("키움증권 API 연결 성공")
            else:
                logger.error(f"키움증권 API 연결 실패: {response.status_code}")
        except Exception as e:
            logger.error(f"키움증권 API 연결 오류: {e}")
    
    def disconnect(self):
        """키움증권 API 연결 종료"""
        self.access_token = None
        logger.info("키움증권 API 연결 종료")
    
    def _get_headers(self):
        """요청 헤더"""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
    
    def get_balance(self) -> Dict:
        """계좌 잔액 조회"""
        try:
            url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
            params = {
                "CANO": self.account_number[:8],  # 계좌번호 앞 8자리
                "ACNT_PRDT_CD": self.account_number[8:10],  # 계좌번호 뒤 2자리
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "N",
                "INQR_DVSN": "02",
                "UNPR_DCD": "01",
                "FUND_STTL_ICLD_YN": "Y",
                "FNCG_AMT_AUTO_CALC_YN": "Y",
                "PRCS_DVSN": "00",
                "CTX_AREA_FK": "",
                "CTX_AREA_NK": "",
            }
            response = requests.get(
                url, 
                params=params,
                headers=self._get_headers()
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "total_balance": data.get("output1", {}).get("sszc_grp_sfdr_5", 0),
                    "available_cash": data.get("output1", {}).get("dnca_tot_amt", 0),
                    "positions": data.get("output2", [])
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
            url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
            params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol}
            response = requests.get(url, params=params, headers=self._get_headers())
            
            if response.status_code == 200:
                data = response.json()
                return float(data.get("output", {}).get("stck_prpr", 0))
            else:
                logger.warning(f"{symbol} 현재가 조회 실패")
                return 0.0
        except Exception as e:
            logger.error(f"현재가 조회 오류: {e}")
            return 0.0
    
    def get_ohlcv(self, symbol: str, timeframe: str = "1d") -> pd.DataFrame:
        """OHLCV 데이터 조회"""
        try:
            url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
            params = {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": symbol,
                "FID_INPUT_DATE_1": "20200101",
                "FID_INPUT_DATE_2": "20260126",
                "FID_PERIOD_DIV_CODE": "D" if timeframe == "1d" else "W",
                "FID_ORG_ADJ_PRC": "0",
                "FID_PWR_RSS_CTR_CODE": "0",
            }
            response = requests.get(url, params=params, headers=self._get_headers())
            
            if response.status_code == 200:
                data = response.json()
                records = data.get("output2", [])
                
                df = pd.DataFrame(records)
                if not df.empty:
                    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d')
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
            url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
            payload = {
                "CANO": self.account_number[:8],
                "ACNT_PRDT_CD": self.account_number[8:10],
                "PDNO": symbol,
                "ORD_QTY": str(int(quantity)),
                "ORD_UNPR": str(int(price)) if price else "0",
                "ORD_DVSN": "00" if price else "01",  # 지정가 또는 시장가
                "CTAC_TLNO": "",
                "CTAC_TLNO2": "",
                "ALGO_NO": "",
                "ALGO_ORD_MGMT_DVSN_CD": "00",
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
            url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
            payload = {
                "CANO": self.account_number[:8],
                "ACNT_PRDT_CD": self.account_number[8:10],
                "PDNO": symbol,
                "ORD_QTY": str(int(quantity)),
                "ORD_UNPR": str(int(price)) if price else "0",
                "ORD_DVSN": "00" if price else "01",  # 지정가 또는 시장가
                "SLL_TYPE": "01",
                "CTAC_TLNO": "",
                "CTAC_TLNO2": "",
                "ALGO_NO": "",
                "ALGO_ORD_MGMT_DVSN_CD": "00",
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
            url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-rvsn"
            payload = {
                "CANO": self.account_number[:8],
                "ACNT_PRDT_CD": self.account_number[8:10],
                "KRX_FWDG_ORD_ORGNO": "",
                "ORGN_ODNO": order_id,
                "ORD_DVSN": "00",
            }
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
            url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-psbl-order"
            params = {
                "CANO": self.account_number[:8],
                "ACNT_PRDT_CD": self.account_number[8:10],
                "ORD_STRT_DT": "20200101",
                "ORD_END_DT": "20260126",
                "ORD_GBN": "",
                "PAGE_SIZE": "100",
                "CTX_AREA_FK": "",
                "CTX_AREA_NK": "",
            }
            response = requests.get(url, params=params, headers=self._get_headers())
            
            if response.status_code == 200:
                data = response.json()
                return data.get("output", [])
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
                    "symbol": item.get("pdno", ""),
                    "quantity": float(item.get("hldg_qty", 0)),
                    "entry_price": float(item.get("pchs_avg_pric", 0))
                })
            return positions
        except Exception as e:
            logger.error(f"포지션 조회 오류: {e}")
            return []
