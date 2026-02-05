#!/usr/bin/env python
"""
자동매매 프로그램 메인 모듈
한국주식 + 암호화폐 자동매매 봇
"""

import logging
import time
import os
import sys
import subprocess

# [Fix] TensorFlow 로그 노이즈 제거 (oneDNN 최적화 메시지 및 INFO 로그 숨김)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import requests
import threading
import pandas as pd
import numpy as np
import multiprocessing
import psutil
import shutil
import joblib
import warnings
import concurrent.futures
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from ta.volatility import AverageTrueRange

from config.settings import TRADING_CONFIG, ML_CONFIG, MONITORING_CONFIG, VOLUME_CONFIG
from api.shinhan_api import ShinhanAPI
from api.kiwoom_api import KiwoomAPI
from api.daishin_api import DaishinAPI
from api.crypto_api import UpbitAPI, BinanceAPI
from models.ml_model import MLPredictor
from trading.strategy import MLStrategy, TechnicalStrategy
from trading.strategy_v2 import HeikinAshiStrategy
from trading.turtle_bollinger_strategy import TurtleBollingerStrategy
from trading.agile_strategy import AgileStrategy
from trading.ma_trend_strategy import MATrendStrategy
from trading.volume_trend_strategy import VolumeTrendStrategy
from trading.early_bird_strategy import EarlyBirdStrategy
from utils.report_manager import ReportManager
from trading.portfolio import Portfolio
from trading.risk_manager import RiskManager
from utils.backtesting import WalkForwardAnalyzer
from utils.logger import setup_logger

# 로거 설정
# .env에서 로그 레벨 읽기 (기본값: INFO)
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logger = setup_logger("trading_bot", log_level)

# 라이브러리 로그 노이즈 제거 (DEBUG 모드 시 너무 많은 로그 방지)
logging.getLogger('ccxt').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# [Request] sklearn 관련 불필요한 경고 무시 (UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# [Fix] Keras/TensorFlow 불필요한 경고 무시 (input_shape 관련)
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

# [Request 3] 병렬 처리를 위한 독립 함수 (Pickling 가능해야 함)
def _train_model_task(symbol, data, ml_config, api_name, models_dir):
    """개별 종목 모델 학습 및 전진분석 태스크 (병렬 처리용)"""
    try:
        # [Request 4] 지표 선행 계산 (Caching 효과)
        # 데이터프레임 전체에 대해 지표를 한 번만 계산하여 컬럼에 추가
        import ta
        import os
        import joblib
        
        # RSI
        data['RSI'] = ta.momentum.rsi(data['close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(data['close'])
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Hist'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(data['close'], window=20, window_dev=2)
        data['BB_Upper'] = bb.bollinger_hband()
        data['BB_Lower'] = bb.bollinger_lband()
        data['BB_Middle'] = bb.bollinger_mavg()
        
        # 데이터가 충분한지 재확인
        if len(data) <= ml_config["lookback_window"]:
            return (symbol, False, 0)

        # 전진분석 검증
        analyzer = WalkForwardAnalyzer(
            data, 
            train_period=200, 
            test_period=50, 
            fee=0.0005
        )
        results = analyzer.run(strategy_type="ml")
        total_return = results['total_return'].sum()
        
        # 모델 학습 및 저장
        if total_return > -10000:
            model = MLPredictor(ml_config["lookback_window"], ml_config["model_type"])
            model.train(data, epochs=5, batch_size=64) # [Request 1] 파라미터 최적화
            
            # [Fix] Worker 프로세스에서 직접 저장 (Keras 모델 Pickling 오류 방지)
            safe_symbol = symbol.replace("/", "_")
            model_path = os.path.join(models_dir, f"{safe_symbol}_{api_name}_model.pkl")
            
            compress_level = 3
            if model.model_type == "lstm":
                h5_path = model_path.replace(".pkl", ".h5")
                model.model.save(h5_path)
                joblib.dump(model.scaler, model_path.replace(".pkl", "_scaler.pkl"), compress=compress_level)
                
                # [New] ONNX 변환 및 저장
                try:
                    import tensorflow as tf
                    import tf2onnx
                    # 모델 입력 형상 자동 감지
                    spec = (tf.TensorSpec(model.model.input_shape, tf.float32, name="input"),)
                    onnx_path = model_path.replace(".pkl", ".onnx")
                    model_proto, _ = tf2onnx.convert.from_keras(model.model, input_signature=spec, opset=13)
                    with open(onnx_path, "wb") as f:
                        f.write(model_proto.SerializeToString())
                except Exception:
                    pass
            else:
                joblib.dump(model.model, model_path, compress=compress_level)
                joblib.dump(model.scaler, model_path.replace(".pkl", "_scaler.pkl"), compress=compress_level)

            return (symbol, True, total_return)
        else:
            return (symbol, False, total_return)
            
    except Exception as e:
        return (symbol, e, 0)

class AutoTradingBot:
    """자동매매 봇 메인 클래스"""
    
    def __init__(self):
        logger.info("=" * 60)
        logger.info("자동매매 봇 초기화 시작")
        logger.info("=" * 60)
        
        logger.info("1. API 객체 및 변수 초기화")
        # API 초기화 (한국 증권사)
        self.shinhan_api = None
        self.kiwoom_api = None
        self.daishin_api = None
        
        # 암호화폐 API
        self.crypto_api = None
        
        # 시스템 리소스 모니터링용
        self.process = psutil.Process(os.getpid())
        self.process.cpu_percent(interval=None) # 초기 호출 (기준점 설정)
        
        logger.info("2. 포트폴리오 데이터 로드")
        # 포트폴리오 초기화
        self.stock_portfolio = Portfolio(
            TRADING_CONFIG["korean_stocks"]["initial_capital"],
            TRADING_CONFIG["korean_stocks"]["max_position_size"]
        )
        self.stock_portfolio.load_state("data/stock_portfolio.json")
        
        self.crypto_portfolio = Portfolio(
            TRADING_CONFIG["crypto"]["initial_capital"],
            TRADING_CONFIG["crypto"]["max_position_size"]
        )
        self.crypto_portfolio.load_state("data/crypto_portfolio.json")
        
        # [New] 바이낸스 현물 포트폴리오
        self.binance_spot_portfolio = Portfolio(
            TRADING_CONFIG["binance_spot"]["initial_capital"],
            TRADING_CONFIG["binance_spot"]["max_position_size"]
        )
        self.binance_spot_portfolio.load_state("data/binance_spot_portfolio.json")

        # [New] 바이낸스 선물 포트폴리오
        self.binance_futures_portfolio = Portfolio(
            TRADING_CONFIG["binance_futures"]["initial_capital"],
            TRADING_CONFIG["binance_futures"]["max_position_size"]
        )
        self.binance_futures_portfolio.load_state("data/binance_futures_portfolio.json")
        
        # [New] GPU 가속 설정 (LSTM 모델용)
        self._setup_gpu()
        
        logger.info("3. 머신러닝 모델 초기화")
        # 머신러닝 모델 초기화
        self.ml_model = MLPredictor(
            ML_CONFIG["lookback_window"],
            ML_CONFIG["model_type"]
        )
        
        logger.info("4. 거래 전략 설정")
        # 거래 전략 초기화
        self.ml_strategy = MLStrategy(self.ml_model, ML_CONFIG["lookback_window"])
        self.technical_strategy = TechnicalStrategy(ML_CONFIG["lookback_window"])

        # [Refactor] 전략 객체 초기화 (메서드로 분리하여 재사용)
        self._initialize_strategies()

        logger.info("5. 리스크 관리자 및 스케줄러 설정")
        # 위험 관리 초기화
        self.stock_risk_manager = RiskManager(
            # 주식은 기본값 사용 (ATR 정보가 없을 경우 비상 손절 작동)
        )
        self.crypto_risk_manager = RiskManager(
            take_profit_percent=TRADING_CONFIG["crypto"]["take_profit_percent"],
            atr_multiplier=TRADING_CONFIG["crypto"].get("atr_multiplier", 2.0),
            trailing_stop_percent=TRADING_CONFIG["crypto"].get("trailing_stop_percent", 0.02)
        )
        
        # [New] 바이낸스 현물 리스크 관리자
        self.binance_spot_risk_manager = RiskManager(
            take_profit_percent=TRADING_CONFIG["binance_spot"]["take_profit_percent"],
            atr_multiplier=TRADING_CONFIG["binance_spot"].get("atr_multiplier", 2.0),
            trailing_stop_percent=TRADING_CONFIG["binance_spot"].get("trailing_stop_percent", 0.02)
        )
        
        # [New] 바이낸스 선물 리스크 관리자
        self.binance_futures_risk_manager = RiskManager(
            take_profit_percent=TRADING_CONFIG["binance_futures"]["take_profit_percent"],
            atr_multiplier=TRADING_CONFIG["binance_futures"].get("atr_multiplier", 2.0),
            trailing_stop_percent=TRADING_CONFIG["binance_futures"].get("trailing_stop_percent", 0.02)
        )

        # 스케줄러
        self.scheduler = BackgroundScheduler()
        self.trade_lock = threading.Lock()  # 거래 중복 실행 방지 락
        
        # 거래량 기반 종목 자동 선택
        self.last_volume_update = 0
        self.crypto_symbols = TRADING_CONFIG["crypto"]["symbols"].copy()
        self.binance_spot_symbols = TRADING_CONFIG["binance_spot"]["symbols"].copy()
        self.binance_futures_symbols = TRADING_CONFIG["binance_futures"]["symbols"].copy()
        self.oco_monitoring_symbols = set() # [New] OCO 주문으로 서버 관리 중인 종목
        self.volatility_monitor = {} # [New] 급등락 모니터링용 데이터
        
        # [Request 3] 봇 웜업 상태 (초기 데이터 수집 안정화)
        self.is_ready = False
        self.warmup_counter = 0
        
        # OHLCV 데이터 캐시 (API 호출 최소화)
        self.ohlcv_cache = {}
        self.last_ohlcv_fetch = {}
        self.fetch_interval = 180  # 3분 (REST API 호출 빈도 대폭 감소)
        self.last_log_time = {} # [New] 로그 스로틀링용 타임스탬프 저장
        
        # .env Hot Reload용 타임스탬프
        self.last_env_mtime = 0
        self.check_env_updates() # 초기 로드
        
        # 현재 적용된 전략 프리셋 로깅
        from config.settings import selected_strategy_name
        logger.info(f"📈 적용된 암호화폐 거래 전략: '{selected_strategy_name}'")
        
        # 포트폴리오에 현재 전략 정보 업데이트 (대시보드 표시용)
        self.crypto_portfolio.metadata.update({
            "strategy": selected_strategy_name,
            "timeframe": TRADING_CONFIG["crypto"]["timeframe"]
        })
        self.crypto_portfolio.save_state("data/crypto_portfolio.json")
        
        # 동적 설정 로드 (백테스팅 결과 반영)
        self.load_dynamic_config()
        
        # 리포트 매니저 (API 초기화 후 사용 가능하므로 여기선 None)
        self.report_manager = None

        logger.info("자동매매 봇 초기화 완료")
    
    def _setup_gpu(self):
        """TensorFlow GPU 가속 설정"""
        try:
            # TensorFlow 로그 레벨 조정 (불필요한 로그 숨김)
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # 메모리 증가 허용 (VRAM 전체 할당 방지)
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"🚀 GPU 가속 활성화됨: {len(gpus)}개의 GPU 감지")
                except RuntimeError as e:
                    logger.warning(f"⚠️ GPU 설정 실패 (이미 초기화됨): {e}")
            else:
                logger.info("ℹ️ GPU가 감지되지 않았습니다. CPU 모드로 동작합니다.")
        except ImportError:
            pass # TF 미설치 시 조용히 넘어감
        except Exception as e:
            logger.warning(f"⚠️ GPU 초기화 중 예외 발생: {e}")

    def initialize_apis(self):
        """API 초기화 (설정에 따라 선택적 초기화)"""
        try:
            logger.info("API 초기화 시작")
            from config.settings import API_CONFIG
            
            # [New] API 활성화 상태 및 키 검증 로깅
            logger.info("=" * 40)
            logger.info("📡 API 활성화 설정 상태:")
            
            all_apis_connected = True # 전체 API 연결 성공 여부
            
            from config.settings import (
                SHINHAN_API_KEY, SHINHAN_API_SECRET, SHINHAN_ACCOUNT,
                KIWOOM_API_KEY, KIWOOM_API_SECRET, KIWOOM_ACCOUNT,
                DAISHIN_API_KEY, DAISHIN_API_SECRET, DAISHIN_ACCOUNT,
                UPBIT_API_KEY, UPBIT_API_SECRET,
                BINANCE_API_KEY, BINANCE_API_SECRET
            )

            for api_name, is_enabled in API_CONFIG.items():
                status = "✅ 활성화" if is_enabled else "❌ 비활성화"
                logger.info(f"   - {api_name.upper()}: {status}")
                
                if is_enabled:
                    missing = []
                    if api_name == "shinhan":
                        if not SHINHAN_API_KEY or "your_" in SHINHAN_API_KEY: missing.append("Key")
                        if not SHINHAN_API_SECRET or "your_" in SHINHAN_API_SECRET: missing.append("Secret")
                        if not SHINHAN_ACCOUNT or "your_" in SHINHAN_ACCOUNT: missing.append("Account")
                    elif api_name == "kiwoom":
                        if not KIWOOM_API_KEY or "your_" in KIWOOM_API_KEY: missing.append("Key")
                        if not KIWOOM_API_SECRET or "your_" in KIWOOM_API_SECRET: missing.append("Secret")
                        if not KIWOOM_ACCOUNT or "your_" in KIWOOM_ACCOUNT: missing.append("Account")
                    elif api_name == "daishin":
                        if not DAISHIN_API_KEY or "your_" in DAISHIN_API_KEY: missing.append("Key")
                        if not DAISHIN_API_SECRET or "your_" in DAISHIN_API_SECRET: missing.append("Secret")
                        if not DAISHIN_ACCOUNT or "your_" in DAISHIN_ACCOUNT: missing.append("Account")
                    elif api_name == "upbit":
                        if not UPBIT_API_KEY or "your_" in UPBIT_API_KEY: missing.append("Key")
                        if not UPBIT_API_SECRET or "your_" in UPBIT_API_SECRET: missing.append("Secret")
                    elif "binance" in api_name:
                        if not BINANCE_API_KEY or "your_" in BINANCE_API_KEY: missing.append("Key")
                        if not BINANCE_API_SECRET or "your_" in BINANCE_API_SECRET: missing.append("Secret")
                    
                    if missing:
                        logger.warning(f"     ⚠️ 경고: API 키 설정이 누락되었거나 기본값입니다! ({', '.join(missing)})")
            
            logger.info("=" * 40)
            
            # 신한투자 API
            if API_CONFIG.get("shinhan", False):
                from config.settings import SHINHAN_API_KEY, SHINHAN_API_SECRET, SHINHAN_ACCOUNT
                self.shinhan_api = ShinhanAPI(SHINHAN_API_KEY, SHINHAN_API_SECRET, SHINHAN_ACCOUNT)
                self.shinhan_api.connect()
                logger.info("✅ 신한투자 API 연결 완료")
            
            # 키움증권 API
            if API_CONFIG.get("kiwoom", False):
                from config.settings import KIWOOM_API_KEY, KIWOOM_API_SECRET, KIWOOM_ACCOUNT
                self.kiwoom_api = KiwoomAPI(KIWOOM_API_KEY, KIWOOM_API_SECRET, KIWOOM_ACCOUNT)
                self.kiwoom_api.connect()
                logger.info("✅ 키움증권 API 연결 완료")
            
            # 대신증권 API
            if API_CONFIG.get("daishin", False):
                from config.settings import DAISHIN_API_KEY, DAISHIN_API_SECRET, DAISHIN_ACCOUNT
                self.daishin_api = DaishinAPI(DAISHIN_API_KEY, DAISHIN_API_SECRET, DAISHIN_ACCOUNT)
                self.daishin_api.connect()
                logger.info("✅ 대신증권 API 연결 완료")
            
            # 업비트 API (암호화폐)
            if API_CONFIG.get("upbit", False):
                try:
                    logger.info("   [UPBIT] API 설정 로드 및 객체 생성 중...")
                    from config.settings import UPBIT_API_KEY, UPBIT_API_SECRET
                    self.crypto_api = UpbitAPI(UPBIT_API_KEY, UPBIT_API_SECRET)
                    
                    logger.info("   [UPBIT] 서버 연결 시도 중...")
                    self.crypto_api.connect()
                    
                    # 리포트 매니저 초기화
                    self.report_manager = ReportManager(self.crypto_api)
                except Exception as e:
                    logger.error(f"❌ 업비트 API 초기화 실패: {e}")
                    all_apis_connected = False
            
            # 바이낸스 현물 API
            if API_CONFIG.get("binance_spot", False):
                try:
                    from config.settings import BINANCE_API_KEY, BINANCE_API_SECRET
                    self.binance_spot_api = BinanceAPI(BINANCE_API_KEY, BINANCE_API_SECRET, account_type='spot')
                    self.binance_spot_api.connect()
                    # [New] 에러 콜백 등록 (연결 끊김 시 즉시 알림)
                    self.binance_spot_api.add_error_callback(self._on_binance_error)
                    logger.info("✅ 바이낸스 현물 API 초기화 완료")
                except Exception as e:
                    logger.error(f"❌ 바이낸스 현물 API 초기화 실패: {e}")
                    self.binance_spot_api = None
                    all_apis_connected = False

            # 바이낸스 선물 API
            if API_CONFIG.get("binance_futures", False):
                try:
                    from config.settings import BINANCE_API_KEY, BINANCE_API_SECRET
                    self.binance_futures_api = BinanceAPI(BINANCE_API_KEY, BINANCE_API_SECRET, account_type='future')
                    self.binance_futures_api.connect()
                    self.binance_futures_api.add_error_callback(self._on_binance_error)
                    logger.info("✅ 바이낸스 선물 API 초기화 완료")
                except Exception as e:
                    logger.error(f"❌ 바이낸스 선물 API 초기화 실패: {e}")
                    self.binance_futures_api = None
                    all_apis_connected = False
            
            if all_apis_connected:
                logger.info("✅ 모든 API 초기화 완료")
                return True
            else:
                logger.error("❌ 일부 API 연결 실패. 재시도를 위해 초기화를 중단합니다.")
                return False
        
        except Exception as e:
            logger.error(f"API 초기화 오류: {e}")
            return False
    
    def load_dynamic_config(self):
        """동적 설정 파일 로드 (백테스팅 결과 반영)"""
        config_file = "data/dynamic_config.json"
        if not os.path.exists(config_file):
            return

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # K값 적용
            if "k_value" in config:
                # [New] .env 우선순위 처리: .env에 K값이 설정되어 있으면 동적 설정 무시
                if os.getenv("CRYPTO_K_VALUE") is not None:
                    logger.info(f"ℹ️ .env 설정 우선: 동적 K-Value({config.get('k_value')})를 무시하고 현재 설정({TRADING_CONFIG['crypto']['k_value']})을 유지합니다.")
                else:
                    k_val = float(config["k_value"])
                    # 안전장치: 0.4 ~ 0.7 범위 확인
                    if 0.4 <= k_val <= 0.7:
                        TRADING_CONFIG["crypto"]["k_value"] = k_val
                        logger.info(f"🔄 동적 설정 적용: K-Value = {k_val} (Updated: {config.get('updated_at')})")
                    else:
                        logger.warning(f"⚠️ 동적 설정 K값({k_val})이 허용 범위(0.4~0.7)를 벗어나 무시합니다.")
        except Exception as e:
            logger.error(f"동적 설정 로드 오류: {e}")

    def _initialize_strategies(self):
        """전략 객체 초기화 및 갱신 (설정 변경 시 호출)"""
        self.strategies = {}
        for key in ["crypto", "binance_spot", "binance_futures"]:
            entry_strategy = TRADING_CONFIG[key].get("entry_strategy", "breakout")
            strategy_type = TRADING_CONFIG[key].get("strategy_type", "technical")
            
            if entry_strategy == "heikin_ashi":
                self.strategies[key] = HeikinAshiStrategy(ML_CONFIG["lookback_window"])
            elif entry_strategy == "turtle_bollinger":
                self.strategies[key] = TurtleBollingerStrategy(ML_CONFIG["lookback_window"])
            elif entry_strategy == "agile":
                self.strategies[key] = AgileStrategy(ML_CONFIG["lookback_window"])
            elif entry_strategy == "volume_trend":
                self.strategies[key] = VolumeTrendStrategy(ML_CONFIG["lookback_window"])
            elif entry_strategy == "ma_trend":
                self.strategies[key] = MATrendStrategy(ML_CONFIG["lookback_window"])
            elif entry_strategy == "early_bird":
                self.strategies[key] = EarlyBirdStrategy(ML_CONFIG["lookback_window"])
            elif strategy_type == "ml":
                self.strategies[key] = self.ml_strategy
            else:
                self.strategies[key] = self.technical_strategy
            
            logger.info(f"🤖 [{key.upper()}] 적용 전략 갱신: {type(self.strategies[key]).__name__} (Mode: {entry_strategy})")

    def check_env_updates(self):
        """
        .env 파일 변경 감지 및 Hot-Reload
        """
        # [수정] 빌드 환경 호환 절대 경로 사용
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(os.path.abspath(sys.executable))
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
        env_path = os.path.join(base_dir, ".env")
        if not os.path.exists(env_path):
            return

        try:
            mtime = os.path.getmtime(env_path)
            if self.last_env_mtime == 0:
                self.last_env_mtime = mtime
                return

            if mtime > self.last_env_mtime:
                logger.info("🔄 .env 파일 변경 감지! 설정을 다시 로드합니다.")
                self.last_env_mtime = mtime
                
                from dotenv import load_dotenv
                load_dotenv(env_path, override=True) # [Fix] 명시적으로 .env 파일만 리로드
                
                # 주요 설정값 갱신
                TRADING_CONFIG["crypto"]["k_value"] = float(os.getenv("CRYPTO_K_VALUE", 0.6))
                TRADING_CONFIG["crypto"]["entry_strategy"] = os.getenv("CRYPTO_ENTRY_STRATEGY", "breakout")
                stop_loss = float(os.getenv("CRYPTO_STOP_LOSS", 0.0))
                if stop_loss > 0:
                    TRADING_CONFIG["crypto"]["stop_loss_percent"] = stop_loss
                
                # [New] 바이낸스 설정 갱신 추가
                TRADING_CONFIG["binance_spot"]["entry_strategy"] = os.getenv("BINANCE_SPOT_ENTRY_STRATEGY", "breakout")
                TRADING_CONFIG["binance_futures"]["entry_strategy"] = os.getenv("BINANCE_FUTURES_ENTRY_STRATEGY", "breakout")
                
                # [New] 바이낸스 파라미터 갱신 (TP/SL)
                TRADING_CONFIG["binance_spot"]["take_profit_percent"] = float(os.getenv("BINANCE_SPOT_TAKE_PROFIT", TRADING_CONFIG["binance_spot"]["take_profit_percent"]))
                TRADING_CONFIG["binance_spot"]["stop_loss_percent"] = float(os.getenv("BINANCE_SPOT_STOP_LOSS", 0.0))
                TRADING_CONFIG["binance_futures"]["take_profit_percent"] = float(os.getenv("BINANCE_FUTURES_TAKE_PROFIT", TRADING_CONFIG["binance_futures"]["take_profit_percent"]))
                TRADING_CONFIG["binance_futures"]["stop_loss_percent"] = float(os.getenv("BINANCE_FUTURES_STOP_LOSS", 0.0))

                # 전략 객체 재초기화 (클래스 변경 대응)
                self._initialize_strategies()
                
                logger.info(f"✅ 설정 갱신 완료 (Hot-Reload)")
        except Exception as e:
            logger.error(f"설정 리로드 중 오류: {e}")

    def update_crypto_symbols(self):
        # """거래량 기반으로 암호화폐 종목 자동 업데이트"""
        """거래량 및 변동성(ATR) 기반 종목 자동 업데이트"""
        if not VOLUME_CONFIG["auto_select_enabled"]:
            return
        
        current_time = time.time()
        if current_time - self.last_volume_update < VOLUME_CONFIG["update_interval"]:
            return  # 아직 업데이트 시간이 아님
        
        try:
            logger.info("거래량 기반 종목 업데이트 시작...")
            
            # [최적화] API 부하 분산을 위한 예외 처리 및 Fallback
            logger.info("🔄 종목 선정 프로세스 시작 (거래량 + 변동성 필터)...")

            # 1. 전체 마켓 조회
            try:
                # Upbit에서 모든 마켓 정보 조회
                markets = self.crypto_api.exchange.fetch_tickers()
            except Exception as e:
                logger.warning(f"⚠️ 마켓 조회 실패: {e}")
                return
            
            # 2. 거래대금 상위 30개 1차 필터링
            candidates = []
            min_vol = max(VOLUME_CONFIG["min_volume_krw"], 10_000_000_000) # 최소 100억

            for symbol, ticker in markets.items():
                if "/KRW" in symbol and ticker.get('quoteVolume') is not None:
                    vol = ticker['quoteVolume']
                    
                    if vol >= min_vol:
                        if symbol not in VOLUME_CONFIG["exclude_symbols"]:
                           # 메이저 제외 옵션
                            if VOLUME_CONFIG.get("exclude_major_coins", False):
                                if symbol in ["BTC/KRW", "ETH/KRW"]:
                                    continue
                            candidates.append({'symbol': symbol, 'volume': vol})
            
            logger.info(f"1차 필터링(거래대금 {min_vol/100000000:.0f}억↑) 통과: {len(candidates)}개")

            # 거래대금 순 정렬 후 상위 30개 추출
            candidates.sort(key=lambda x: x['volume'], reverse=True)
            top_30 = candidates[:30]
            
            logger.info(f"📊 거래대금 상위 {len(top_30)}개 종목 분석 중 (데이터 검증 및 ATR 계산)...")
            
            final_candidates = []
            
            # 3. 데이터 검증 및 ATR 계산
            for item in top_30:
                symbol = item['symbol']
                
                # 일봉 데이터 200개 요청 (데이터 충분한지 검증 + ATR 계산)
                # crypto_api.get_ohlcv의 검증 로직을 통과한 데이터만 사용 (min_required_data=200)
                # 200개 미만인 신규 코인은 여기서 자동으로 걸러짐
                df = self.crypto_api.get_ohlcv(symbol, timeframe="1d", count=200, min_required_data=200)
                
                if df.empty:
                    continue
                
                # ATR 계산 (변동성 지표)
                try:
                    atr_indicator = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
                    atr = atr_indicator.average_true_range().iloc[-1]
                    price = df['close'].iloc[-1]
                    
                    # 변동률(%)로 변환
                    atr_pct = (atr / price) * 100
                    
                    final_candidates.append({
                        'symbol': symbol,
                        'atr_pct': atr_pct,
                        'volume': item['volume']
                    })
                except Exception as e:
                    logger.warning(f"{symbol} 지표 계산 오류: {e}")
                
                time.sleep(0.1) # Rate Limit
            
            # 4. 변동성(ATR) 순으로 정렬하여 상위 10개 선정
            # 변동성이 높아야 봇이 수익을 낼 기회가 많음
            final_candidates.sort(key=lambda x: x['atr_pct'], reverse=True)
            
            selected = final_candidates[:VOLUME_CONFIG["max_symbols"]]
            self.crypto_symbols = [x['symbol'] for x in selected]
            
            # [New] 대시보드 표시를 위해 포트폴리오 메타데이터에 저장
            self.crypto_portfolio.metadata["selected_symbols"] = self.crypto_symbols
            self.crypto_portfolio.save_state("data/crypto_portfolio.json")
            
            logger.info(f"✅ 최종 선정된 {len(self.crypto_symbols)}개 종목 (변동성 Top):")
            for item in selected:
                logger.info(f"  - {item['symbol']} (ATR: {item['atr_pct']:.2f}%, Vol: {item['volume']/100000000:.0f}억)")
            
            self.last_volume_update = current_time
            
            # 웹소켓 갱신
            if self.crypto_api and hasattr(self.crypto_api, 'subscribe_websocket'):
                all_symbols = list(set(self.crypto_symbols) | set(self.crypto_portfolio.positions.keys()))
                self.crypto_api.subscribe_websocket(all_symbols)

        except Exception as e:
            logger.error(f"종목 업데이트 중 오류: {e}")         
                            
            
    
    def recommend_strategy(self, auto_update: bool = False):
        """현재 시장 변동성을 분석하여 전략 추천"""
        if not self.crypto_api:
            return

        try:
            # 대표 코인(BTC)으로 시장 상황 분석
            symbol = "BTC/KRW"
            # 일봉 데이터 조회 (최근 30일)
            df = self.crypto_api.get_ohlcv(symbol, timeframe="1d")
            if df.empty or len(df) < 20:
                logger.warning("데이터 부족으로 전략 추천 불가")
                return

            # 변동성 계산 (최근 14일 기준 일일 수익률의 표준편차)
            returns = df['close'].pct_change()
            volatility = returns.tail(14).std() * 100  # 퍼센트 단위
            
            logger.info("=" * 60)
            logger.info(f"📊 시장 상황 분석 ({symbol})")
            logger.info(f"   - 일일 변동성(Volatility): {volatility:.2f}%")
            
            recommended = "mid_term" # 기본값
            reason = ""

            # 변동성 기준 전략 추천 로직
            if volatility >= 4.0:
                recommended = "scalping"
                reason = "매우 높은 변동성 (4%↑) → 리스크 최소화를 위한 초단타(Scalping) 유리"
            elif volatility >= 2.0:
                recommended = "short_term"
                reason = "높은 변동성 (2%~4%) → 데이트레이딩(Short Term) 유리"
            elif volatility >= 1.0:
                recommended = "mid_term"
                reason = "보통 변동성 (1%~2%) → 스윙(Mid Term) 유리"
            else:
                recommended = "long_term"
                reason = "낮은 변동성 (1%↓) → 긴 호흡의 추세 추종(Long Term) 유리"
            
            logger.info(f"💡 AI 전략 추천: '{recommended}'")
            logger.info(f"   - 이유: {reason}")
            
            # 현재 설정과 비교
            from config.settings import selected_strategy_name
            
            if auto_update and recommended != selected_strategy_name:
                logger.info(f"🔄 전략 자동 변경 실행: '{selected_strategy_name}' → '{recommended}'")
                self.update_strategy_config(recommended)
            elif recommended != selected_strategy_name:
                logger.info(f"⚠️ 현재 설정된 전략('{selected_strategy_name}')과 추천 전략이 다릅니다.")
                logger.info(f"   👉 .env 파일에서 CRYPTO_STRATEGY_PRESET={recommended} 로 변경을 고려해보세요.")
            else:
                logger.info(f"✅ 현재 설정된 전략이 시장 상황에 적합합니다.")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"전략 추천 중 오류: {e}")

    def update_strategy_config(self, strategy_name: str):
        """전략 설정을 동적으로 업데이트"""
        try:
            from config.settings import STRATEGY_PRESETS, TRADING_CONFIG
            
            if strategy_name not in STRATEGY_PRESETS:
                logger.error(f"존재하지 않는 전략 프리셋: {strategy_name}")
                return

            new_config = STRATEGY_PRESETS[strategy_name]
            
            # 1. 메모리 상의 설정 업데이트
            TRADING_CONFIG["crypto"]["take_profit_percent"] = new_config["take_profit_percent"]
            TRADING_CONFIG["crypto"]["timeframe"] = new_config["timeframe"]
            
            # [중요] 타임프레임 변경 시 OHLCV 캐시 초기화 (데이터 불일치 방지)
            self.ohlcv_cache.clear()
            self.last_ohlcv_fetch.clear()
            logger.info("🧹 전략 변경으로 인한 OHLCV 데이터 캐시 초기화 완료")
            
            # 2. 리스크 관리자 설정 즉시 반영
            self.crypto_risk_manager.take_profit_percent = new_config["take_profit_percent"]
            
            # 3. 포트폴리오 메타데이터 업데이트 및 저장
            self.crypto_portfolio.metadata.update({
                "strategy": strategy_name,
                "timeframe": new_config["timeframe"]
            })
            self.crypto_portfolio.save_state("data/crypto_portfolio.json")
            
            logger.info(f"✅ 전략 업데이트 완료: {strategy_name}")
            logger.info(f"   - 타임프레임: {new_config['timeframe']}")
            logger.info(f"   - 익절: {new_config['take_profit_percent']*100:.1f}%")
            
        except Exception as e:
            logger.error(f"전략 업데이트 중 오류: {e}")

    def check_ws_latency(self):
        """웹소켓 데이터 수신 지연 확인"""
        # 바이낸스 현물/선물 각각 체크
        for api_name, api in [("SPOT", getattr(self, 'binance_spot_api', None)), 
                              ("FUTURES", getattr(self, 'binance_futures_api', None))]:
            if api and api.use_websocket:
                last_update = api.last_ws_update
                if api.is_ws_ready and last_update > 0 and (time.time() - last_update > 60):
                    msg = f"⚠️ [BINANCE_{api_name}] 웹소켓 지연! (마지막: {int(time.time() - last_update)}초 전)"
                    logger.warning(msg)
                    self._send_telegram_alert(msg)
                    api.reconnect_websocket()

    def refresh_binance_websocket(self):
        """바이낸스 웹소켓 정기 재연결 (50분 주기)"""
        if getattr(self, 'binance_spot_api', None) and self.binance_spot_api.use_websocket:
            self.binance_spot_api.reconnect_websocket()
        if getattr(self, 'binance_futures_api', None) and self.binance_futures_api.use_websocket:
            self.binance_futures_api.reconnect_websocket()

    def check_api_health(self):
        """API 연결 상태 주기적 점검"""
        if getattr(self, 'binance_spot_api', None):
            self.binance_spot_api.health_check()
        if getattr(self, 'binance_futures_api', None):
            self.binance_futures_api.health_check()

    def _check_liquidation_safety(self, symbol: str):
        """[New] 바이낸스 선물 청산 위험 모니터링 및 강제 종료"""
        if not getattr(self, 'binance_futures_api', None):
            return

        risk_data = self.binance_futures_api.get_liquidation_risk(symbol)
        if not risk_data:
            return

        dist_pct = risk_data.get('distance_pct', 1.0)
        # 청산가까지 거리가 20% 미만이면 위험 (강제 청산)
        if dist_pct < 0.20:
            msg = f"🚨 [LIQUIDATION_ALERT] {symbol} 청산 위험 감지! (거리: {dist_pct*100:.2f}%) -> 강제 포지션 종료 실행"
            logger.critical(msg)
            self._send_telegram_alert(msg)
            # 시장가로 즉시 전량 청산
            qty = self.binance_futures_portfolio.positions.get(symbol, 0)
            if qty > 0:
                self.binance_futures_api.sell(symbol, qty, is_stop_loss=True)

    def _on_binance_error(self, message: str):
        """바이낸스 API 에러 콜백 처리"""
        self._send_telegram_alert(f"🚨 [BINANCE] {message}")

    def _send_telegram_alert(self, message, parse_mode=None):
        """텔레그램으로 긴급 알림 전송"""
        try:
            from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
                if parse_mode:
                    data["parse_mode"] = parse_mode
                response = requests.post(url, data=data, timeout=5)
                if response.status_code != 200:
                    logger.error(f"텔레그램 전송 실패: {response.text}")
        except Exception as e:
            logger.error(f"텔레그램 전송 실패: {e}")

    def _send_config_summary(self):
        """봇 시작 시 현재 설정 요약 전송"""
        try:
            from config.settings import selected_strategy_name
            from config.settings import selected_strategy_name, spot_strategy_name, futures_strategy_name
            
            msg = "🤖 *자동매매 봇 시작 알림*\n\n"
            msg += f"📌 *적용 전략*: `{selected_strategy_name.upper()}`\n"
            
            # Crypto Config
            c_conf = TRADING_CONFIG["crypto"]
            msg += "\n📊 *[UPBIT] 설정*\n"
            msg += f"📊 *[UPBIT] 설정 ({selected_strategy_name.upper()})*\n"
            msg += f"• 진입전략: `{c_conf.get('entry_strategy', 'Unknown')}`\n"
            msg += f"• 타임프레임: `{c_conf['timeframe']}`\n"
            msg += f"• K-Value: `{c_conf['k_value']}`\n"
            msg += f"• 익절률: `{c_conf['take_profit_percent']*100:.1f}%`\n"
            msg += f"• 손절률: `{c_conf['stop_loss_percent']*100:.1f}%` (0=ATR)\n"
            msg += f"• 트레일링스탑: `{c_conf['trailing_stop_percent']*100:.1f}%`\n"
            msg += f"• 최대보유: `{c_conf['max_positions']}종목`\n"
            
            # Binance Spot
            if getattr(self, 'binance_spot_api', None):
                b_conf = TRADING_CONFIG["binance_spot"]
                msg += "\n📊 *[BINANCE SPOT] 설정*\n"
                msg += f"\n📊 *[BINANCE SPOT] 설정 ({spot_strategy_name.upper()})*\n"
                msg += f"• 진입전략: `{b_conf.get('entry_strategy', 'Unknown')}`\n"
                msg += f"• 타임프레임: `{b_conf['timeframe']}`\n"
                msg += f"• 익절률: `{b_conf['take_profit_percent']*100:.1f}%`\n"
                msg += f"• 트레일링스탑: `{b_conf.get('trailing_stop_percent', 0)*100:.1f}%`\n"

            # Binance Futures
            if getattr(self, 'binance_futures_api', None):
                b_conf = TRADING_CONFIG["binance_futures"]
                msg += "\n📊 *[BINANCE FUTURES] 설정*\n"
                msg += f"\n📊 *[BINANCE FUTURES] 설정 ({futures_strategy_name.upper()})*\n"
                msg += f"• 진입전략: `{b_conf.get('entry_strategy', 'Unknown')}`\n"
                msg += f"• 타임프레임: `{b_conf['timeframe']}`\n"
                msg += f"• 레버리지: `{b_conf.get('leverage', 1)}x`\n"
                msg += f"• 익절률: `{b_conf['take_profit_percent']*100:.1f}%`\n"
                msg += f"• 트레일링스탑: `{b_conf.get('trailing_stop_percent', 0)*100:.1f}%`\n"
            
            self._send_telegram_alert(msg, parse_mode="Markdown")
            logger.info("✅ 설정 요약 텔레그램 전송 완료")
            
        except Exception as e:
            logger.error(f"설정 요약 전송 실패: {e}")

    def _update_status(self, status=None):
        """봇 상태(Heartbeat) 업데이트"""
        # [New] status가 명시되지 않은 경우(스케줄러 호출) 내부 상태에 따라 결정
        if status is None:
            status = "running" if getattr(self, 'is_ready', False) else "warming_up"
            
        status_file = "data/bot_status.json"
        try:
            # CPU 사용량 (non-blocking, 이전 호출과의 차이)
            cpu_usage = self.process.cpu_percent(interval=None)
            # 메모리 사용량 (MB 단위)
            memory_usage = self.process.memory_info().rss / (1024 * 1024)

            data = {
                "status": status,
                "timestamp": time.time(),
                "pid": os.getpid(),
                "cpu": cpu_usage,
                "memory": memory_usage,
                "warmup_current": getattr(self, 'warmup_counter', 0),
                "warmup_total": 3
            }
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"상태 업데이트 오류: {e}")

    def _check_for_commands(self):
        """대시보드 등 외부로부터 들어온 명령을 확인하고 실행"""
        command_file = "data/command.json"
        if not os.path.exists(command_file):
            return

        try:
            with open(command_file, 'r', encoding='utf-8') as f:
                command_data = json.load(f)
            
            # 오래된 커맨드 무시 (1분 이상)
            if time.time() - command_data.get("timestamp", 0) > 60:
                os.remove(command_file)
                return

            cmd = command_data.get("command")
            payload = command_data.get("payload")

            if cmd == "change_strategy":
                logger.info("="*60)
                logger.info(f"🕹️ 대시보드로부터 전략 변경 명령 수신: '{payload}'")
                self.update_strategy_config(payload)
                logger.info("="*60)
                os.remove(command_file) # 처리 후 파일 삭제
            
            elif cmd == "restart_bot":
                os.remove(command_file) # 재시작 전 파일 삭제
                logger.warning("🔄 대시보드로부터 재시작 명령 수신. 봇을 재시작합니다...")
                self._update_status("restarting")
                self.stop()
                # 현재 프로세스 재시작 (운영체제별 호환성 고려)
                os.execv(sys.executable, [sys.executable] + sys.argv)
                
            elif cmd == "stop_bot":
                os.remove(command_file) # 종료 전 파일 삭제
                logger.warning("🛑 대시보드로부터 종료 명령 수신. 봇을 종료합니다...")
                self._update_status("stopped")
                self.stop()
                os._exit(0)
            
        except Exception as e:
            logger.error(f"커맨드 처리 중 오류: {e}")
            if os.path.exists(command_file):
                os.remove(command_file)

    def _update_env_file(self, key: str, value: str):
        """Update .env file safely"""
        try:
            # [수정] 빌드 환경 호환 절대 경로 사용
            if getattr(sys, 'frozen', False):
                base_dir = os.path.dirname(os.path.abspath(sys.executable))
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))
            
            env_path = os.path.join(base_dir, ".env")
            
            lines = []
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            
            key_found = False
            new_lines = []
            for line in lines:
                if line.strip().startswith(f"{key}="):
                    new_lines.append(f"{key}={value}\n")
                    key_found = True
                else:
                    new_lines.append(line)
            
            if not key_found:
                new_lines.append(f"{key}={value}\n")
            
            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
                
            logger.info(f"💾 .env 파일 갱신 완료: {key}={value}")
            
        except Exception as e:
            logger.error(f".env 파일 업데이트 오류: {e}")

    def run_periodic_backtest(self, wait: bool = False):
        """정기 백테스트 및 설정 최적화 실행 (별도 프로세스)"""
        logger.info("🧪 [Scheduler] 정기 백테스트 및 최적화 작업 시작...")
        try:
            # 스크립트 경로 (main.py와 같은 폴더에 있다고 가정)
            if getattr(sys, 'frozen', False):
                base_dir = os.path.dirname(os.path.abspath(sys.executable))
                # [Fix] 빌드된 환경에서는 별도 실행 파일(Backtester) 실행
                if sys.platform == "win32":
                    target_exe = os.path.join(base_dir, "Backtester.exe")
                else:
                    target_exe = os.path.join(base_dir, "Backtester")
                
                if os.path.exists(target_exe):
                    if wait:
                        self._send_telegram_alert("⏳ *[시스템 알림]*\n봇 초기화 중... 최신 데이터를 기반으로 전략 최적화를 진행합니다.\n(약 1~3분 소요)", parse_mode="Markdown")
                        logger.info("⏳ 초기 백테스트 실행 중... (완료 시까지 대기)")
                        subprocess.call([target_exe], cwd=base_dir)
                        logger.info("✅ 초기 백테스트 완료. 설정을 갱신합니다.")
                        self.check_env_updates()
                    else:
                        subprocess.Popen([target_exe], cwd=base_dir)
                        logger.info(f"🚀 백테스트 프로세스(EXE) 시작: {target_exe}")
                else:
                    logger.error(f"❌ 백테스트 실행 파일을 찾을 수 없습니다: {target_exe}")
                return
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                script_path = os.path.join(base_dir, "run_backtest_all.py")
                
                if not os.path.exists(script_path):
                    logger.error(f"❌ 백테스트 스크립트를 찾을 수 없습니다: {script_path}")
                    return

                # 비동기 실행 (봇 메인 루프 차단 방지)
                if wait:
                    self._send_telegram_alert("⏳ *[시스템 알림]*\n봇 초기화 중... 최신 데이터를 기반으로 전략 최적화를 진행합니다.\n(약 1~3분 소요)", parse_mode="Markdown")
                    logger.info("⏳ 초기 백테스트 실행 중... (완료 시까지 대기)")
                    subprocess.call([sys.executable, script_path], cwd=base_dir)
                    logger.info("✅ 초기 백테스트 완료. 설정을 갱신합니다.")
                    self.check_env_updates()
                else:
                    subprocess.Popen([sys.executable, script_path], cwd=base_dir)
                    logger.info("🚀 백테스트 프로세스(Script)가 백그라운드에서 시작되었습니다.")
            
        except Exception as e:
            logger.error(f"백테스트 실행 중 오류: {e}")

    def optimize_strategy_params(self):
        """전략 파라미터(K, 익절, 손절) 자동 최적화"""
        if not self.crypto_api: return

        logger.info("⚙️ 전략 파라미터(K, TP, SL) 자동 최적화 시작 (최근 7일 데이터)...")
        
        # 최적화 후보군 (Grid Search) - 핵심 값 위주로 테스트
        k_candidates = [0.4, 0.5, 0.6]
        tp_candidates = [0.03, 0.05, 0.10] # 3%, 5%, 10%
        sl_candidates = [0.01, 0.03, 0.05] # 1%, 3%, 5%
        
        # 분석 대상 종목
        targets = ["BTC/KRW", "ETH/KRW", "XRP/KRW", "SOL/KRW"]
        targets.extend(self.crypto_symbols[:2]) # 상위 2개 추가
        targets = list(set(targets))
        
        # 현재 설정값
        current_k = TRADING_CONFIG["crypto"]["k_value"]
        current_tp = TRADING_CONFIG["crypto"]["take_profit_percent"]
        current_sl = TRADING_CONFIG["crypto"]["stop_loss_percent"]
        
        best_params = (current_k, current_tp, current_sl)
        best_score = -1.0 # 승률 기준
        
        # 원본 설정 백업 (오류 시 복구용)
        original_config = {
            "k": current_k,
            "tp": current_tp,
            "sl": current_sl
        }
        
        try:
            import itertools
            combinations = list(itertools.product(k_candidates, tp_candidates, sl_candidates))
            
            logger.info(f"🧪 총 {len(combinations)}개 파라미터 조합 테스트 진행...")
            
            for k, tp, sl in combinations:
                # K값은 전략 내부 로직에 영향을 주므로 설정 변경 필요
                TRADING_CONFIG["crypto"]["k_value"] = k
                
                total_trades = 0
                total_wins = 0
                
                for symbol in targets:
                    # 데이터 수집 (캐싱 활용, 15분봉 500개)
                    df = self.crypto_api.get_ohlcv(symbol, timeframe="15m", count=500)
                    if df.empty or len(df) < 100: continue
                    
                    # 최근 데이터로 테스트
                    test_len = min(len(df), 480)
                    test_data = df.tail(test_len)
                    
                    analyzer = WalkForwardAnalyzer(
                        test_data, 
                        train_period=20,
                        test_period=len(test_data)-50,
                        fee=0.0005,
                        slippage=0.001,
                        take_profit=tp,  # 익절 적용
                        stop_loss=sl     # 손절 적용
                    )
                    
                    strategy = TechnicalStrategy(lookback_window=20)
                    res = analyzer._backtest_period(strategy, test_data, lookback=50)
                    
                    if res['trade_count'] > 0:
                        total_trades += res['trade_count']
                        total_wins += (res['win_rate'] * res['trade_count'])
                
                avg_win_rate = total_wins / total_trades if total_trades > 0 else 0
                
                if avg_win_rate > best_score:
                    best_score = avg_win_rate
                    best_params = (k, tp, sl)
            
            # 최적값 적용
            new_k, new_tp, new_sl = best_params
            
            if best_params != (current_k, current_tp, current_sl):
                logger.info(f"✅ 최적 파라미터 발견! (승률 {best_score*100:.1f}%)")
                logger.info(f"   K: {current_k} -> {new_k}")
                logger.info(f"   TP: {current_tp*100:.1f}% -> {new_tp*100:.1f}%")
                logger.info(f"   SL: {current_sl*100:.1f}% -> {new_sl*100:.1f}%")
                
                self._update_env_file("CRYPTO_K_VALUE", str(new_k))
                self._update_env_file("CRYPTO_TAKE_PROFIT", str(new_tp))
                self._update_env_file("CRYPTO_STOP_LOSS", str(new_sl))
                
                TRADING_CONFIG["crypto"]["k_value"] = new_k
                TRADING_CONFIG["crypto"]["take_profit_percent"] = new_tp
                TRADING_CONFIG["crypto"]["stop_loss_percent"] = new_sl
                
                # 리스크 관리자에도 반영
                self.crypto_risk_manager.take_profit_percent = new_tp
                
                self._send_telegram_alert(f"⚙️ [AUTO_OPT] 전략 파라미터 최적화 완료\nK: {new_k}\nTP: {new_tp*100:.1f}%\nSL: {new_sl*100:.1f}%\n(예상 승률: {best_score*100:.1f}%)")
            else:
                logger.info("ℹ️ 현재 설정이 최적입니다.")
                # 설정 원복 (K값 등)
                TRADING_CONFIG["crypto"]["k_value"] = current_k
                
        except Exception as e:
            logger.error(f"파라미터 최적화 중 오류: {e}")
            # 오류 시 원복
            TRADING_CONFIG["crypto"]["k_value"] = original_config["k"]

    def find_best_k(self):
        """
        [미니 전진분석] 매 시간 최근 데이터를 복기하여 최적의 K값 탐색
        로직: 최근 200개 캔들 기준, K값 0.3~0.8 시뮬레이션 -> 최적값 메모리 반영
        """
        if not self.crypto_api: return

        logger.info("🧪 [미니 전진분석] 최적 K값 탐색 시작 (최근 200 캔들)...")
        
        # 대표 종목으로 테스트 (BTC/KRW)
        target_symbol = "BTC/KRW"
        timeframe = TRADING_CONFIG["crypto"].get("timeframe", "15m")
        
        # 데이터 수집 (최근 200개 + 지표 계산용 여유분 100개)
        df = self.crypto_api.get_ohlcv(target_symbol, timeframe=timeframe, count=300)
        
        if df.empty or len(df) < 200:
            logger.warning(f"⚠️ 데이터 부족으로 K값 최적화 스킵 ({len(df)} rows)")
            return

        # 테스트할 K값 범위 (0.3 ~ 0.8, 0.05 단위)
        k_candidates = [round(x, 2) for x in np.arange(0.3, 0.81, 0.05)]
        
        best_k = 0.6 # 기본값
        best_return = -float('inf')
        original_k = TRADING_CONFIG["crypto"]["k_value"]
        
        try:
            # 최근 200개 데이터만 사용 (시장 상황 반영)
            test_data = df.tail(200)
            
            for k in k_candidates:
                # 설정 임시 변경 (TechnicalStrategy가 참조함)
                TRADING_CONFIG["crypto"]["k_value"] = k
                
                # 백테스팅 시뮬레이션
                analyzer = WalkForwardAnalyzer(
                    test_data, 
                    train_period=20, # 최소 지표 계산 기간
                    test_period=len(test_data)-20, 
                    fee=0.0005,
                    slippage=0.001
                )
                strategy = TechnicalStrategy(lookback_window=20)
                res = analyzer._backtest_period(strategy, test_data, lookback=50)
                
                net_return = res['total_return']
                if net_return > best_return:
                    best_return = net_return
                    best_k = k
            
            # 결과 적용 및 로그
            # 가상 자본 1억 기준 수익률 환산
            return_pct = (best_return / 100000000) * 100
            
            if best_return <= 0:
                logger.info(f"⚠️ [OPTIMIZE] 모든 K값 성과 저조 (최고 {return_pct:.2f}%). 보수적 기본값(0.6) 유지.")
                TRADING_CONFIG["crypto"]["k_value"] = 0.6
            else:
                logger.info(f"✅ [OPTIMIZE] 최적 K값 발견: {best_k} (예상 수익률: {return_pct:.2f}%)")
                TRADING_CONFIG["crypto"]["k_value"] = best_k
                # .env 파일은 수정하지 않고 메모리 상에서만 유지
                
        except Exception as e:
            logger.error(f"K값 미니 최적화 중 오류: {e}")
            TRADING_CONFIG["crypto"]["k_value"] = original_k # 오류 시 원복

    def train_ml_model(self):
        """머신러닝 모델 학습"""
        import os
        import time
        logger.info("머신러닝 모델 학습 시작")
        
        # [Fix] EXE 실행 시 절대 경로 사용 (models 폴더 위치 보장)
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(os.path.abspath(sys.executable))
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
        # [Fix] 설정 파일에서 모델 폴더명 로드 (기본값: models)
        models_folder = ML_CONFIG.get("models_dir", "models")
        models_dir = os.path.join(base_dir, models_folder)
        
        # [New] 사용자가 경로를 명확히 알 수 있도록 로그 출력
        logger.info(f"📂 모델 파일 저장 경로: {models_dir}")
        
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            logger.info(f"📂 {models_folder} 폴더를 생성했습니다: {models_dir}")

        try:
            # 여러 API에서 데이터 수집
            apis = [
                ("UPBIT", self.crypto_api), # 암호화폐 추가
                ("신한투자", self.shinhan_api),
                ("키움증권", self.kiwoom_api),
                ("대신증권", self.daishin_api),
            ]
            
            for api_name, api in apis:
                if not api:
                    continue
                
                # API별 대상 종목 선정
                targets = TRADING_CONFIG["korean_stocks"]["symbols"] if api_name != "UPBIT" else \
                          list(set(self.crypto_symbols) | set(self.crypto_portfolio.positions.keys()))
                skipped_symbols = []

                # [Request 2] 종목 선별: 상위 5개 종목만 집중 학습 (속도 향상)
                if len(targets) > 5:
                    targets = targets[:5]
                    logger.info(f"⚡ 학습 대상 최적화: 상위 5개 종목만 학습합니다. ({', '.join(targets)})")

                # 데이터 수집 (순차적 실행 - API Rate Limit 준수)
                training_data_map = {}
                for symbol in targets:
                    if api_name == "UPBIT":
                        timeframe = TRADING_CONFIG["crypto"].get("timeframe", "1d")
                        data = api.get_ohlcv(symbol, timeframe, count=2000, min_required_data=ML_CONFIG["lookback_window"])
                    else:
                        timeframe = TRADING_CONFIG["korean_stocks"].get("timeframe", "1d")
                        data = api.get_ohlcv(symbol, timeframe)
                    
                    if len(data) > ML_CONFIG["lookback_window"]:
                        training_data_map[symbol] = data
                    else:
                        skipped_symbols.append(f"{symbol}({len(data)})")
                    
                    time.sleep(0.2) # Rate Limit

                # [Request 3] 병렬 처리 (Multiprocessing)
                # CPU 코어 수만큼 병렬로 학습 및 검증 수행
                max_workers = min(os.cpu_count(), len(training_data_map))
                if max_workers > 0:
                    logger.info(f"🚀 {max_workers}개의 프로세스로 병렬 학습 시작...")
                    
                    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                        future_to_symbol = {
                            executor.submit(_train_model_task, sym, df, ML_CONFIG, api_name, models_dir): sym 
                            for sym, df in training_data_map.items()
                        }
                        
                        for future in concurrent.futures.as_completed(future_to_symbol):
                            symbol = future_to_symbol[future]
                            try:
                                result_symbol, success, ret = future.result()
                                if isinstance(success, Exception):
                                    logger.error(f"[{symbol}] 학습 중 오류: {success}")
                                elif success:
                                    logger.info(f"✅ [{symbol}] 모델 학습 및 저장 완료 (검증 수익: {ret:,.0f})")
                                else:
                                    logger.warning(f"⚠️ [{symbol}] 전진분석 결과 저조(수익: {ret:,.0f}). 학습 모델을 저장하지 않습니다.")
                            except Exception as e:
                                logger.error(f"[{symbol}] 병렬 처리 결과 수신 중 오류: {e}")
                
                if skipped_symbols:
                    logger.warning(f"⚠️ [{api_name}] 데이터 부족으로 학습 스킵 ({len(skipped_symbols)}종목): {', '.join(skipped_symbols)}")
        
        except Exception as e:
            logger.error(f"모델 학습 오류: {e}")
    
    def daily_routine(self):
        """일일 루틴: 분석 -> 학습 -> 전략 수립 (매일 아침 9시 5분 실행)"""
        logger.info("=" * 60)
        logger.info("🌅 일일 분석 및 전략 수립 시작 (Daily Routine)")
        logger.info("=" * 60)
        
        # 1. 머신러닝 모델 재학습 (어제 데이터 반영)
        self.train_ml_model()
        
        # 2. 시장 분석 및 전략 자동 업데이트
        self.recommend_strategy(auto_update=True)
        
        # 3. 전략 파라미터 자동 최적화 (매일 아침 갱신)
        self.optimize_strategy_params()
        
        # 4. 동적 설정(K값 등) 다시 로드
        self.load_dynamic_config()
        
        # 5. 전략 성과 리포트 생성 및 알림
        if self.report_manager:
            self.report_manager.generate_daily_report("BTC/KRW")
            self.report_manager.report_portfolio_status(self.crypto_portfolio, "UPBIT", api=self.crypto_api)
            if getattr(self, 'binance_spot_portfolio', None):
                self.report_manager.report_portfolio_status(self.binance_spot_portfolio, "BINANCE SPOT", api=self.binance_spot_api)
            if getattr(self, 'binance_futures_portfolio', None):
                self.report_manager.report_portfolio_status(self.binance_futures_portfolio, "BINANCE FUTURES", api=self.binance_futures_api)
        
        # 6. 일별 포트폴리오 상태 저장 (MDD 계산용)
        self._update_daily_portfolio_history()
        
        logger.info("=" * 60)
        logger.info("✅ 일일 루틴 완료. 최적화된 전략으로 매매를 지속합니다.")
        logger.info("=" * 60)

    def _update_daily_portfolio_history(self):
        """모든 포트폴리오의 일별 상태 업데이트"""
        try:
            # Crypto
            if self.crypto_api:
                prices = {}
                for sym in self.crypto_portfolio.positions:
                    prices[sym] = self.crypto_api.get_price(sym)
                self.crypto_portfolio.update_daily_status(prices)
                self.crypto_portfolio.save_state("data/crypto_portfolio.json")
                
            # Binance Spot
            if getattr(self, 'binance_spot_api', None):
                prices = {}
                for sym in self.binance_spot_portfolio.positions:
                    prices[sym] = self.binance_spot_api.get_price(sym)
                self.binance_spot_portfolio.update_daily_status(prices)
                self.binance_spot_portfolio.save_state("data/binance_spot_portfolio.json")
                
            # Binance Futures
            if getattr(self, 'binance_futures_api', None):
                prices = {}
                for sym in self.binance_futures_portfolio.positions:
                    prices[sym] = self.binance_futures_api.get_price(sym)
                self.binance_futures_portfolio.update_daily_status(prices)
                self.binance_futures_portfolio.save_state("data/binance_futures_portfolio.json")
            
            logger.info("📅 일별 포트폴리오 히스토리 업데이트 완료")
        except Exception as e:
            logger.error(f"일별 히스토리 업데이트 오류: {e}")

    def cancel_old_orders(self):
        """오래된 미체결 주문 취소 (지정가 주문 미체결 대비)"""
        if not self.crypto_api:
            return

        try:
            # 타임아웃 설정 (기본 300초, .env에서 CRYPTO_CANCEL_TIMEOUT으로 변경 가능)
            TIMEOUT_SECONDS = TRADING_CONFIG["crypto"].get("cancel_timeout", 300)
            current_timestamp = time.time() * 1000  # 현재 시간 (밀리초)

            # 모니터링 대상 심볼 (관심 종목 + 보유 종목)
            target_symbols = set(self.crypto_symbols) | set(self.crypto_portfolio.positions.keys())
            
            for symbol in target_symbols:
                # 해당 심볼의 미체결 주문 조회
                open_orders = self.crypto_api.get_open_orders(symbol)
                
                if not open_orders:
                    continue
                
                for order in open_orders:
                    order_id = order.get('id')
                    order_time = order.get('timestamp')  # 주문 생성 시간 (밀리초)
                    order_side = order.get('side')       # buy 또는 sell
                    order_price = order.get('price')
                    
                    if not order_time:
                        continue
                    
                    # 경과 시간 계산 (초 단위)
                    elapsed_seconds = (current_timestamp - order_time) / 1000
                    
                    if elapsed_seconds > TIMEOUT_SECONDS:
                        logger.warning("=" * 60)
                        logger.warning(f"⏳ 오래된 미체결 주문 취소 실행")
                        logger.warning(f"   - 종목: {symbol}")
                        logger.warning(f"   - 주문: {order_side.upper()} @ {order_price:,.0f}")
                        logger.warning(f"   - 경과: {elapsed_seconds:.1f}초 (기준: {TIMEOUT_SECONDS}초)")
                        
                        cancel_result = self.crypto_api.cancel_order(order_id, symbol)
                        
                        if cancel_result:
                            logger.warning(f"   ✅ 주문 취소 성공")
                        else:
                            logger.error(f"   ❌ 주문 취소 실패")
                        logger.warning("=" * 60)

        except Exception as e:
            logger.error(f"미체결 주문 관리 중 오류: {e}")

    def monitor_and_trade(self):
        """모니터링 및 거래 실행"""
        # 락을 사용하여 중복 실행 방지 (APScheduler의 max_instances 경고 회피)
        # blocking=False로 설정하여, 이미 실행 중이면 대기하지 않고 즉시 리턴(스킵)
        if not self.trade_lock.acquire(blocking=False):
            return

        try:
            logger.debug("모니터링 및 거래 실행")
            
            # [Request 3] 웜업 로직 (초기 3회 루프 동안은 매매 제한)
            if not self.is_ready:
                self.warmup_counter += 1
                if self.warmup_counter > 3:
                    self.is_ready = True
                    logger.info("✅ 봇 웜업 완료 (데이터 수집 안정화). 실제 매매를 시작합니다.")
                    self._send_telegram_alert("✅ 웜업 완료! 매매를 시작합니다.")
                else:
                    logger.info(f"⏳ 봇 웜업 및 데이터 수집 중... ({self.warmup_counter}/3)")
            
            # .env 변경 확인 (Hot Reload)
            self.check_env_updates()
            
            # 한국주식 거래
            self._trade_korean_stocks()
            
            # 암호화폐 거래
            # 1. 업비트 (KRW)
            self._trade_upbit()
            
            # 2. 바이낸스 현물 (USDT)
            self._trade_binance_spot()

            # 3. 바이낸스 선물 (USDT)
            self._trade_binance_futures()
        
        except Exception as e:
            # 429 Too Many Requests 또는 IP Ban 처리
            error_msg = str(e)
            if "429" in error_msg or "Too Many Requests" in error_msg or "ban" in error_msg.lower():
                logger.critical(f"🚨 API 호출 한도 초과 또는 차단 감지! 5분간 대기합니다. ({e})")
                self._send_telegram_alert(f"🚨 API 차단 감지! 5분간 봇을 일시 중지합니다.\n오류: {e}")
                time.sleep(300) # 5분 대기
            else:
                logger.error(f"거래 실행 오류: {e}")
        finally:
            self.trade_lock.release()
    
    def _trade_korean_stocks(self):
        """한국주식 거래"""
        try:
            # 여러 API 사용
            apis = [
                ("신한투자", self.shinhan_api),
                ("키움증권", self.kiwoom_api),
                ("대신증권", self.daishin_api),
            ]
            
            # [New] 활성화된 API가 없는 경우 경고 출력 (이동 시 .env 누락 확인용)
            if not any(api for _, api in apis):
                # logger.warning("⚠️ [학습 중단] 연결된 API가 없습니다. .env 파일이 실행 파일과 같은 폴더에 있는지 확인하세요.")
                # 한국 주식 API가 없으면 조용히 리턴 (암호화폐 전용 모드)
                return

            for api_name, api in apis:
                if not api:
                    continue
                
                for symbol in TRADING_CONFIG["korean_stocks"]["symbols"]:
                    # 데이터 수집
                    timeframe = TRADING_CONFIG["korean_stocks"].get("timeframe", "1d")
                    data = api.get_ohlcv(symbol, timeframe)
                    if len(data) == 0:
                        continue
                    
                    current_price = api.get_price(symbol)
                    
                    # 신호 생성
                    signal = self.ml_strategy.generate_signal(symbol, data, self.stock_portfolio.current_capital)
                    
                    if signal and signal.action == "BUY":
                        # 매수
                        quantity = 1  # 실제로는 자본 비율에 따라 계산
                        result = api.buy(symbol, quantity)
                        
                        # 수수료 정보 가져오기
                        buy_fee = TRADING_CONFIG["fees"]["stock_fee_rate"]
                        sell_fee = TRADING_CONFIG["fees"]["stock_fee_rate"] + TRADING_CONFIG["fees"]["stock_tax_rate"]

                        if result:
                            self.stock_portfolio.add_position(symbol, quantity, current_price, fee_rate=buy_fee, atr_value=signal.atr_value if signal else 0.0)
                            self.stock_risk_manager.set_stop_loss(symbol, current_price, atr_value=signal.atr_value if signal else 0.0)
                            # 익절 목표가에 매수+매도 수수료 포함
                            self.stock_risk_manager.set_take_profit(symbol, current_price, fee_rate=buy_fee + sell_fee)
                            self.stock_portfolio.save_state("data/stock_portfolio.json")
                            buy_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            logger.warning("="*70)
                            logger.warning(f"[BUY] [{api_name}] {symbol}")
                            logger.warning(f"시간: {buy_time} | 수량: {quantity}주 | 가격: {current_price:,.0f}원")
                            logger.warning(f"총액: {current_price * quantity:,.0f}원")
                            logger.warning("="*70)
                    
                    # 손실/수익 확인
                    exit_reason = self.stock_risk_manager.check_exit_conditions(symbol, current_price)
                    if exit_reason and symbol in self.stock_portfolio.positions:
                        quantity = self.stock_portfolio.positions[symbol]
                        entry_price = self.stock_portfolio.entry_prices[symbol]
                        
                        # 수수료 + 세금 계산
                        fee_rate = TRADING_CONFIG["fees"]["stock_fee_rate"] + TRADING_CONFIG["fees"]["stock_tax_rate"]
                        result = api.sell(symbol, quantity)
                        
                        if result:
                            # 포트폴리오 업데이트 (수수료 반영)
                            self.stock_portfolio.close_position(symbol, quantity, current_price, fee_rate)
                            
                            # 로그용 단순 계산
                            pnl = ((current_price - entry_price) * quantity) - (current_price * quantity * fee_rate)
                            pnl_percent = (pnl / (entry_price * quantity)) * 100
                            
                            self.stock_risk_manager.remove_position(symbol)
                            self.stock_portfolio.save_state("data/stock_portfolio.json")
                            sell_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            logger.warning("="*70)
                            logger.warning(f"[SELL] [{api_name}] {symbol}")
                            logger.warning(f"시간: {sell_time} | 수량: {quantity}주")
                            logger.warning(f"매입가: {entry_price:,.0f}원 | 매도가: {current_price:,.0f}원")
                            logger.warning(f"손익: {pnl:,.0f}원 ({pnl_percent:+.2f}%) | 사유: {exit_reason}")
                            logger.warning("="*70)
        
        except Exception as e:
            logger.error(f"한국주식 거래 오류: {e}")
    
    def sync_with_exchange(self):
        """거래소 API를 호출해 현재 보유 중인 모든 종목의 실제 평단가와 수량을 가져와서 RiskManager에 강제로 등록"""
        logger.info("🔄 [Sync] 거래소 데이터와 강제 동기화 시작 (RiskManager 등록)...")
        self.sync_wallet()
        logger.info("✅ [Sync] 거래소 동기화 완료.")

    def sync_wallet(self):
        """지갑 동기화 (외부 매매 반영)"""
        if self.crypto_api:
            self._sync_portfolio(self.crypto_api, self.crypto_portfolio, self.crypto_risk_manager, "KRW", "data/crypto_portfolio.json")
            
        if getattr(self, 'binance_spot_api', None):
            try:
                self._sync_portfolio(self.binance_spot_api, self.binance_spot_portfolio, self.binance_spot_risk_manager, "USDT", "data/binance_spot_portfolio.json")
            except Exception as e:
                logger.error(f"⚠️ 바이낸스 현물 동기화 실패: {e}")

        if getattr(self, 'binance_futures_api', None):
            try:
                self._sync_portfolio(self.binance_futures_api, self.binance_futures_portfolio, self.binance_futures_risk_manager, "USDT", "data/binance_futures_portfolio.json")
            except Exception as e:
                # -2015 에러 등 발생 시 봇 중단 방지
                logger.error(f"⚠️ 바이낸스 지갑 동기화 실패 (건너뜀): {e}")

    def _sync_portfolio(self, api, portfolio, risk_manager, currency, save_path):
        """포트폴리오 동기화 공통 로직"""
        try:
            balance = api.get_balance()
            # [요청사항 4] 지갑 동기화 및 자산 업데이트 대기
            time.sleep(0.5)
            cash_free = balance.get("free", {}).get(currency, 0)
            portfolio.current_capital = cash_free
            
            api_positions = api.get_positions()
            api_pos_map = {p['symbol']: p for p in api_positions}
            
            # [New] 최소 주문 금액 기준 가져오기 (Dust 필터링용)
            config_key = "crypto" if currency == "KRW" else "binance_futures" if getattr(api, 'is_future', False) else "binance_spot"
            min_order_amount = TRADING_CONFIG.get(config_key, {}).get("min_order_amount", 0)

            for sym, data in api_pos_map.items():
                qty = data['quantity']
                price = data['entry_price']
                
                should_sync = False
                if sym not in portfolio.positions:
                    # [Fix] 신규 발견 시 Dust(소액) 여부 체크하여 무시
                    # 매매 불가능한 소액 잔고가 계속 재등록되는 것을 방지 (특히 BNB 수수료 잔고)
                    if min_order_amount > 0:
                        current_p = api.get_price(sym)
                        # 가격 조회 성공 시 가치 계산
                        if current_p > 0:
                            val = qty * current_p
                            if val < min_order_amount:
                                logger.debug(f"🧹 [SYNC] {sym} 소액 잔고 무시 (가치: {val:.2f} < 최소: {min_order_amount})")
                                continue

                    logger.warning(f"⚠️ [SYNC_WARNING] {sym} 거래소에는 존재하나 봇 포트폴리오에 없는 종목 발견! (수량: {qty}) -> 장부에 신규 등록합니다.")
                    should_sync = True
                elif abs(portfolio.positions[sym] - qty) > 0.00000001:
                    diff = qty - portfolio.positions[sym]
                    logger.warning(f"⚠️ [SYNC_WARNING] {sym} 수량 불일치 감지! (장부: {portfolio.positions[sym]} vs 실제: {qty}) -> 차이: {diff:+.8f}")
                    logger.warning("   👉 실제 잔고 기준으로 봇의 장부를 강제 업데이트합니다.")
                    should_sync = True
                
                if should_sync:
                    # [Fix] API 평단가가 0인 경우(바이낸스 현물), 기존 장부가가 있으면 유지
                    if price == 0 and sym in portfolio.entry_prices and portfolio.entry_prices[sym] > 0:
                        price = portfolio.entry_prices[sym]
                        logger.debug(f"ℹ️ [SYNC] {sym} API 평단가 0 -> 기존 장부가({price}) 유지")

                    # [Fix] 저장된 ATR 값 복원 (재시작 시 정보 유지)
                    saved_atr = portfolio.atr_values.get(sym, 0.0)
                    
                    if sym not in risk_manager.stop_loss_prices:
                        fee_rate = TRADING_CONFIG["fees"].get("binance_fee_rate" if currency == "USDT" else "crypto_fee_rate", 0.001)
                        
                        # [Fix] Sync 시 ATR 계산하여 동적 손절가 복원 (재시작 후에도 리스크 관리 유지)
                        # 저장된 값이 있으면 우선 사용, 없으면 재계산
                        atr_value = saved_atr
                        
                        try:
                            # 최근 데이터 조회하여 ATR 계산
                            timeframe = TRADING_CONFIG["crypto"].get("timeframe", "15m")
                            # _get_latest_ohlcv는 캐싱되므로 부담 적음
                            df = self._get_latest_ohlcv(api, sym, timeframe)
                            if not df.empty and len(df) >= 20:
                                atr_indicator = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
                                current_atr = atr_indicator.average_true_range().iloc[-1]
                                if atr_value <= 0: # 저장된 값이 없을 때만 현재 값 사용
                                    atr_value = current_atr
                        except Exception:
                            pass

                        risk_manager.set_stop_loss(sym, price, atr_value=atr_value)
                        risk_manager.set_take_profit(sym, price, fee_rate=fee_rate * 2, atr_value=atr_value)
                    
                    portfolio.sync_position(sym, qty, price, atr_value=saved_atr)
            
            for sym in list(portfolio.positions.keys()):
                if sym not in api_pos_map:
                    logger.warning(f"👻 [SYNC_WARNING] {sym} 유령 포지션 감지! (봇 장부에는 있으나 거래소 잔고 없음) -> 장부에서 삭제합니다.")
                    portfolio.remove_position(sym)
                    risk_manager.remove_position(sym)
            
            portfolio.save_state(save_path)
        except Exception as e:
            logger.error(f"{currency} 포트폴리오 동기화 오류: {e}")

    def _calculate_pyramiding_buy(self, symbol, current_price, atr, current_qty):
        """
        피라미딩(불타기) 추가 매수 수량 계산
        조건: 0.5N 상승, 기존 수량의 25%, 최대 4회, 총 리스크 2% 제한
        """
        if not TRADING_CONFIG["crypto"].get("pyramiding_enabled", False):
            return 0.0

        # 포트폴리오에서 피라미딩 상태 가져오기
        info = self.crypto_portfolio.pyramiding_info.get(symbol, {
            'count': 0, 
            'last_entry_price': self.crypto_portfolio.entry_prices.get(symbol, 0)
        })
        
        last_entry = info['last_entry_price']
        count = info['count']
        
        # 1. 횟수 제한 (최대 4회)
        if count >= 4:
            return 0.0
            
        # 2. 가격 상승 조건 (0.5N 상승 시)
        # N = ATR (Signal에서 전달받음)
        if not atr or current_price < last_entry + (0.5 * atr):
            return 0.0
            
        # 3. 수량 계산 (기존 수량의 25%)
        add_qty = current_qty * 0.25
        
        # 4. 리스크 관리 (총 리스크 <= 자산의 2%)
        # 총 리스크 = (총 수량) * (2 * ATR)  <-- 2N 손절 기준
        # 자산 = 초기 자본금 기준 (보수적 접근)
        total_equity = TRADING_CONFIG["crypto"]["initial_capital"]
        max_risk = total_equity * 0.02
        
        new_total_qty = current_qty + add_qty
        current_risk = new_total_qty * (2 * atr)
        
        if current_risk > max_risk:
            # 리스크 초과 시 매수 불가 (엄격한 리스크 관리)
            return 0.0
                
        return add_qty

    def _get_latest_ohlcv(self, api, symbol: str, timeframe: str, current_price: float = None) -> pd.DataFrame:
        """
        OHLCV 데이터 조회 (캐싱 + 웹소켓 실시간 업데이트)
        REST API 호출 빈도를 줄이고, 웹소켓 현재가를 반영하여 최신 상태 유지
        """
        current_time = time.time()
        
        # [최적화] 전략별/타임프레임별 갱신 주기 차별화
        # 1분봉 등 단기는 60초, 그 외는 180초
        is_short_term = timeframe in ["1m", "3m", "5m"]
        fetch_interval = 60 if is_short_term else 180
        
        # 1. 캐시 유효성 확인 및 REST API 호출
        if (symbol not in self.ohlcv_cache or 
            current_time - self.last_ohlcv_fetch.get(symbol, 0) > fetch_interval):
            
            # API 호출 전 미세 지연 (429 에러 방지)
            time.sleep(0.2)
            df = api.get_ohlcv(symbol, timeframe)
            if not df.empty:
                self.ohlcv_cache[symbol] = df
                self.last_ohlcv_fetch[symbol] = current_time
        
        # 2. 캐시된 데이터 가져오기
        df = self.ohlcv_cache.get(symbol)
        if df is None or df.empty:
            return pd.DataFrame()
            
        # 3. 웹소켓 실시간 가격 반영 (메모리 상에서만 업데이트)
        if current_price is None or current_price <= 0:
            current_price = api.get_price(symbol)
            
        if current_price and current_price > 0:
            df = df.copy() # 원본 보존
            df.iloc[-1, df.columns.get_loc('close')] = current_price
            if current_price > df.iloc[-1]['high']: df.iloc[-1, df.columns.get_loc('high')] = current_price
            if current_price < df.iloc[-1]['low']: df.iloc[-1, df.columns.get_loc('low')] = current_price
                
        return df

    def _execute_sell(self, api, portfolio, risk_manager, symbol, current_price, exit_reason, fee_rate, save_path, quantity=None):
        """매도 실행 공통 로직 (정기 매매 & 실시간 매매 공용)"""
        try:
            # [New] 거래소 이름 식별
            exchange_name = "UPBIT" if isinstance(api, UpbitAPI) else "BINANCE_FUTURES" if getattr(api, 'is_future', False) else "BINANCE_SPOT"

            if quantity is None:
                quantity = portfolio.positions.get(symbol, 0)
            if quantity <= 0:
                return

            entry_price = portfolio.entry_prices.get(symbol, 0)
            
            # 매도 가능 최소 금액 체크 (업비트 5,000원)
            current_value = quantity * current_price
            # 바이낸스는 10달러 등 다름. 설정값 참조
            min_order = 5000 if "KRW" in symbol else 10
            if current_value < min_order:
                logger.warning(f"[{exchange_name}] ⚠️ 매도 금액({current_value:,.0f})이 최소 주문 금액({min_order}) 미만입니다. 매도 불가.")
                # [Fix] 로그 포맷 개선 (소수점 표시 및 메시지 명확화)
                logger.warning(f"[{exchange_name}] ⚠️ [DUST] {symbol} 잔고 가치({current_value:.2f})가 최소 주문 금액({min_order}) 미만입니다. 매도 불가.")
                
                # [New] 바이낸스 현물인 경우 소액 잔고(Dust)를 BNB로 변환 시도
                if isinstance(api, BinanceAPI) and not getattr(api, 'is_future', False):
                    try:
                        target_coin = symbol.split('/')[0]
                        # [Fix] BNB는 BNB로 변환할 수 없으므로 제외
                        if target_coin != "BNB":
                            logger.info(f"🧹 [BINANCE] {target_coin} 소액 잔고(Dust) BNB 변환 시도...")
                            api.convert_dust_to_bnb([target_coin])
                            # [Fix] 중복 호출 제거 (아래에서 result를 받아서 처리)
                            result = api.convert_dust_to_bnb([target_coin])
                            if result and result.get('totalTransfered'):
                                bnb_amount = result.get('totalTransfered')
                                self._send_telegram_alert(f"🧹 [DUST] {target_coin} 소액 잔고가 BNB로 변환되었습니다.\n변환 수량: {bnb_amount} BNB")
                    except Exception as e:
                        logger.error(f"BNB 변환 실패: {e}")

                # [Fix] 최소 금액 미만인 경우 무한 루프 방지를 위해 포트폴리오에서 제거 (Dust 처리)
                logger.warning(f"[{exchange_name}] 🗑️ 거래 불가능한 소액 잔고(Dust)로 판단하여 봇 관리 목록에서 제외합니다: {symbol}")
                portfolio.remove_position(symbol)
                risk_manager.remove_position(symbol)
                portfolio.save_state(save_path)
                self._send_telegram_alert(f"⚠️ [DUST] {symbol} 잔고가 최소 주문 금액 미만({current_value:.2f} < {min_order})이라 매도할 수 없습니다. 봇 관리에서 제외합니다.")
                # [Fix] 반복적인 Dust 경고 알림은 로그로만 남기고 텔레그램 전송 중단 (알림 폭탄 방지)
                return

            # [수정] 손절 여부 확인
            is_stop_loss = "stop_loss" in str(exit_reason).lower() or "손절" in str(exit_reason)
            
            # [요청사항 3] 매도 시도 로그 (블랙박스형)
            # 가격은 API 내부에서 결정되므로 current_price로 로깅
            logger.info(f"[{exchange_name}] [SELL_TRY] 종목: {symbol}, 사유: {exit_reason}, 기준가: {current_price:,.0f}, 수량: {quantity}, 급격한손절: {is_stop_loss}")
            
            # price=None, is_stop_loss 전달 -> 공격적 지정가 또는 시장가(손절시) 실행
            result = api.sell(symbol, quantity, price=None, is_stop_loss=is_stop_loss)
            
            if result:
                portfolio.close_position(symbol, quantity, current_price, fee_rate)
                
                pnl = ((current_price - entry_price) * quantity) - (current_price * quantity * fee_rate)
                
                # [Fix] Zero Division Error 방지
                denom = entry_price * quantity
                pnl_percent = (pnl / denom) * 100 if denom > 0 else 0.0
                
                # [요청사항 3] 매도 성공 로그 (실제 체결가 및 수익률)
                # API 결과에 average(평균체결가)가 있으면 사용, 없으면 price 사용
                avg_price = result.get('average') or result.get('price') or current_price
                
                # [New] 슬리피지 경고 (0.5% 이상 차이 발생 시)
                try:
                    exec_price = float(avg_price)
                    if current_price > 0:
                        slippage = (current_price - exec_price) / current_price * 100
                        if abs(slippage) >= 0.5:
                            warn_msg = f"[{exchange_name}] ⚠️ [SLIPPAGE] {symbol} 매도 체결가 괴리 경고!\n기준: {current_price:,.0f} -> 체결: {exec_price:,.0f} ({slippage:+.2f}%)"
                            logger.warning(warn_msg.replace("\n", " "))
                            self._send_telegram_alert(warn_msg)
                except Exception as e:
                    logger.warning(f"슬리피지 체크 오류: {e}")

                logger.info(f"[{exchange_name}] [PROFIT_REPORT] 실제체결가: {avg_price:,.0f}, 실질수익률: {pnl_percent:+.2f}%, 손익금액: {pnl:+.0f}")

                # [로그 상세화] 매도 사유 태그 생성
                tag = "[매도]"
                reason_lower = str(exit_reason).lower()
                if "emergency" in reason_lower:
                    tag = "🚨 [긴급매도]"
                elif "stop_loss" in reason_lower or "손절" in reason_lower:
                    tag = "[손절실행]"
                elif "take_profit" in reason_lower or "익절" in reason_lower:
                    tag = "[수익확정]"
                elif "trailing_stop" in reason_lower:
                    tag = "[수익확정(TS)]"
                elif "break-even" in reason_lower or "본절" in reason_lower:
                    tag = "[본절보존]"

                # [요청사항 5] 바이낸스 선물 레버리지 정보 추가
                leverage = None
                liq_info = ""
                if getattr(api, 'is_future', False):
                    leverage = TRADING_CONFIG["binance_futures"].get("leverage", 1)
                    # [New] 청산 위험도 정보 조회 (바이낸스 API인 경우)
                    if isinstance(api, BinanceAPI):
                        risk_data = api.get_liquidation_risk(symbol)
                        if risk_data:
                            dist_pct = risk_data.get('distance_pct', 0) * 100
                            liq_price = risk_data.get('liquidation_price', 0)
                            liq_info = f" | 청산가: {liq_price:,.4f} (거리: {dist_pct:.2f}%)"

                risk_manager.remove_position(symbol)
                portfolio.save_state(save_path)
                sell_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.warning("="*70)
                logger.warning(f"[{exchange_name}] {tag} [{symbol.split('/')[1]}] {symbol}")
                logger.warning(f"시간: {sell_time} | 수량: {quantity}")
                logger.warning(f"매입가: {entry_price:,.0f}원 | 매도가: {current_price:,.0f}원")
                logger.warning(f"실현손익: {pnl:,.0f}원 (수익률: {pnl_percent:+.2f}%) | 사유: {exit_reason}{liq_info}")
                logger.warning("="*70)

                # [New] 텔레그램 알림 전송
                if self.report_manager:
                    # Telegram Markdown 파싱 에러 방지 (stop_loss -> stop-loss)
                    safe_reason = str(exit_reason).replace("_", "-")
                    
                    self.report_manager.send_trade_alert(
                        symbol, "SELL", current_price, quantity, pnl, pnl_percent, safe_reason,
                        leverage=leverage
                    )
        except Exception as e:
            # [요청사항 4] 봇 중단 방지 및 텔레그램 알림
            logger.error(f"{symbol} 매도 실행 중 치명적 오류: {e}")
            self._send_telegram_alert(f"🚨 {symbol} 매도 실패! 봇 점검 필요.\n오류: {e}")

    def _on_realtime_price(self, symbol: str, current_price: float):
        """실시간 가격 업데이트 콜백 (RiskManager 즉시 체크)"""
        # [New] 급등락 감지 로직 실행
        self._check_price_volatility(symbol, current_price)

        # 보유 종목에 대해서만 리스크 관리 체크
        if symbol in self.crypto_portfolio.positions:
            # 락 획득 시도 (메인 루프와 충돌 방지, blocking=False로 대기 없이 스킵)
            if not self.trade_lock.acquire(blocking=False):
                return

            try:
                exit_reason = self.crypto_risk_manager.check_exit_conditions(symbol, current_price)
                if exit_reason:
                    logger.info(f"⚡ [WebSocket] {symbol} 즉각 매도 신호 감지: {exit_reason}")
                    self._execute_sell(self.crypto_api, self.crypto_portfolio, self.crypto_risk_manager, symbol, current_price, exit_reason, TRADING_CONFIG["fees"]["crypto_fee_rate"], "data/crypto_portfolio.json")
            finally:
                self.trade_lock.release()

    def _check_price_volatility(self, symbol: str, current_price: float):
        """실시간 급등락 감지 (3분 내 3% 이상 변동 시 알림)"""
        try:
            now = time.time()
            
            if symbol not in self.volatility_monitor:
                self.volatility_monitor[symbol] = {
                    'base_price': current_price,
                    'base_time': now,
                    'last_alert_time': 0
                }
                return

            data = self.volatility_monitor[symbol]
            
            # 기준 시간(3분) 경과 시 기준가 리셋 (완만한 변동은 무시)
            if now - data['base_time'] > 180:
                data['base_price'] = current_price
                data['base_time'] = now
                return

            # 변동률 계산
            if data['base_price'] > 0:
                change_pct = (current_price - data['base_price']) / data['base_price'] * 100
                
                # 알림 조건: 3% 이상 변동 AND 쿨타임 10분(600초)
                if abs(change_pct) >= 3.0:
                    if now - data['last_alert_time'] > 600:
                        emoji = "🚀" if change_pct > 0 else "📉"
                        direction = "급등" if change_pct > 0 else "급락"
                        
                        msg = f"{emoji} [{symbol}] 가격 {direction} 경고!\n"
                        msg += f"현재가: {current_price:,.0f} ({change_pct:+.2f}%)\n"
                        msg += f"(기준가: {data['base_price']:,.0f} / 3분 내)"
                        
                        self._send_telegram_alert(msg)
                        
                        # 알림 후 상태 업데이트 (연속 알림 방지)
                        data['last_alert_time'] = now
                        data['base_price'] = current_price
                        data['base_time'] = now
        except Exception as e:
            logger.error(f"급등락 체크 오류: {e}")

    def _trade_upbit(self):
        """업비트 거래 (KRW)"""
        if not self.crypto_api: return
        self._process_crypto_trading(
            self.crypto_api, 
            self.crypto_portfolio, 
            self.crypto_risk_manager, 
            self.crypto_symbols, 
            "crypto", 
            "data/crypto_portfolio.json"
        )

    def _trade_binance_spot(self):
        """바이낸스 현물 거래"""
        if not getattr(self, 'binance_spot_api', None): return
        self._process_crypto_trading(
            self.binance_spot_api, 
            self.binance_spot_portfolio, 
            self.binance_spot_risk_manager, 
            self.binance_spot_symbols, 
            "binance_spot", 
            "data/binance_spot_portfolio.json"
        )

    def _trade_binance_futures(self):
        """바이낸스 선물 거래"""
        if not getattr(self, 'binance_futures_api', None): return
        self._process_crypto_trading(
            self.binance_futures_api, 
            self.binance_futures_portfolio, 
            self.binance_futures_risk_manager, 
            self.binance_futures_symbols, 
            "binance_futures", 
            "data/binance_futures_portfolio.json"
        )
            
        # [New] 선물 모드일 경우 청산 리스크 추가 점검
        if getattr(self, 'binance_futures_api', None):
            for symbol in self.binance_futures_portfolio.positions.keys():
                self._check_liquidation_safety(symbol)

    def _process_crypto_trading(self, api, portfolio, risk_manager, symbols, config_key, save_path):
        """암호화폐 거래 공통 로직"""
        try:
            exchange_name = config_key.upper()

            # 거래량 기반 종목 자동 업데이트 (1시간마다)
            if config_key == "crypto": # 업비트만 자동 업데이트 지원
                self.update_crypto_symbols()
            
            # [Phase 1] 보유 종목 관리 (매도/손절/OCO) - 별도 루프 (안전성 강화)
            # 매수 로직과 분리하여, 매수 중 에러가 발생해도 보유 종목 관리는 멈추지 않도록 함
            current_positions = list(portfolio.positions.keys())
            if current_positions:
                logger.debug(f"[{exchange_name}] 🛡️ 보유 종목 관리 시작 ({len(current_positions)}개): {current_positions}")

            for symbol in current_positions:
                try:
                    # [New] OCO 주문 감시 모드 확인 (바이낸스 현물)
                    if config_key == "binance_spot" and symbol in self.oco_monitoring_symbols:
                        # 미체결 주문 확인 (주문이 없으면 체결되었거나 취소된 것)
                        # 미체결 주문 확인 (주문이 없으면 체결되었거나 취소된 것) - API 호출 1회
                        open_orders = api.get_open_orders(symbol)
                        if not open_orders:
                            logger.info(f"[{exchange_name}] 🔓 {symbol} OCO 주문 종료(체결/취소) -> 실시간 감시 재개")
                            self.oco_monitoring_symbols.remove(symbol)
                            
                            # 잔액 확인하여 매도 여부 판단
                            try:
                                balance = api.get_balance()
                                target_coin = symbol.split('/')[0]
                                available_qty = float(balance.get('free', {}).get(target_coin, 0.0))
                                
                                # 보유량이 포트폴리오 수량의 10% 미만이면 전량 매도된 것으로 간주 (먼지 고려)
                                pf_qty = portfolio.positions.get(symbol, 0)
                                if pf_qty > 0 and available_qty < (pf_qty * 0.1):
                                    logger.info(f"[{exchange_name}] ✅ {symbol} OCO 매도 체결 확인 -> 포트폴리오 정리")
                                    portfolio.remove_position(symbol)
                                    risk_manager.remove_position(symbol)
                                    portfolio.save_state(save_path)
                                    continue # 루프 종료 (더 이상 보유 종목 아님)
                                else:
                                    logger.info(f"[{exchange_name}] ⚠️ {symbol} OCO 주문 취소됨 (잔고 보유) -> 실시간 감시로 전환")
                            except Exception as e:
                                logger.error(f"OCO 상태 확인 중 오류: {e}")
                        else:
                            # OCO 대기 중이면 봇의 매도 로직 스킵 (서버가 관리함)
                            continue

                    # 1. 현재가 조회 (가장 먼저 수행하여 매도 판단 속도 향상)
                    current_price = api.get_price(symbol)
                    
                    # [Fallback] 웹소켓 지연 등으로 현재가가 0이면 REST API로 재조회
                    if current_price == 0:
                        try:
                            ticker = api.get_ticker(symbol)
                            current_price = float(ticker.get('last', 0))
                            if current_price > 0:
                                logger.info(f"[{exchange_name}] ⚠️ {symbol} 웹소켓 가격 0 -> REST API Fallback 성공: {current_price}")
                        except Exception as e:
                            logger.warning(f"[{exchange_name}] {symbol} 가격 조회 Fallback 실패: {e}")

                    if current_price == 0:
                        continue

                    # [New] 비상 손절 체크 (ATR 정보 부재 시 안전장치)
                    # 다른 PC에서 실행 시 ATR 정보가 없을 수 있으므로, 평단가 대비 -10% 하락 시 무조건 시장가 매도
                    entry_price = portfolio.entry_prices.get(symbol, 0)
                    
                    # [Fix] 평단가 0원일 경우 현재가로 대체 (Zero Division 방지)
                    if entry_price <= 0:
                        logger.warning(f"[{exchange_name}] ⚠️ {symbol} 평단가 0원 감지 -> 현재가({current_price})로 보정")
                        entry_price = current_price
                        portfolio.entry_prices[symbol] = entry_price
                        if symbol in risk_manager.entry_prices:
                            risk_manager.entry_prices[symbol] = entry_price
                        portfolio.save_state(save_path)

                    if entry_price > 0 and current_price < (entry_price * 0.95):
                        exit_reason = "Emergency Stop Loss (Hard Limit -5%)"
                        logger.warning(f"🚨 {symbol} 비상 손절 발동! (현재가 {current_price:,.0f} < 평단가 {entry_price:,.0f}의 95%)")
                        self._send_telegram_alert(f"🚨 [긴급 매도] {symbol} 비상 손절(-5%) 발동!\n현재가: {current_price:,.0f}원")
                        
                        fee = TRADING_CONFIG["fees"]["binance_fee_rate"] if "binance" in config_key else TRADING_CONFIG["fees"]["crypto_fee_rate"]
                        self._execute_sell(api, portfolio, risk_manager, symbol, current_price, exit_reason, fee, save_path)
                        continue

                    # 2. 매도 조건 확인 (보유 중일 경우)
                    # 2-1. 리스크 관리 (손절/익절) 확인
                    exit_reason = risk_manager.check_exit_conditions(symbol, current_price)
                    
                    # 2-2. 전략적 매도 신호 확인 (이미 리스크 관리로 매도 결정된 경우 건너뜀)
                    if not exit_reason:
                        timeframe = TRADING_CONFIG[config_key].get("timeframe", "1d")
                        data = api.get_ohlcv(symbol, timeframe) # 캐싱 미적용 (간소화)
                        
                        if not data.empty:
                            # [New] 전략에 포지션 정보 전달 (동적 청산용)
                            pos_qty = portfolio.positions.get(symbol, 0)
                            entry_prc = portfolio.entry_prices.get(symbol, 0)
                            
                            signal = self.strategies[config_key].generate_signal(
                                symbol, data, portfolio.current_capital, 
                                entry_price=entry_prc, position_quantity=pos_qty
                            )
                            if signal and signal.action == "SELL":
                                exit_reason = f"전략 매도 신호 ({signal.reason})"
                                # [New] 부분 매도 지원
                                if signal.suggested_quantity > 0:
                                    sell_qty = signal.suggested_quantity
                                    logger.info(f"[{exchange_name}] ⚖️ 전략적 부분 매도 실행: {symbol} {sell_qty}개 (보유량의 {sell_qty/pos_qty*100:.1f}%)")
                                    fee = TRADING_CONFIG["fees"]["binance_fee_rate"] if "binance" in config_key else TRADING_CONFIG["fees"]["crypto_fee_rate"]
                                    self._execute_sell(api, portfolio, risk_manager, symbol, current_price, exit_reason, fee, save_path, quantity=sell_qty)
                                    continue # 부분 매도 후 루프 계속 (전량 매도 아님)

                    if exit_reason:
                        fee = TRADING_CONFIG["fees"]["binance_fee_rate"] if "binance" in config_key else TRADING_CONFIG["fees"]["crypto_fee_rate"]
                        self._execute_sell(api, portfolio, risk_manager, symbol, current_price, exit_reason, fee, save_path)

                except Exception as e:
                    logger.error(f"[{exchange_name}] 🚨 보유 종목({symbol}) 관리 중 오류: {e}")
                    continue

            # [Phase 2] 신규 진입 (매수) - 별도 루프
            # 관심 종목 전체 대상 (보유 종목 포함하여 피라미딩 기회 포착)
            target_symbols = symbols
            
            for symbol in target_symbols:
                try:
                    # 1. 현재가 조회
                    current_price = api.get_price(symbol)
                    if current_price == 0: continue

                    # 보유 여부 확인
                    is_holding = symbol in portfolio.positions

                    # 2. 진입 여부 판단
                    # 신규 진입인데 최대 보유 종목 수 꽉 찼으면 스킵
                    if not is_holding and len(portfolio.positions) >= TRADING_CONFIG[config_key].get("max_positions", 5):
                        continue
                    
                    # [전략 및 타임프레임 분리]
                    # 비트코인: 4시간봉 (중기 추세)
                    # 알트코인: Breakout 전략 (설정된 타임프레임, 보통 15m/1h)
                    if "BTC" in symbol: # BTC/KRW or BTC/USDT
                        target_timeframe = "4h"
                        target_strategy = "breakout" # 비트도 돌파 매매 사용 (안정적)
                    else:
                        target_timeframe = TRADING_CONFIG[config_key].get("timeframe", "15m")
                        target_strategy = TRADING_CONFIG[config_key].get("entry_strategy", "breakout") # 설정된 전략 사용
                    
                    # [MTF 필터] 알트코인 매매 시(15m 등), 1시간봉 EMA 50 위에서만 매수 (대세 상승장 확인)
                    if "BTC" not in symbol and target_timeframe in ["15m", "5m", "1m"]:
                        try:
                            # 1시간봉 데이터 조회 (최근 200개)
                            df_1h = api.get_ohlcv(symbol, "1h")
                            if not df_1h.empty and len(df_1h) >= 50:
                                ema50_1h = df_1h['close'].ewm(span=50, adjust=False).mean().iloc[-1]
                                if current_price < ema50_1h * 0.95:
                                    logger.debug(f"🚫 {symbol} 1시간봉 EMA50의 95%({ema50_1h*0.95:,.0f}) 아래(하락세) -> 매수 스킵")
                                    continue
                        except Exception as e:
                            logger.warning(f"MTF 필터 체크 중 오류: {e}")
                            
                    # 데이터 수집 (시간이 걸리므로 매도 체크 후에 수행)
                    # [Request: Rate Limit] 종목별 수집 간 0.2초 딜레이
                    time.sleep(0.2)
                    
                    # [Request 1] 초기 데이터 수집량 설정
                    # MA Trend 전략은 4시간봉 리샘플링을 위해 많은 데이터(3000개) 필요
                    limit_count = 200
                    if target_strategy == "ma_trend":
                        limit_count = 3000
                        
                    data = api.get_ohlcv(symbol, target_timeframe, limit=limit_count)
                    
                    # [Request: Data Integrity] 200개 요청했으나 100개 이상이면 전략 실행 허용
                    min_required = 100
                    if len(data) < min_required:
                        logger.info(f"[{exchange_name}] [SAFE_WAIT] {symbol}: 데이터 부족/타임아웃으로 매매 대기 (수신: {len(data)}개 / 최소: {min_required}개)")
                        continue
                    
                    # [Request 3] 웜업 로직 동기화 - 데이터 로드 확인 로그
                    if not self.is_ready and len(data) >= 200:
                        logger.info(f"[{exchange_name}] [{symbol}] 웜업 데이터 로드 완료: {len(data)}개 (목표: 200)")
                    
                    # 신호 생성
                    signal = self.strategies[config_key].generate_signal(
                        symbol, 
                        data, 
                        portfolio.current_capital,
                        strategy_override=target_strategy
                    )
                    # [수정] ATR NoneType 방지 및 0.0 처리 (None 비교 에러 방지)
                    atr = signal.atr_value if signal and signal.atr_value is not None else 0.0
                    
                    # [Fix] 전략에서 ATR을 반환하지 않았더라도 데이터가 충분하면 직접 계산 (AgileStrategy 등 대응)
                    if atr <= 0 and len(data) >= 20:
                        try:
                            atr_indicator = AverageTrueRange(data['high'], data['low'], data['close'], window=14)
                            atr = atr_indicator.average_true_range().iloc[-1]
                            if signal:
                                signal.atr_value = atr # 신호 객체에도 업데이트
                        except Exception:
                            pass # 계산 실패 시 0.0 유지 (아래 로직에서 기본값 적용)

                    # [로그 가시성] 진입 보류 시 이유 출력
                    if signal and signal.action == "HOLD":
                        logger.debug(f"[{exchange_name}] 🚫 {symbol} 진입 보류: {signal.reason}")
                    
                    buy_amount = 0.0
                    is_pyramiding = False
                    
                    # 3-2. 매수 수량 및 조건 계산
                    if not is_holding:
                        # [신규 진입]
                        if signal and signal.action == "BUY":
                            # [검증] 가격 유효성 체크
                            if current_price is None or current_price <= 0:
                                logger.warning(f"[{exchange_name}] ⚠️ {symbol} 현재가 오류({current_price}) -> 매수 스킵")
                                continue

                            # 사용 가능한 자본을 기준으로 매수 금액 계산
                            # 전략에서 계산된 수량(터틀 유닛)이 있으면 우선 사용
                            
                            # [안전장치] suggested_quantity가 None일 경우 0.0 처리
                            suggested_qty = signal.suggested_quantity if signal.suggested_quantity is not None else 0.0
                            if suggested_qty > 0:
                                buy_amount = suggested_qty * current_price
                            else:
                                # [Request 2] 0값 방어 로직 (주문 계산 중단)
                                if atr <= 0:
                                    # [New] ATR 데이터 부족 시 기본값(1%) 할당 및 로그 스로틀링
                                    atr = current_price * 0.01
                                    now = time.time()
                                    if now - self.last_log_time.get(f"{symbol}_atr", 0) > 60:
                                        logger.info(f"[{exchange_name}] [WAIT] {symbol}: ATR 지표 부족 -> 기본값(1%) 적용 (ATR: {atr:.2f})")
                                        self.last_log_time[f"{symbol}_atr"] = now

                                capital = portfolio.current_capital
                                buy_amount = capital * TRADING_CONFIG[config_key].get("max_position_size", 0.1)
                    else:
                        # [피라미딩 진입]
                        current_qty = portfolio.positions[symbol]
                        # 피라미딩 로직은 현재 crypto_portfolio에 의존적임. 바이낸스용으로 확장 필요하나 일단 스킵하거나 공통화 필요.
                        # 여기서는 간단히 스킵 (바이낸스 피라미딩은 추후 구현)
                        if config_key == "crypto":
                            # ATR 유효성 검증 (피라미딩은 ATR 필수)
                            if atr is None or atr <= 0:
                                logger.debug(f"[{exchange_name}] ⚠️ {symbol} 피라미딩 스킵: ATR 값 없음({atr})")
                                add_qty = 0.0
                            else:
                                add_qty = self._calculate_pyramiding_buy(symbol, current_price, atr, current_qty)
                            
                            if add_qty > 0:
                                buy_amount = add_qty * current_price
                                is_pyramiding = True

                    # 3-3. 매수 실행
                    if buy_amount > 0:
                        # [Request 3] 웜업 상태 체크 (매수 차단)
                        if not self.is_ready:
                            logger.info(f"[{exchange_name}] 🛡️ [WARMUP] {symbol} 매수 신호 감지되었으나 웜업 중이라 주문을 생략합니다.")
                            continue

                        # 매수 금액 계산 (설정값 기반)
                        # 최소 주문 금액 보정 (업비트 최소 5,000원)
                        # 매도 시 수수료 및 가격 하락을 고려하여 6,000원 이상으로 설정 (안전마진 확보)
                        min_order_amount = TRADING_CONFIG[config_key].get("min_order_amount", 5000)
                        safe_min_amount = min_order_amount * 1.1 # 10% 여유
                        
                        if buy_amount < safe_min_amount:
                            buy_amount = safe_min_amount
                        
                        # [New] 매수 진입 전 해당 종목의 미체결 주문 정리 (설계 초기화)
                        api.cancel_all_orders(symbol)
                        
                        # 잔액 확인 후 매수 시도 (에러 핸들링 강화)
                        try:
                            balance = api.get_balance()
                            
                            # [안전장치] 잔액 조회 실패 시 스킵
                            if balance is None:
                                logger.warning(f"[{exchange_name}] ⚠️ {symbol} 잔액 조회 실패(None) -> 매수 스킵")
                                continue
                                
                            currency = "KRW" if config_key == "crypto" else "USDT"
                            available_cash = balance.get("free", {}).get(currency, 0)
                            
                            # [안전장치] available_cash None 및 숫자 검증
                            if available_cash is None:
                                logger.warning(f"[{exchange_name}] ⚠️ {symbol} 가용 현금({currency}) 데이터 없음 -> 매수 스킵")
                                continue
                            
                            try:
                                available_cash = float(available_cash)
                            except (ValueError, TypeError):
                                logger.error(f"[{exchange_name}] ⚠️ {symbol} 가용 현금 데이터 형식 오류: {available_cash}")
                                continue

                            # 1. 잔액 체크
                            if available_cash < buy_amount:
                                logger.info(f"[{exchange_name}] 매수 대기: 잔액 부족 ({symbol}, 가용: {available_cash:,.0f}, 필요: {buy_amount:,.0f})")
                                
                                # [New] 예수금 부족 시 전체 미체결 매수 주문 취소하여 현금 확보
                                logger.info(f"[{exchange_name}] 💰 가용 현금 확보를 위해 타 종목 미체결 매수 주문 취소 시도...")
                                cancelled = api.cancel_all_orders(None, side='buy')
                                
                                if cancelled > 0:
                                    time.sleep(0.5) # 잔액 반영 대기
                                    balance = api.get_balance()
                                    available_cash = balance.get("free", {}).get(currency, 0)
                                    logger.info(f"[{exchange_name}] ✨ 미체결 취소 후 가용 현금: {available_cash:,.0f}")
                                
                                if available_cash < buy_amount:
                                    continue
                            
                            # 2. 매수 시도
                            if available_cash >= buy_amount:
                                # [요청사항 2] 예수금 99.5% 사용 안전장치 (수수료/오차 버퍼)
                                safe_limit = available_cash * 0.995
                                if buy_amount > safe_limit:
                                    buy_amount = safe_limit
                                
                                # [수정] 공격적 지정가 매수 (crypto_api 내부에서 처리)
                                # 수량 계산을 위한 참고용 가격
                                ticker = api.get_ticker(symbol)
                                ask_price = ticker.get('ask') or current_price
                                
                                # [방어 코드] 가격 0 체크 (Division by Zero 방지)
                                if ask_price <= 0:
                                    logger.debug(f"[{exchange_name}] ⚠️ {symbol} 매수 가격(ask_price)이 0입니다. 매수 스킵.")
                                    continue
                                
                                # 수량 계산: (매수금액) / (가격 * (1 + 수수료율))
                                fee_rate = TRADING_CONFIG["fees"]["binance_fee_rate"] if "binance" in config_key else TRADING_CONFIG["fees"]["crypto_fee_rate"]
                                
                                denominator = ask_price * (1 + fee_rate)
                                if denominator == 0:
                                    logger.debug(f"[{exchange_name}] ⚠️ {symbol} 수량 계산 분모가 0입니다. 매수 스킵.")
                                    continue
                                
                                buy_qty = buy_amount / denominator
                                buy_qty = float(f"{buy_qty:.8f}") # 소수점 8자리 제한 (API 오류 방지)
                                
                                # [New] 레버리지 정보 추출
                                leverage = signal.suggested_leverage if signal else 1
                                
                                # [Request] 바이낸스 선물 동적 레버리지 적용 (주문 직전 계산 및 반영)
                                if config_key == "binance_futures":
                                    try:
                                        # ATR(14) 및 장기 평균 ATR(100) 계산 (데이터는 이미 200개 확보됨)
                                        if len(data) >= 114:
                                            atr_indicator = AverageTrueRange(data['high'], data['low'], data['close'], window=14)
                                            atr_series = atr_indicator.average_true_range()
                                            atr_current = atr_series.iloc[-1]
                                            atr_avg = atr_series.tail(100).mean()
                                            
                                            base_lev = TRADING_CONFIG["binance_futures"].get("leverage", 1)
                                            # 설정에 없으면 기본 20배 제한
                                            max_lev_limit = TRADING_CONFIG["binance_futures"].get("max_leverage_limit", 20)
                                            
                                            # 전봉 종가 (Panic 감지용)
                                            prev_close = data['close'].iloc[-2]
                                            
                                            # 동적 레버리지 계산
                                            leverage = risk_manager.get_dynamic_leverage(
                                                symbol, atr_current, atr_avg, base_lev, max_lev_limit, current_price, prev_close
                                            )
                                            
                                            # 거래소에 레버리지 설정 반영 (주문 전 필수)
                                            api.set_leverage(symbol, leverage)
                                    except Exception as lev_e:
                                        logger.error(f"동적 레버리지 계산 오류: {lev_e}")
                                
                                # [로그 상세화] 매수 진입 전 지표 요약
                                atr_val = signal.atr_value if signal and signal.atr_value else 0.0
                                conf_score = signal.confidence if signal else 0.0
                                logger.info(f"[{exchange_name}] 🚀 매수 진입 시도: {symbol} | Score: {conf_score:.2f} | ATR: {atr_val:.1f} | Reason: {signal.reason if signal else ''}")

                                if is_pyramiding:
                                    logger.info(f"[{exchange_name}] 🔥 피라미딩(불타기) 주문: {symbol} {buy_qty:.8f}개 @ {ask_price:,.0f}원")
                                else:
                                    logger.info(f"[{exchange_name}] 매수 주문 시도: {symbol} {buy_qty:.8f}개 @ {ask_price:,.0f}원")
                                
                                # price=None을 전달하여 공격적 지정가 로직 활성화
                                result = api.buy(symbol, buy_qty, price=None, leverage=leverage)
                                
                                if result:
                                    # 수수료 및 실제 구매 수량 계산
                                    actual_buy_amount = buy_amount * (1 - fee_rate)
                                    quantity = actual_buy_amount / current_price
                                    
                                    portfolio.add_position(symbol, quantity, current_price, fee_rate=fee_rate, atr_value=atr)
                                    
                                    # 피라미딩 상태 업데이트
                                    if config_key == "crypto":
                                        portfolio.update_pyramiding_state(symbol, current_price, is_reset=not is_pyramiding)
                                    
                                    # ATR 기반 추천 손절가가 있으면 사용, 없으면 기본값 사용
                                    risk_manager.set_stop_loss(symbol, current_price, atr_value=atr, custom_stop_loss=signal.suggested_stop_loss)
                                    # [수정] 익절 목표 설정 (ATR 기반 동적 익절 적용)
                                    risk_manager.set_take_profit(symbol, current_price, fee_rate=fee_rate * 2, atr_value=atr)
                                    portfolio.save_state(save_path)
                                    buy_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    type_str = "PYRAMIDING" if is_pyramiding else "BUY"
                                    
                                    # [New] 레버리지 정보 표시 (바이낸스 선물)
                                    lev_info = ""
                                    if config_key == "binance_futures":
                                        lev_info = f" (Lev: {leverage}x)"

                                    logger.warning("="*70)
                                    logger.warning(f"[{exchange_name}] [{type_str}] {symbol}{lev_info}")
                                    logger.warning(f"시간: {buy_time} | 수량: {quantity:.8f} | 가격: {current_price:,.0f}원")
                                    logger.warning(f"총액: {buy_amount:,.0f}원")
                                    logger.warning("="*70)
                                    
                                    # [New] 텔레그램 알림 전송
                                    if self.report_manager:
                                        self.report_manager.send_trade_alert(
                                            symbol, type_str, ask_price, quantity, reason=signal.reason if signal else ""
                                        )
                                    
                                    # [New] 바이낸스 현물인 경우 OCO 주문 실행 (안전장치)
                                    if config_key == "binance_spot":
                                        try:
                                            # 체결 및 잔액 반영 대기
                                            time.sleep(1.0)
                                            
                                            # 잔액 재조회 (수수료 차감 후 실제 보유량 확인)
                                            target_coin = symbol.split('/')[0]
                                            balance = api.get_balance()
                                            available_qty = balance.get('free', {}).get(target_coin, 0.0)
                                            
                                            if available_qty > 0:
                                                # 매수 평단가 확인
                                                buy_price = result.get('average') or result.get('price') or ask_price
                                                if buy_price:
                                                    buy_price = float(buy_price)
                                                    tp_pct = TRADING_CONFIG["binance_spot"].get("take_profit_percent", 0.05)
                                                    sl_pct = TRADING_CONFIG["binance_spot"].get("stop_loss_percent", 0.02)
                                                    
                                                    # 1차 시도
                                                    oco_order = api.create_oco_order(symbol, available_qty, buy_price, tp_pct, sl_pct)
                                                    
                                                    # [1단계] 실패 시 보정 후 1회 재시도 (간격 20% 확대)
                                                    if not oco_order:
                                                        logger.warning(f"[{exchange_name}] ⚠️ {symbol} OCO 1차 실패. 간격 재보정(20% 확대) 후 재시도...")
                                                        oco_order = api.create_oco_order(symbol, available_qty, buy_price, tp_pct * 1.2, sl_pct * 1.2)
                                                    
                                                    if oco_order:
                                                        self.oco_monitoring_symbols.add(symbol)
                                                        logger.info(f"[{exchange_name}] ✅ {symbol} OCO 주문 등록 완료")
                                                    else:
                                                        # [2, 3, 4단계] 최종 실패 시 대응
                                                        current_p = api.get_price(symbol)
                                                        sl_price = buy_price * (1 - sl_pct)
                                                        
                                                        # 4단계: 위급 상황 (이미 손절가 이탈) -> 시장가 매도
                                                        if current_p > 0 and current_p < sl_price:
                                                            logger.warning(f"[{exchange_name}] 🚨 {symbol} OCO 실패 & 손절가 이탈({current_p} < {sl_price}) -> 즉시 시장가 매도")
                                                            sell_res = api.sell(symbol, available_qty, is_stop_loss=True)
                                                            if sell_res:
                                                                self._send_telegram_alert(f"🚨 {symbol} OCO 실패 및 손절가 이탈로 시장가 매도 실행!")
                                                                # 포트폴리오 정리
                                                                portfolio.remove_position(symbol)
                                                                risk_manager.remove_position(symbol)
                                                                portfolio.save_state(save_path)
                                                        else:
                                                            # 2단계 & 3단계: 로컬 감시 전환 + 알림
                                                            msg = f"[{exchange_name}] ⚠️ {symbol} OCO 주문 실패! 봇이 직접 실시간 감시합니다. (Fallback)"
                                                            logger.warning(msg)
                                                            self._send_telegram_alert(msg)
                                        except Exception as oco_e:
                                            logger.error(f"[{exchange_name}] OCO 주문 처리 중 오류: {oco_e}")

                        except Exception as e:
                            # 잔액 확인 또는 실제 주문 과정에서 발생하는 모든 오류를 여기서 처리
                            logger.error(f"[{exchange_name}] 매수 시도 중 오류 발생 ({symbol}): {e}")
                            continue
                except Exception as e:
                    logger.error(f"[{exchange_name}] ⚠️ 신규 매수 처리 중 오류 ({symbol}): {e}")
                    continue
        
        except Exception as e:
            logger.error(f"{config_key} 거래 오류: {e}")
    
    def print_portfolio_status(self):
        """포트폴리오 상태 출력"""
        logger.info("=" * 60)
        logger.info("포트폴리오 상태")
        logger.info("=" * 60)

        from config.settings import API_CONFIG
        
        # 한국주식 포트폴리오
        # API가 활성화되어 있고, 포지션이 있거나 초기 자본금이 설정된 경우에만 출력
        if (API_CONFIG.get("shinhan") or API_CONFIG.get("kiwoom") or API_CONFIG.get("daishin")) and \
           (self.stock_portfolio.positions or self.stock_portfolio.initial_capital > 0):
            logger.info("[한국주식]")
            current_prices = {}
            stock_apis = [api for api in [self.shinhan_api, self.kiwoom_api, self.daishin_api] if api]
            if stock_apis:
                api = stock_apis[0]
                for symbol in self.stock_portfolio.positions:
                    try:
                        price = api.get_price(symbol)
                        current_prices[symbol] = price
                    except:
                        current_prices[symbol] = 0.0
            
            stats = self.stock_portfolio.get_statistics(current_prices, use_entry_price_fallback=True)
            logger.info(f"총 자산: {stats['total_value']:,.0f}원")
            logger.info(f"수익/손실: {stats['total_profit_loss']:,.0f}원 "
                       f"({stats['total_profit_loss_percent']:.2f}%) | MDD: {stats.get('mdd', 0):.2f}%")
            
            # 종목별 상세 출력
            for symbol, quantity in self.stock_portfolio.positions.items():
                current_price = current_prices.get(symbol, 0) or self.stock_portfolio.entry_prices.get(symbol, 0)
                entry_price = self.stock_portfolio.entry_prices.get(symbol, 0)
                pnl = (current_price - entry_price) * quantity
                pnl_pct = (pnl / (entry_price * quantity) * 100) if entry_price > 0 else 0.0
                emoji = "🔴" if pnl > 0 else "🔵"
                logger.info(f"  {emoji} {symbol}: 현재가 {current_price:,.0f}원 | 평단가 {entry_price:,.0f}원 | "
                            f"평가손익 {pnl:,.0f}원 ({pnl_pct:+.2f}%) | 보유 {quantity}주")
        
        # 암호화폐 포트폴리오
        if API_CONFIG.get("upbit") and \
           (self.crypto_portfolio.positions or self.crypto_portfolio.initial_capital > 0):
            logger.info("[암호화폐]")
            current_prices = {}
            if self.crypto_api:
                for symbol in self.crypto_portfolio.positions:
                    price = self.crypto_api.get_price(symbol)
                    current_prices[symbol] = price
            
            stats = self.crypto_portfolio.get_statistics(current_prices, use_entry_price_fallback=True)
            logger.info(f"총 자산: {stats['total_value']:,.0f}원")
            logger.info(f"수익/손실: {stats['total_profit_loss']:,.0f}원 "
                       f"({stats['total_profit_loss_percent']:.2f}%) | MDD: {stats.get('mdd', 0):.2f}%")
            
            # 종목별 상세 출력
            for symbol, quantity in self.crypto_portfolio.positions.items():
                current_price = current_prices.get(symbol, 0) or self.crypto_portfolio.entry_prices.get(symbol, 0)
                entry_price = self.crypto_portfolio.entry_prices.get(symbol, 0)
                pnl = (current_price - entry_price) * quantity
                pnl_pct = (pnl / (entry_price * quantity) * 100) if entry_price > 0 else 0.0
                emoji = "🔴" if pnl > 0 else "🔵"
                logger.info(f"  {emoji} {symbol}: 현재가 {current_price:,.0f}원 | 평단가 {entry_price:,.0f}원 | "
                            f"평가손익 {pnl:,.0f}원 ({pnl_pct:+.2f}%) | 보유 {quantity:.4f}")

        # 바이낸스 현물
        if API_CONFIG.get("binance_spot") and getattr(self, 'binance_spot_portfolio', None) and \
           (self.binance_spot_portfolio.positions or self.binance_spot_portfolio.initial_capital > 0):
            logger.info("[바이낸스 현물]")
            current_prices = {}
            if getattr(self, 'binance_spot_api', None):
                for symbol in self.binance_spot_portfolio.positions:
                    price = self.binance_spot_api.get_price(symbol)
                    current_prices[symbol] = price

            stats = self.binance_spot_portfolio.get_statistics(current_prices, use_entry_price_fallback=True)
            logger.info(f"총 자산: {stats['total_value']:,.2f} USDT")
            logger.info(f"수익/손실: {stats['total_profit_loss']:,.2f} USDT "
                       f"({stats['total_profit_loss_percent']:.2f}%) | MDD: {stats.get('mdd', 0):.2f}%")
            
            # 종목별 상세 출력
            for symbol, quantity in self.binance_spot_portfolio.positions.items():
                current_price = current_prices.get(symbol, 0) or self.binance_spot_portfolio.entry_prices.get(symbol, 0)
                entry_price = self.binance_spot_portfolio.entry_prices.get(symbol, 0)
                pnl = (current_price - entry_price) * quantity
                pnl_pct = (pnl / (entry_price * quantity) * 100) if entry_price > 0 else 0.0
                emoji = "🔴" if pnl > 0 else "🔵"
                logger.info(f"  {emoji} {symbol}: 현재가 {current_price:,.4f} | 평단가 {entry_price:,.4f} | "
                            f"평가손익 {pnl:,.2f} ({pnl_pct:+.2f}%) | 보유 {quantity:.4f}")

        # 바이낸스 선물
        if API_CONFIG.get("binance_futures") and getattr(self, 'binance_futures_portfolio', None) and \
           (self.binance_futures_portfolio.positions or self.binance_futures_portfolio.initial_capital > 0):
            logger.info("[바이낸스 선물]")
            current_prices = {}
            if getattr(self, 'binance_futures_api', None):
                for symbol in self.binance_futures_portfolio.positions:
                    price = self.binance_futures_api.get_price(symbol)
                    current_prices[symbol] = price

            stats = self.binance_futures_portfolio.get_statistics(current_prices, use_entry_price_fallback=True)
            logger.info(f"총 자산: {stats['total_value']:,.2f} USDT")
            logger.info(f"수익/손실: {stats['total_profit_loss']:,.2f} USDT "
                       f"({stats['total_profit_loss_percent']:.2f}%) | MDD: {stats.get('mdd', 0):.2f}%")
            
            # 종목별 상세 출력
            for symbol, quantity in self.binance_futures_portfolio.positions.items():
                current_price = current_prices.get(symbol, 0) or self.binance_futures_portfolio.entry_prices.get(symbol, 0)
                entry_price = self.binance_futures_portfolio.entry_prices.get(symbol, 0)
                pnl = (current_price - entry_price) * quantity
                pnl_pct = (pnl / (entry_price * quantity) * 100) if entry_price > 0 else 0.0
                emoji = "🔴" if pnl > 0 else "🔵"
                logger.info(f"  {emoji} {symbol}: 현재가 {current_price:,.4f} | 평단가 {entry_price:,.4f} | "
                            f"평가손익 {pnl:,.2f} ({pnl_pct:+.2f}%) | 보유 {quantity:.4f}")
    
    def backup_data(self):
        """data 폴더 백업 및 텔레그램 전송"""
        try:
            # 1. 백업 폴더 생성
            backup_dir = "backups"
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)

            # 2. 압축 파일 생성 (data 폴더 -> backups/data_backup_YYYYMMDD_HHMMSS.zip)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_backup_{timestamp}"
            archive_path = os.path.join(backup_dir, filename)
            
            if os.path.exists("data"):
                shutil.make_archive(archive_path, 'zip', "data")
                zip_path = f"{archive_path}.zip"
                logger.info(f"📦 데이터 백업 완료: {zip_path}")
                
                # 3. 텔레그램으로 전송
                from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
                if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                    try:
                        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
                        with open(zip_path, 'rb') as f:
                            files = {'document': f}
                            data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': f"💾 데이터 백업 ({timestamp})"}
                            requests.post(url, data=data, files=files, timeout=60)
                        logger.info("✅ 텔레그램으로 백업 파일 전송 완료")
                    except Exception as te:
                        logger.error(f"텔레그램 백업 전송 실패: {te}")

                # 4. 오래된 백업 정리 (최근 20개만 유지)
                backups = sorted([f for f in os.listdir(backup_dir) if f.startswith("data_backup_")])
                if len(backups) > 20:
                    for old in backups[:-20]:
                        try:
                            os.remove(os.path.join(backup_dir, old))
                            logger.info(f"🗑️ 오래된 백업 삭제: {old}")
                        except:
                            pass
        except Exception as e:
            logger.error(f"데이터 백업 중 오류: {e}")

    def _warmup_market_data(self):
        """초기 데이터 강제 로드 (Warm-up)"""
        logger.info("🔥 [WARMUP] 초기 시장 데이터 수집 시작 (최소 200 캔들)...")
        
        # Upbit
        if self.crypto_api:
            for symbol in self.crypto_symbols:
                timeframe = TRADING_CONFIG["crypto"].get("timeframe", "15m")
                self._get_latest_ohlcv(self.crypto_api, symbol, timeframe)
        
        # Binance Spot
        if getattr(self, 'binance_spot_api', None):
            for symbol in self.binance_spot_symbols:
                timeframe = TRADING_CONFIG["binance_spot"].get("timeframe", "15m")
                self._get_latest_ohlcv(self.binance_spot_api, symbol, timeframe)

        # Binance Futures
        if getattr(self, 'binance_futures_api', None):
            for symbol in self.binance_futures_symbols:
                timeframe = TRADING_CONFIG["binance_futures"].get("timeframe", "15m")
                self._get_latest_ohlcv(self.binance_futures_api, symbol, timeframe)
                
        logger.info("✅ [READY] 모든 지표 계산 및 데이터 수집 완료")

    def start(self):
        """봇 시작"""
        logger.info("="*60)
        logger.info("자동매매 봇 시작")
        logger.info("="*60)
        
        # API 초기화
        # [수정] 초기화 실패 시 종료하지 않고 재시도 (무한 루프)
        while True:
            if self.initialize_apis():
                break
            
            error_msg = "❌ API 초기화 실패. 60초 후 재시도합니다."
            logger.error(error_msg)
            self._send_telegram_alert(error_msg)
            time.sleep(60)
            
        # 시작 시 지갑 동기화 (기존 보유 종목 로드)
        self.sync_with_exchange()
        
        # 웹소켓 구독 시작 (관심 종목 + 보유 종목)
        if self.crypto_api and hasattr(self.crypto_api, 'subscribe_websocket'):
            all_symbols = list(set(self.crypto_symbols) | set(self.crypto_portfolio.positions.keys()))
            self.crypto_api.subscribe_websocket(all_symbols)
            self.crypto_api.add_price_callback(self._on_realtime_price)
        
        # [New] 바이낸스 현물 웹소켓
        if getattr(self, 'binance_spot_api', None):
            all_symbols = list(set(self.binance_spot_symbols) | set(self.binance_spot_portfolio.positions.keys()))
            self.binance_spot_api.subscribe_websocket(all_symbols)
            self.binance_spot_api.add_price_callback(self._on_realtime_price)

        # [New] 바이낸스 선물 웹소켓
        if getattr(self, 'binance_futures_api', None):
            all_symbols = list(set(self.binance_futures_symbols) | set(self.binance_futures_portfolio.positions.keys()))
            self.binance_futures_api.subscribe_websocket(all_symbols)
            self.binance_futures_api.add_price_callback(self._on_realtime_price)
        
        # 전략 추천 실행
        self.recommend_strategy()
        
        # [New] 초기 기동 시 전체 백테스트 수행 및 설정 반영 (블로킹)
        self.run_periodic_backtest(wait=True)
        
        # [New] 시작 시 전략 파라미터 최적화 즉시 실행 (K, TP, SL)
        self.optimize_strategy_params()
        
        # 모델 학습
        self.train_ml_model()
        
        # [New] 웜업 실행 (데이터 선행 로드)
        self._warmup_market_data()
        
        # [New] 설정 요약 전송
        self._send_config_summary()
        
        # [즉시 실행] 봇 시작 직후 초기 매매 판단 실행
        logger.info("🚀 봇 시작 직후 초기 매매 판단을 실행합니다...")
        self.monitor_and_trade()
        
        # 스케줄 설정: 일일 루틴 (매일 아침 09:05 KST - 업비트 일봉 마감 직후)
        self.scheduler.add_job(
            self.daily_routine,
            'cron',
            hour=9, 
            minute=5
        )
        
        # [New] 매일 아침 9시에 전체 전략 백테스트 및 최적화 실행 (별도 프로세스)
        self.scheduler.add_job(
            self.run_periodic_backtest,
            'cron',
            hour=9,
            minute=0
        )

        self.scheduler.add_job(
            self.monitor_and_trade,
            'interval',
            seconds=MONITORING_CONFIG["check_interval"],
            max_instances=10  # 스케줄러 단의 스킵 경고를 방지하기 위해 여유 있게 설정
        )
        
        # 5초마다 대시보드 커맨드 확인
        self.scheduler.add_job(
            self._check_for_commands,
            'interval',
            seconds=5,
            max_instances=1
        )
        
        # 5초마다 상태 업데이트 (Heartbeat)
        self.scheduler.add_job(
            self._update_status,
            'interval',
            seconds=5,
            max_instances=1
        )
        
        # [Request] 정해진 시간에 리포트 전송 (09, 12, 18, 22시)
        for h in [9, 12, 18, 22]:
            self.scheduler.add_job(
                self.send_portfolio_report,
                'cron',
                hour=h,
                minute=0
            )
        
        # [New] 매일 아침 8시에 USDT 잔고 간편 보고
        self.scheduler.add_job(
            self.report_usdt_balance,
            'cron',
            hour=8,
            minute=0
        )
        
        # 1분마다 지갑 동기화 (외부 매매 내역 반영)
        self.scheduler.add_job(
            self.sync_wallet,
            'interval',
            minutes=1
        )
        
        self.scheduler.add_job(
            self.print_portfolio_status,
            'interval',
            hours=1
        )
        
        # 미체결 주문 관리 (1분마다 확인)
        self.scheduler.add_job(
            self.cancel_old_orders,
            'interval',
            minutes=1
        )
        
        # 6시간마다 데이터 백업 및 텔레그램 전송
        self.scheduler.add_job(
            self.backup_data,
            'interval',
            hours=6
        )
        
        # 웹소켓 지연 감시 (1분마다)
        self.scheduler.add_job(
            self.check_ws_latency,
            'interval',
            minutes=1
        )
        
        # 바이낸스 웹소켓 정기 재연결 (50분마다)
        self.scheduler.add_job(
            self.refresh_binance_websocket,
            'interval',
            minutes=50
        )
        
        # API 헬스 체크 (5분마다)
        self.scheduler.add_job(
            self.check_api_health,
            'interval',
            minutes=5
        )
        
        # [Request 2] 전진분석 주기 변경: 4시간마다 실행 (부하 감소)
        self.scheduler.add_job(
            self.find_best_k,
            'cron',
            hour='*/4', minute=0
        )
        
        # 스케줄러 시작
        self.scheduler.start()
        
        self._update_status("running") # 시작 시 즉시 상태 업데이트
        
        logger.info("자동매매 봇 시작 완료")
        
        # 지속적 실행
        try:
            while True:
                time.sleep(1)
        
        except (KeyboardInterrupt, SystemExit):
            self.stop()
    
    def send_portfolio_report(self):
        """포트폴리오 현황 텔레그램 전송"""
        if self.report_manager:
            self.report_manager.report_portfolio_status(self.crypto_portfolio, "UPBIT", api=self.crypto_api)
            if getattr(self, 'binance_spot_portfolio', None):
                self.report_manager.report_portfolio_status(self.binance_spot_portfolio, "BINANCE SPOT", api=self.binance_spot_api)
            if getattr(self, 'binance_futures_portfolio', None):
                self.report_manager.report_portfolio_status(self.binance_futures_portfolio, "BINANCE FUTURES", api=self.binance_futures_api)

    def report_usdt_balance(self):
        """매일 아침 USDT 잔고 간편 보고"""
        if not self.report_manager: return
        
        # 최신 잔고 동기화
        self.sync_wallet()
        
        msg = "🌅 *[일일 USDT 잔고 보고]*\n\n"
        has_data = False
        
        # 바이낸스 현물
        if getattr(self, 'binance_spot_portfolio', None):
            usdt = self.binance_spot_portfolio.current_capital
            msg += f"🟡 *Binance Spot*: `{usdt:,.2f} USDT`\n"
            has_data = True
            
        # 바이낸스 선물
        if getattr(self, 'binance_futures_portfolio', None):
            usdt = self.binance_futures_portfolio.current_capital
            msg += f"🔴 *Binance Futures*: `{usdt:,.2f} USDT`\n"
            has_data = True
            
        if has_data:
            self.report_manager.send_telegram_message(msg)

    def stop(self):
        """봇 종료"""
        # 중복 종료 방지
        if hasattr(self, '_stopping'):
            return
        self._stopping = True
        
        logger.info("자동매매 봇 종료 중...")
        
        if self.scheduler.running:
            # try:하여 현재 실행 중인 잡(이 함수를 호출한 잡 포함)이
            try:
                # [수정] wait=False로 설정하여 현재 실행 중인 잡(이 함수를 호출한 잡 포함)이
                # 종료될 때까지 기다리지 않도록 함 (데드락/Join 에러 방지)
                self.scheduler.shutdown(wait=False)
            except Exception as e:
                logger.error(f"스케줄러 종료 중 오류: {e}")
            
        self.stock_portfolio.save_state("data/stock_portfolio.json")
        self.crypto_portfolio.save_state("data/crypto_portfolio.json")
        self.binance_spot_portfolio.save_state("data/binance_spot_portfolio.json")
        self.binance_futures_portfolio.save_state("data/binance_futures_portfolio.json")
        
        if self.shinhan_api:
            self.shinhan_api.disconnect()
        
        if self.kiwoom_api:
            self.kiwoom_api.disconnect()
        
        if self.daishin_api:
            self.daishin_api.disconnect()
        
        if self.crypto_api:
            self.crypto_api.disconnect()
            
        logger.info("자동매매 봇 종료 완료")


if __name__ == "__main__":
    # [중요] PyInstaller 빌드 시 멀티프로세싱(pyupbit 등) 무한 재실행 방지
    multiprocessing.freeze_support()
    bot = AutoTradingBot()
    bot.start()
