import os
import sys
from dotenv import load_dotenv

# [변경] 빌드 환경(PyInstaller) 지원을 위한 .env 절대 경로 설정
if getattr(sys, 'frozen', False):
    # exe로 실행될 때: 실행 파일이 있는 경로 기준
    BASE_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    # 스크립트로 실행될 때: 프로젝트 루트 기준 (config 폴더의 상위)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENV_PATH = os.path.join(BASE_DIR, '.env')
load_dotenv(ENV_PATH)

"""
================================================================================
                        자동매매 봇 설정 파일
================================================================================

[주의] 아래 설정값들을 수정하고 봇을 재기동하면 변경사항이 적용됩니다.
      run.bat 파일을 더블클릭하면 자동으로 봇이 시작됩니다.

================================================================================
1. API 인증 정보 - .env 파일에서 관리 (여기선 수정하지 마세요)
================================================================================
"""

def get_env_float(key, default):
    """환경변수에서 float 값을 가져오거나 기본값을 반환"""
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return float(default)

def get_env_int(key, default):
    """환경변수에서 int 값을 가져오거나 기본값을 반환"""
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return int(default)

def get_env_bool(key, default):
    """환경변수에서 bool 값을 가져오거나 기본값을 반환"""
    val = os.getenv(key, str(default)).lower()
    return val in ["true", "1", "yes", "on"]

SHINHAN_API_KEY = os.getenv("SHINHAN_API_KEY", "your_shinhan_api_key")
SHINHAN_API_SECRET = os.getenv("SHINHAN_API_SECRET", "your_shinhan_api_secret")
SHINHAN_ACCOUNT = os.getenv("SHINHAN_ACCOUNT", "your_account_number")

MIRAE_API_KEY = os.getenv("MIRAE_API_KEY", "your_mirae_api_key")
MIRAE_API_SECRET = os.getenv("MIRAE_API_SECRET", "your_mirae_api_secret")
MIRAE_ACCOUNT = os.getenv("MIRAE_ACCOUNT", "your_account_number")

KIWOOM_API_KEY = os.getenv("KIWOOM_API_KEY", "your_kiwoom_api_key")
KIWOOM_API_SECRET = os.getenv("KIWOOM_API_SECRET", "your_kiwoom_api_secret")
KIWOOM_ACCOUNT = os.getenv("KIWOOM_ACCOUNT", "your_account_number")

DAISHIN_API_KEY = os.getenv("DAISHIN_API_KEY", "your_daishin_api_key")
DAISHIN_API_SECRET = os.getenv("DAISHIN_API_SECRET", "your_daishin_api_secret")
DAISHIN_ACCOUNT = os.getenv("DAISHIN_ACCOUNT", "your_account_number")

UPBIT_API_KEY = os.getenv("UPBIT_API_KEY", "")
UPBIT_API_SECRET = os.getenv("UPBIT_API_SECRET", "")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "your_binance_api_key")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "your_binance_api_secret")

BINANCE_SYMBOLS_STR = os.getenv("BINANCE_SYMBOLS", "BTC/USDT,ETH/USDT,BNB/USDT,XRP/USDT")
BINANCE_SYMBOLS = [s.strip() for s in BINANCE_SYMBOLS_STR.split(",")]

# 텔레그램 설정
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

"""
================================================================================
2. 거래 설정 - 여기서 자유롭게 수정 가능
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────
# 전략 프리셋 정의
# ─────────────────────────────────────────────────────────────────────────
STRATEGY_PRESETS = {
    "scalping": { # 초단타 (1분봉: 노이즈 대비 손절 여유, 수수료 극복 위한 익절 상향)
        "take_profit_percent": 0.05,   # 2% -> 5%로 상향 (추세를 더 길게 먹어야 함)
        "trailing_stop_percent": 0.015, # 0.8% -> 1.5%로 조정 (노이즈에 털리지 않게)
        "timeframe": "1m",
    },
    "short_term": { # 단기 (15분봉: 데이트레이딩, 손익비 1:2 목표)
        "take_profit_percent": 0.06,    # 6.0%
        "trailing_stop_percent": 0.015,
        "timeframe": "15m",
    },
    "mid_term": { # 중기 (4시간봉: 스윙, 추세 추종)
        "take_profit_percent": 0.20,
        "trailing_stop_percent": 0.04,  # 4.0%
        "timeframe": "4h",
    },
    "long_term": { # 장기 (일봉: 대세 상승)
        "take_profit_percent": 0.50,
        "trailing_stop_percent": 0.10,
        "timeframe": "1d",
    }
}

# .env에서 사용할 전략 프리셋 선택 (기본값: scalping)
# 사용 가능한 값: "scalping", "short_term", "mid_term", "long_term"
# 설정 방법: .env 파일에 CRYPTO_STRATEGY_PRESET=mid_term 와 같이 입력
selected_strategy_name = os.getenv("CRYPTO_STRATEGY_PRESET", "scalping").lower()
selected_strategy = STRATEGY_PRESETS.get(selected_strategy_name, STRATEGY_PRESETS["scalping"])


TRADING_CONFIG = {
    # ─────────────────────────────────────────────────────────────────────────
    # 한국 주식 거래 설정
    # ─────────────────────────────────────────────────────────────────────────
    "korean_stocks": {
        "symbols": ["005930", "000660", "035420"],     # 거래 대상 주식 (삼성전자, SK하이닉스, NAVER)
        "initial_capital": 0,                           # 초기 자본 (0 = 현재 비활성화)
        "max_position_size": 0.3,                       # 한 종목당 최대 포트폴리오의 30%
        "stop_loss_percent": 0.05,                      # 손실 5% 시 자동 매도
        "take_profit_percent": 0.10,                    # 이익 10% 시 자동 매도
        "timeframe": "1d",                              # 캔들 기준 (1d=일봉, 15m=15분봉)
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # 암호화폐 거래 설정 (현재 활성화됨)
    # ─────────────────────────────────────────────────────────────────────────
    "crypto": {
        "symbols": ["SOL/KRW", "XRP/KRW", "DOGE/KRW", "AVAX/KRW"],  # [변경] 알트코인 위주 구성
        "initial_capital": get_env_float("CRYPTO_INITIAL_CAPITAL", 300000),
        "max_position_size": get_env_float("CRYPTO_MAX_POSITION_SIZE", 0.2),
        # 전략 프리셋 또는 .env 개별 설정으로 값 결정
        "take_profit_percent": get_env_float("TAKE_PROFIT_PCT", get_env_float("CRYPTO_TAKE_PROFIT", selected_strategy["take_profit_percent"])),
        "trailing_stop_percent": get_env_float("CRYPTO_TRAILING_STOP", 0.015), # 기본값 1.5%
        "timeframe": os.getenv("CRYPTO_TIMEFRAME", selected_strategy["timeframe"]),
        "max_positions": get_env_int("CRYPTO_MAX_POSITIONS", 5),      # 최대 보유 종목 수 (기본값 5개)
        "min_order_amount": get_env_float("CRYPTO_MIN_ORDER_AMOUNT", 5000), # 최소 주문 금액 (기본값 5000원)
        "cancel_timeout": get_env_int("CRYPTO_CANCEL_TIMEOUT", 300),  # 미체결 주문 취소 대기 시간 (초, 기본값 300초)
        "slippage_ticks": get_env_int("CRYPTO_SLIPPAGE_TICKS", 2),    # [New] 공격적 지정가 슬리피지 허용 틱 (기본 2틱)
        "order_wait_seconds": get_env_int("CRYPTO_ORDER_WAIT_SECONDS", 5), # [New] 주문 체결 대기 시간 (초)
        "atr_window": get_env_int("CRYPTO_ATR_WINDOW", 20), # ATR 계산 기간 (기본값 20)
        "atr_multiplier": get_env_float("CRYPTO_ATR_MULTIPLIER", 2.0), # ATR 기반 손절 배수 (보통 1.5~3.0 사용) 코인의 높은 변동성을 고려해 2.5N으로 여유 부여
        "adx_threshold": get_env_float("CRYPTO_ADX_THRESHOLD", 15.0),  # ADX 추세 강도 필터 (15로 완화하여 빠른 진입)
        "volatility_threshold": get_env_float("CRYPTO_VOLATILITY_THRESHOLD", 0.8), # 최소 변동성 기준 (0.5% -> 0.8% 상향 조정)
        "stop_loss_percent": get_env_float("STOP_LOSS_PCT", get_env_float("CRYPTO_STOP_LOSS", 0.0)), # 고정 손절 % (0이면 ATR 사용)
        "entry_strategy": os.getenv("CRYPTO_ENTRY_STRATEGY", "breakout"), # 진입 전략 (breakout, combined, rsi, macd, bollinger, bb_breakout, rsi_bollinger, turtle_bollinger)
        "k_value": get_env_float("CRYPTO_K_VALUE", 0.6), # 변동성 돌파 전략 K값
        "strategy_type": os.getenv("STRATEGY_TYPE", os.getenv("CRYPTO_STRATEGY_TYPE", "technical")), # 사용할 메인 전략
        # 피라미딩(불타기) 설정
        "pyramiding_enabled": True,   # 피라미딩 활성화 여부
        "pyramiding_threshold": 0.012, # 1.25N(약 1.2%) 상승 시마다 추가 매수 (익절가보다 낮게 설정)
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # 바이낸스 거래 설정 (USDT 마켓)
    # ─────────────────────────────────────────────────────────────────────────
    "binance": {
        "symbols": BINANCE_SYMBOLS,
        "initial_capital": get_env_float("BINANCE_INITIAL_CAPITAL", 1000), # USDT 기준
        "max_position_size": get_env_float("BINANCE_MAX_POSITION_SIZE", 0.3),
        "leverage": get_env_int("BINANCE_LEVERAGE", 1), # [New] 레버리지 설정 (기본 1배)
        "futures_enabled": get_env_bool("BINANCE_FUTURES_ENABLED", False), # [New] 선물 모드 활성화
        "take_profit_percent": get_env_float("BINANCE_TAKE_PROFIT", 0.10),
        "trailing_stop_percent": get_env_float("BINANCE_TRAILING_STOP", 0.02),
        "timeframe": "15m",
        "max_positions": get_env_int("BINANCE_MAX_POSITIONS", 3),
        "min_order_amount": get_env_float("BINANCE_MIN_ORDER_AMOUNT", 10), # 최소 10 USDT
        "cancel_timeout": 300,
        "slippage_ticks": 2,
        "order_wait_seconds": 5,
        "atr_window": 20,
        "atr_multiplier": 2.0,
        "stop_loss_percent": 0.0,
        "entry_strategy": "breakout",
        "pyramiding_enabled": True,
    }
}

# ─────────────────────────────────────────────────────────────────────────
# 수수료 설정 (수익률 계산 현실화)
# ─────────────────────────────────────────────────────────────────────────
TRADING_CONFIG["fees"] = {
    "stock_fee_rate": 0.00015,  # 0.015% (일반적인 증권사 수수료)
    "stock_tax_rate": 0.0020,   # 0.20% (증권거래세 + 농특세)
    "crypto_fee_rate": 0.0005,  # 0.05% (업비트 기준)
    "binance_fee_rate": 0.001,  # 0.1% (바이낸스 현물 기준)
}

"""
================================================================================
3. 머신러닝 설정 - AI 예측 관련
================================================================================
"""

ML_CONFIG = {
    "model_type": "random_forest",      # 사용할 ML 모델: random_forest, xgboost
    "lookback_window": 200,             # [수정] 변동성 돌파 일봉 계산을 위해 룩백 기간 상향 (60 -> 200)
    "prediction_horizon": 5,            # 5일 후 가격 예측
    "train_ratio": 0.8,                 # 학습 데이터 80%
    "validation_ratio": 0.1,            # 검증 데이터 10%
    "test_ratio": 0.1,                  # 테스트 데이터 10%
}

"""
================================================================================
4. 모니터링 설정 - 봇 실행 주기 및 로그
================================================================================
"""

MONITORING_CONFIG = {
    "check_interval": 3,                # 3초마다 시장 확인 (API 호출 제한 고려)
    "websocket_enabled": os.getenv("USE_WEBSOCKET", "True").lower() == "true", # 실시간 가격 데이터 수신 (.env 제어)
    "log_level": "INFO",                # 로그 레벨: DEBUG, INFO, WARNING, ERROR
    "max_log_size": 100000000,          # 최대 로그 크기: 100MB 도달 시 새파일
}

"""
================================================================================
4-2. 거래량 기반 자동 종목 선택 설정
================================================================================
"""

VOLUME_CONFIG = {
    "auto_select_enabled": True,        # 거래량 기반 자동 종목 선택 활성화
    "min_volume_krw": 100000000,        # 최소 거래량: 1억원 이상만 선택
    "max_symbols": get_env_int("MAX_SYMBOLS", 10), # 최대 감시 종목 수 (기본값 10개)
    "update_interval": 3600,            # 1시간마다 종목 목록 업데이트 (초 단위)
    "exclude_major_coins": get_env_bool("EXCLUDE_MAJOR_COINS", True), # [변경] BTC, ETH 제외 (알트코인 집중)
    "exclude_symbols": [],              # 제외할 종목 (예: ["DOGE/KRW", "SHIB/KRW"])
}

"""
================================================================================
5. 포트폴리오 설정 - 자산 배분
================================================================================
"""

PORTFOLIO_CONFIG = {
    "rebalance_interval": "monthly",    # 월간 자산 재배분
    "target_allocation": {
        "korean_stocks": 0.5,           # 국내주식 50%
        "crypto": 0.3,                  # 암호화폐 30%
        "cash": 0.2,                    # 현금 보유 20%
    }
}

"""
================================================================================
6. API 사용 선택 - 활성화할 거래소/증권사 선택 (True/False)
================================================================================
주의: 각 거래소의 API 인증 정보가 .env 파일에 설정되어야 합니다.
"""

API_CONFIG = {
    "shinhan": get_env_bool("ENABLE_SHINHAN", False),   # 신한투자증권
    "kiwoom": get_env_bool("ENABLE_KIWOOM", False),     # 키움증권
    "daishin": get_env_bool("ENABLE_DAISHIN", False),   # 대신증권
    "upbit": get_env_bool("ENABLE_UPBIT", False),        # 업비트 암호화폐 거래소
    "binance": get_env_bool("ENABLE_BINANCE", False),   # 바이낸스 암호화폐 거래소
}
