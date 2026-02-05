"""
현재 설정(settings.py) 기반 백테스팅 실행 스크립트
"""
import os
import logging
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from config.settings import TRADING_CONFIG, API_CONFIG, ML_CONFIG
from api.crypto_api import UpbitAPI, BinanceAPI
from utils.backtesting import WalkForwardAnalyzer
from trading.strategy import TechnicalStrategy, MLStrategy
from trading.strategy_v2 import HeikinAshiStrategy
from trading.turtle_bollinger_strategy import TurtleBollingerStrategy
from trading.agile_strategy import AgileStrategy
from trading.volume_trend_strategy import VolumeTrendStrategy
from trading.ma_trend_strategy import MATrendStrategy
from trading.early_bird_strategy import EarlyBirdStrategy
from utils.logger import setup_logger

# 환경변수 로드
load_dotenv()

# [New] 전략 래퍼 클래스 (백테스트 로깅용)
class StrategyWrapper:
    def __init__(self, strategy):
        self.strategy = strategy
        self.last_reason = {} # 심볼별 마지막 사유 저장 {symbol: reason}
        
    def __getattr__(self, name):
        # lookback_window 등 속성 접근 위임
        return getattr(self.strategy, name)

    def generate_signal(self, symbol, data, current_capital=0.0, strategy_override=None):
        signal = self.strategy.generate_signal(symbol, data, current_capital, strategy_override)
        logger = logging.getLogger("backtest")
        
        if signal:
            if signal.action != "HOLD":
                logger.info(f"   👉 [신호발생] {signal.action} {symbol} | 사유: {signal.reason} | Conf: {signal.confidence}")
                self.last_reason[symbol] = "ENTRY"
            else:
                # HOLD 사유가 변경되었을 때만 로그 출력 (로그 폭주 방지)
                last = self.last_reason.get(symbol, "")
                if signal.reason != last:
                    logger.info(f"   💤 [진입보류] {symbol} | 사유: {signal.reason} | Conf: {signal.confidence}")
                    self.last_reason[symbol] = signal.reason
        return signal

def get_strategy(strategy_name, lookback):
    """전략 객체 생성"""
    strategy_name = strategy_name.lower()
    if strategy_name == "heikin_ashi":
        return HeikinAshiStrategy(lookback_window=lookback)
    elif strategy_name == "turtle_bollinger":
        return TurtleBollingerStrategy(lookback_window=lookback)
    elif strategy_name == "agile":
        return AgileStrategy(lookback_window=lookback)
    elif strategy_name == "volume_trend":
        return VolumeTrendStrategy(lookback_window=lookback)
    elif strategy_name == "ma_trend":
        return MATrendStrategy(lookback_window=lookback)
    elif strategy_name == "early_bird":
        return EarlyBirdStrategy(lookback_window=lookback)
    else:
        # 기본값 또는 technical
        return TechnicalStrategy(lookback_window=lookback)

def run_backtest_for_config(config_key, api_class, api_key_env, api_secret_env):
    """특정 설정에 대한 백테스트 실행"""
    conf = TRADING_CONFIG.get(config_key)
    if not conf:
        return

    logger = logging.getLogger("backtest")
    
    api_key = os.getenv(api_key_env)
    api_secret = os.getenv(api_secret_env)
    
    if not api_key or not api_secret:
        logger.warning(f"⚠️ {config_key}: API 키가 설정되지 않아 백테스트를 건너뜁니다.")
        return

    try:
        if config_key == "binance_futures":
            api = api_class(api_key, api_secret, account_type='future')
        elif config_key == "binance_spot":
            api = api_class(api_key, api_secret, account_type='spot')
        else:
            api = api_class(api_key, api_secret)
        
        # 연결 시도 (데이터 조회를 위해 필요)
        api.connect()
        
    except Exception as e:
        logger.error(f"❌ {config_key}: API 연결 실패 - {e}")
        return

    symbols = conf.get("symbols", [])
    if not symbols:
        logger.warning(f"⚠️ {config_key}: 설정된 종목이 없습니다.")
        return

    timeframe = conf.get("timeframe", "15m")
    entry_strategy_name = conf.get("entry_strategy", "breakout")
    strategy_type = conf.get("strategy_type", "technical")
    
    # [Hack] TechnicalStrategy가 TRADING_CONFIG["crypto"]를 참조하므로,
    # 현재 테스트하려는 설정값으로 잠시 덮어씌웁니다. (바이낸스 테스트 시 중요)
    original_crypto_conf = TRADING_CONFIG["crypto"].copy()
    TRADING_CONFIG["crypto"].update(conf)
    
    # 전략 객체 생성
    strategy = get_strategy(strategy_type, ML_CONFIG["lookback_window"])
    
    # [New] 전략 래퍼 적용 (로그 출력용)
    strategy = StrategyWrapper(strategy)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 [{config_key.upper()}] 백테스팅 시작")
    logger.info(f"   - 대상 종목: {len(symbols)}개 ({', '.join(symbols[:5])}...)")
    logger.info(f"   - 타임프레임: {timeframe}")
    logger.info(f"   - 메인 전략: {strategy_type} ({type(strategy).__name__})")
    logger.info(f"   - 진입 전략: {entry_strategy_name}")
    
    # [New] 적용된 프리셋 확인 (Crypto인 경우)
    if config_key == "crypto":
        from config.settings import selected_strategy_name
        logger.info(f"   - 전략 프리셋: {selected_strategy_name} (파라미터 결정)")

    logger.info(f"   - 익절: {conf.get('take_profit_percent', 0)*100:.1f}%")
    logger.info(f"   - 손절: {conf.get('stop_loss_percent', 0)*100:.1f}%")
    logger.info(f"{'='*60}")

    # [New] 설정값 정합성 체크 및 경고
    if entry_strategy_name == "agile":
        tp = conf.get('take_profit_percent', 0)
        if tp > 0.05: # Agile인데 익절이 5% 넘으면 경고
             logger.warning(f"⚠️ [설정 주의] Agile 전략은 초단타용이나, 현재 익절({tp*100:.1f}%)이 매우 높게 설정되어 있습니다.")
             logger.warning("   👉 .env 파일의 CRYPTO_TAKE_PROFIT 설정이 프리셋을 덮어쓰고 있는지 확인하세요.")
             logger.warning("   👉 또는 CRYPTO_STRATEGY_PRESET이 'agile'이 아닌 다른 값(예: short_term)으로 설정되어 있는지 확인하세요.")

    total_pnl = 0
    
    # 수수료 설정
    if "binance" in config_key:
        fee_rate = TRADING_CONFIG["fees"].get("binance_fee_rate", 0.001)
    else:
        fee_rate = TRADING_CONFIG["fees"].get("crypto_fee_rate", 0.0005)

    for symbol in symbols:
        logger.info(f"🔍 분석 중: {symbol}...")
        
        # 데이터 수집 (최근 3000개 - 넉넉하게)
        try:
            df = api.get_ohlcv(symbol, timeframe=timeframe, count=3000)
        except Exception as e:
            logger.error(f"   ❌ 데이터 조회 실패: {e}")
            continue
        
        if df.empty or len(df) < 200:
            logger.warning(f"   ⚠️ {symbol}: 데이터 부족 ({len(df)} rows)")
            continue

        # 테스트 기간: 최근 30% 구간 (전진분석)
        test_len = int(len(df) * 0.3)
        
        # [Fix] 타임프레임에 따른 적절한 Lookback 계산
        # 변동성 돌파(일봉 필요) 등을 위해 충분한 데이터 확보
        if timeframe == "1m":
            lookback = 3000 # 약 2일치 (1440 * 2)
        elif timeframe in ["3m", "5m"]:
            lookback = 1000
        elif timeframe == "15m":
            lookback = 300  # 약 3일치 (96 * 3)
        else:
            lookback = 100

        analyzer = WalkForwardAnalyzer(
            df,
            symbol=symbol, # [New] 심볼 전달 (로그에 정확한 종목명 표시)
            train_period=200, # 지표 계산용 여유분
            test_period=test_len,
            slippage=0.001, # 0.1%
            fee=fee_rate,
            stop_loss=conf.get("stop_loss_percent", 0.0),
            take_profit=conf.get("take_profit_percent", 0.0),
            trailing_stop=conf.get("trailing_stop_percent", 0.0)
        )
        
        results = analyzer.run(strategy_type=strategy, lookback_window=lookback)
        
        if not results.empty:
            sym_return = results['total_return'].sum()
            sym_win_rate = results['win_rate'].mean()
            trade_count = results['trade_count'].sum()
            total_pnl += sym_return
            
            logger.info(f"   👉 결과: 수익 {sym_return:,.0f} | 승률 {sym_win_rate*100:.1f}% | 거래 {trade_count}회")
        else:
            logger.info(f"   👉 결과: 거래 없음")

    logger.info(f"\n💰 [{config_key.upper()}] 총 예상 수익: {total_pnl:,.0f}")
    
    # 설정 원복
    TRADING_CONFIG["crypto"] = original_crypto_conf

def main():
    # 로거 설정
    log_filename = f"backtest_current_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logger("backtest", filename=log_filename)
    
    logger = logging.getLogger("backtest")
    logger.info("=" * 60)
    logger.info("현재 설정(settings.py) 기반 백테스팅")
    logger.info("=" * 60)
    
    # 1. Upbit
    if API_CONFIG["upbit"]:
        run_backtest_for_config("crypto", UpbitAPI, "UPBIT_API_KEY", "UPBIT_API_SECRET")
        
    # 2. Binance Spot
    if API_CONFIG["binance_spot"]:
        run_backtest_for_config("binance_spot", BinanceAPI, "BINANCE_API_KEY", "BINANCE_API_SECRET")

    # 3. Binance Futures
    if API_CONFIG["binance_futures"]:
        run_backtest_for_config("binance_futures", BinanceAPI, "BINANCE_API_KEY", "BINANCE_API_SECRET")

    print(f"\n✅ 백테스팅 완료. 결과가 logs/{log_filename} 파일에 저장되었습니다.")

if __name__ == "__main__":
    main()
