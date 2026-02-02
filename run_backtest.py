"""
백테스팅 실행 스크립트
체크리스트 반영:
1. settings.py의 TRADING_CONFIG 설정 연동
2. 슬리피지 및 수수료 적용
"""

import os
import logging
import json
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from config.settings import TRADING_CONFIG
from api.crypto_api import UpbitAPI
from utils.backtesting import WalkForwardAnalyzer
from trading.strategy_v2 import HeikinAshiStrategy
from trading.turtle_bollinger_strategy import TurtleBollingerStrategy
from utils.logger import setup_logger

# 환경변수 로드
load_dotenv()

def main():
    # 로거 설정 (백테스트 결과를 별도 파일로 저장)
    log_filename = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger("backtest", filename=log_filename)
    logger.info("=" * 60)
    logger.info("백테스팅 시작")
    logger.info("=" * 60)

    # API 키 확인
    api_key = os.getenv("UPBIT_API_KEY")
    api_secret = os.getenv("UPBIT_API_SECRET")
    
    if not api_key or not api_secret:
        logger.error("❌ .env 파일에 UPBIT API 키가 설정되지 않았습니다.")
        return

    api = UpbitAPI(api_key, api_secret)
    api.connect()

    # 1. 비교 시나리오 설정 (비트코인 vs 알트코인)
    scenarios = [
        {"symbol": "BTC/KRW", "timeframe": "4h", "desc": "비트코인 (4시간봉)"},
        {"symbol": "ETH/KRW", "timeframe": "15m", "desc": "이더리움 (15분봉)"},
        {"symbol": "XRP/KRW", "timeframe": "15m", "desc": "리플 (15분봉)"},
    ]

    slippage = 0.001        # [수정] 슬리피지 0.1% (업비트 상위 종목 기준)
    fee = TRADING_CONFIG["fees"]["crypto_fee_rate"]
    stop_loss = 0.04        # SL 4%
    take_profit = 0.12      # TP 12%
    trailing_stop = None    # [수정] 손익비 테스트를 위해 트레일링 스탑 해제 (순수 R/R 검증)
    confidence_threshold = 0.5

    comparison_results = []

    for sc in scenarios:
        symbol = sc["symbol"]
        timeframe = sc["timeframe"]
        desc = sc["desc"]
        
        # 타임프레임별 변동성 필터 조정
        if timeframe in ["1m", "3m", "5m", "10m", "15m", "30m", "1h"]:
            TRADING_CONFIG["crypto"]["volatility_threshold"] = 0.1
            TRADING_CONFIG["crypto"]["adx_threshold"] = 15.0
        else:
            TRADING_CONFIG["crypto"]["volatility_threshold"] = 0.5
            TRADING_CONFIG["crypto"]["adx_threshold"] = 20.0

        logger.info("\n" + "=" * 60)
        logger.info(f"🧪 시나리오 분석: {desc} - {symbol}")
        logger.info("=" * 60)

        # 파라미터 로깅
        logger.info("-" * 50)
        logger.info(f"📋 백테스팅 파라미터 요약 ({timeframe})")
        logger.info("-" * 50)
        logger.info(f"   • 타임프레임      : {timeframe}")
        logger.info(f"   • K-Value         : {TRADING_CONFIG['crypto']['k_value']}")
        logger.info(f"   • 슬리피지        : {slippage*100:.2f}%")
        logger.info(f"   • 손절(SL)        : {stop_loss*100:.2f}%")
        logger.info(f"   • 익절(TP)        : {take_profit*100:.2f}%")
        logger.info("-" * 50)

        # 데이터 수집
        count = 13000
        logger.info(f"   데이터 수집 중... (최대 {count}개 캔들)")
        df = api.get_ohlcv(symbol, timeframe=timeframe, count=count)
        
        if df.empty:
            logger.error(f"❌ {symbol} 데이터를 가져오지 못했습니다.")
            continue

        logger.info(f"   수집된 데이터: {len(df)}개 ({df.index[0]} ~ {df.index[-1]})")

        # 테스트 기간 계산
        if timeframe.endswith('m'):
            minutes = int(timeframe[:-1])
            test_len = int((60 * 24 * 7) / minutes)
        elif timeframe == '1h':
            test_len = 24 * 7
        elif timeframe == '4h':
            test_len = 6 * 7
        else:
            test_len = 100

        # 비교할 전략 목록
        strategies_to_test = [
            ("Breakout", "technical"),
            ("HeikinAshi", "heikin_ashi"),
            ("TurtleBollinger", "turtle_bollinger")
        ]

        for strat_name, strat_type in strategies_to_test:
            # 전략 설정
            if strat_type == "technical":
                TRADING_CONFIG["crypto"]["entry_strategy"] = "breakout"
                TRADING_CONFIG["crypto"]["k_value"] = 0.6
                strategy_arg = "technical"
            elif strat_type == "heikin_ashi":
                # HeikinAshiStrategy 인스턴스 생성 (lookback_window는 Analyzer와 맞춰줌)
                strategy_arg = HeikinAshiStrategy(lookback_window=400)
            elif strat_type == "turtle_bollinger":
                strategy_arg = TurtleBollingerStrategy(lookback_window=400)

            analyzer = WalkForwardAnalyzer(
                df, 
                train_period=60, 
                test_period=test_len, # 자동 계산된 기간 적용
                slippage=slippage,
                fee=fee,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop=trailing_stop,
                confidence_threshold=confidence_threshold
            )
            
            logger.info(f"\n🚀 전략 검증 실행: {strat_name} ({desc})...")
            results = analyzer.run(strategy_type=strategy_arg)
            
            logger.info(f"\n📈 {strat_name} 백테스팅 결과:")
            
            # [수정] 포맷터를 사용하여 승률은 소수점 표시, 금액은 정수 표시
            formatters = {
                'total_return': '{:,.0f}'.format,
                'max_drawdown': '{:,.0f}'.format,
                'win_rate': '{:.2f}'.format,  # 승률 소수점 2자리 (예: 0.50)
                'trade_count': '{:.0f}'.format
            }
            result_str = results[['test_period', 'total_return', 'win_rate', 'max_drawdown', 'trade_count']].to_string(formatters=formatters)
            logger.info("\n" + result_str)
            
            total_return = results['total_return'].sum()
            win_rate = results['win_rate'].mean()
            logger.info(f"\n💰 {strat_name} 총 예상 수익: {total_return:,.0f} KRW")
            
            comparison_results.append({
                'label': f"{desc} - {strat_name}", 
                'return': total_return, 
                'win_rate': win_rate,
                'df': results
            })

    # 최종 비교 출력
    logger.info("\n" + "=" * 60)
    logger.info("📊 전략별 수익률 비교 결과 (Breakout vs HeikinAshi)")
    logger.info("=" * 60)
    for res in comparison_results:
        logger.info(f"   • {res['label']:<30}: 수익 {res['return']:,.0f} KRW (승률 {res['win_rate']*100:.1f}%)")
    logger.info("=" * 60)

    print(f"✅ 백테스팅 완료. 결과가 logs/{log_filename} 파일에 저장되었습니다.")

if __name__ == "__main__":
    main()