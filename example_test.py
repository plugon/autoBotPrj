"""
예제: 간단한 테스트
"""

import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from models.ml_model import MLPredictor
from trading.strategy import MLStrategy, TechnicalStrategy
from trading.portfolio import Portfolio
from trading.risk_manager import RiskManager


def generate_sample_data(days: int = 200) -> pd.DataFrame:
    """샘플 데이터 생성"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # 랜덤 보행 생성
    returns = np.random.normal(0.001, 0.02, days)
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.005, days)),
        'high': prices * (1 + abs(np.random.normal(0, 0.01, days))),
        'low': prices * (1 - abs(np.random.normal(0, 0.01, days))),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, days),
    })
    
    return data


def test_ml_predictor():
    """머신러닝 모델 테스트"""
    print("=" * 60)
    print("머신러닝 모델 테스트")
    print("=" * 60)
    
    # 샘플 데이터 생성
    data = generate_sample_data(200)
    
    # 모델 생성 및 학습
    model = MLPredictor(lookback_window=60, model_type="lstm")
    model.train(data)
    
    # 예측
    recent_data = data.tail(60)
    current_price = data['close'].iloc[-1]
    
    predicted_price = model.predict(recent_data)
    direction = model.predict_direction(recent_data, current_price)
    
    print(f"현재가: {current_price:.2f}")
    print(f"예측가: {predicted_price:.2f}")
    print(f"방향성: {direction}")
    print()


def test_portfolio():
    """포트폴리오 테스트"""
    print("=" * 60)
    print("포트폴리오 테스트")
    print("=" * 60)
    
    # 포트폴리오 생성
    portfolio = Portfolio(initial_capital=10000000, max_position_size=0.3)
    
    # 포지션 추가
    portfolio.add_position("005930", 100, 70000)  # 삼성전자 100주 @ 70,000원
    portfolio.add_position("000660", 50, 100000)  # SK하이닉스 50주 @ 100,000원
    
    # 포트폴리오 현재 상태
    prices = {
        "005930": 72000,  # 현재가
        "000660": 102000,
    }
    
    # 요약
    summary = portfolio.get_portfolio_summary(prices)
    print(summary)
    
    # 통계
    stats = portfolio.get_statistics(prices)
    print(f"\n총 자산: {stats['total_value']:,.0f}원")
    print(f"수익/손실: {stats['total_profit_loss']:,.0f}원 ({stats['total_profit_loss_percent']:.2f}%)")
    print()


def test_risk_manager():
    """위험 관리 테스트"""
    print("=" * 60)
    print("위험 관리 테스트")
    print("=" * 60)
    
    # 위험 관리자 생성
    risk_mgr = RiskManager(stop_loss_percent=0.05, take_profit_percent=0.15)
    
    # 손실제한 설정
    entry_price = 100000
    risk_mgr.set_stop_loss("005930", entry_price)
    risk_mgr.set_take_profit("005930", entry_price)
    
    # 다양한 가격에서 확인
    test_prices = [100000, 99000, 98000, 97500, 115000, 120000]
    
    for price in test_prices:
        exit_reason = risk_mgr.check_exit_conditions("005930", price)
        print(f"현재가: {price:,}원 → {exit_reason if exit_reason else '보유'}")
    print()


def test_strategy():
    """거래 전략 테스트"""
    print("=" * 60)
    print("기술적 거래 전략 테스트")
    print("=" * 60)
    
    # 샘플 데이터 생성
    data = generate_sample_data(200)
    
    # 전략 생성
    strategy = TechnicalStrategy(lookback_window=60)
    
    # 신호 생성
    signal = strategy.generate_signal("005930", data)
    
    print(f"종목: {signal.symbol}")
    print(f"신호: {signal.action}")
    print(f"신뢰도: {signal.confidence:.2f}")
    print(f"이유: {signal.reason}")
    print()


if __name__ == "__main__":
    test_ml_predictor()
    test_portfolio()
    test_risk_manager()
    test_strategy()
    
    print("=" * 60)
    print("모든 테스트 완료")
    print("=" * 60)
