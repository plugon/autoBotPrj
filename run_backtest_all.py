"""
모든 전략 조합에 대해 백테스팅을 수행하고 최적의 전략을 찾는 스크립트
"""
import sys
import os
import logging
import itertools
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from config.settings import TRADING_CONFIG, API_CONFIG, ML_CONFIG
from api.crypto_api import UpbitAPI, BinanceAPI
from utils.backtesting import WalkForwardAnalyzer
from trading.strategy import TechnicalStrategy
from trading.strategy_v2 import HeikinAshiStrategy
from trading.turtle_bollinger_strategy import TurtleBollingerStrategy
from trading.agile_strategy import AgileStrategy
from trading.volume_trend_strategy import VolumeTrendStrategy
from trading.ma_trend_strategy import MATrendStrategy
from trading.early_bird_strategy import EarlyBirdStrategy
from utils.logger import setup_logger

# 환경변수 로드
load_dotenv()

def send_telegram_report(message):
    """텔레그램으로 백테스트 결과 전송"""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id: return
    
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'}
        requests.post(url, data=data, timeout=5)
    except Exception as e:
        print(f"텔레그램 전송 실패: {e}")

def update_env_file(updates: dict):
    """Update .env file with best strategies"""
    try:
        # [수정] 빌드 환경 호환 절대 경로 사용
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(os.path.abspath(sys.executable))
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(base_dir, ".env")
        
        print(f"📂 .env 파일 경로: {env_path}")
        
        lines = []
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        
        new_lines = []
        processed_keys = set()
        
        for line in lines:
            # 빈 줄이나 주석은 그대로 유지
            if not line.strip() or line.strip().startswith('#'):
                new_lines.append(line)
                continue
            
            if '=' in line:
                key = line.split('=')[0].strip()
                if key in updates:
                    new_lines.append(f"{key}={updates[key]}\n")
                    processed_keys.add(key)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        # 없는 키 추가
        for key, value in updates.items():
            if key not in processed_keys:
                # 마지막 줄이 개행문자로 끝나지 않으면 추가
                if new_lines and not new_lines[-1].endswith('\n'):
                    new_lines[-1] += '\n'
                new_lines.append(f"{key}={value}\n")
        
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
            
    except Exception as e:
        print(f"❌ .env 파일 업데이트 오류: {e}")

# 전략 래퍼 (로그용)
class StrategyWrapper:
    def __init__(self, strategy):
        self.strategy = strategy
        
    def __getattr__(self, name):
        return getattr(self.strategy, name)

    def generate_signal(self, symbol, data, current_capital=0.0, strategy_override=None):
        # 전략 내부에서 strategy_override를 사용할 수 있도록 전달
        signal = self.strategy.generate_signal(symbol, data, current_capital, strategy_override)
        return signal

def get_strategy_instance(strategy_type, lookback):
    """전략 객체 생성 팩토리"""
    if strategy_type == "heikin_ashi":
        return HeikinAshiStrategy(lookback_window=lookback)
    elif strategy_type == "turtle_bollinger":
        return TurtleBollingerStrategy(lookback_window=lookback)
    elif strategy_type == "agile":
        return AgileStrategy(lookback_window=lookback)
    elif strategy_type == "volume_trend":
        return VolumeTrendStrategy(lookback_window=lookback)
    elif strategy_type == "ma_trend":
        return MATrendStrategy(lookback_window=lookback)
    elif strategy_type == "early_bird":
        return EarlyBirdStrategy(lookback_window=lookback)
    else:
        return TechnicalStrategy(lookback_window=lookback)

def run_comparison(api_config_key, api_class, api_key_env, api_secret_env):
    """특정 거래소 설정에 대해 모든 전략 테스트"""
    conf = TRADING_CONFIG.get(api_config_key)
    if not conf: return []

    api_key = os.getenv(api_key_env)
    api_secret = os.getenv(api_secret_env)
    if not api_key or not api_secret: return []

    try:
        if api_config_key == "binance_futures":
            api = api_class(api_key, api_secret, account_type='future')
        elif api_config_key == "binance_spot":
            api = api_class(api_key, api_secret, account_type='spot')
        else:
            api = api_class(api_key, api_secret)
        api.connect()
    except Exception as e:
        print(f"API Connection Error: {e}")
        return []

    symbols = conf.get("symbols", [])
    if not symbols: return []
    
    # 테스트할 전략 목록 정의
    strategies_to_test = [
        {"name": "Agile (스캘핑)", "type": "agile", "entry": None},
        {"name": "TurtleBollinger (추세)", "type": "turtle_bollinger", "entry": None},
        {"name": "VolumeTrend (거래량추세)", "type": "volume_trend", "entry": None},
        {"name": "MATrend (이평선추세)", "type": "ma_trend", "entry": None},
        {"name": "EarlyBird (선취매)", "type": "early_bird", "entry": None},
        {"name": "HeikinAshi (추세)", "type": "heikin_ashi", "entry": None},
        {"name": "Tech_Breakout (변동성돌파)", "type": "technical", "entry": "breakout"},
        {"name": "Tech_Combined (종합)", "type": "technical", "entry": "combined"},
        {"name": "Tech_RSI_BB (역추세)", "type": "technical", "entry": "rsi_bollinger"},
        {"name": "Tech_Pullback (눌림목)", "type": "technical", "entry": "pullback"},
    ]

    timeframe = conf.get("timeframe", "15m")
    
    # 데이터 미리 로드 (API 호출 최소화)
    market_data = {}
    print(f"\n[{api_config_key.upper()}] 데이터 수집 중... ({len(symbols)} 종목, Timeframe: {timeframe})")
    
    for symbol in symbols:
        try:
            # 충분한 데이터 확보 (기존 1000개 -> 5000개 상향)
            count = 5000
            df = api.get_ohlcv(symbol, timeframe=timeframe, count=count)
            if not df.empty and len(df) > 200:
                market_data[symbol] = df
                print(f"  - {symbol}: {len(df)} 캔들 수집 완료")
        except Exception as e:
            print(f"  - {symbol}: 수집 실패 ({e})")

    results = []

    # Config 백업 (TechnicalStrategy가 전역 설정을 참조하므로 임시 수정 필요)
    original_crypto_conf = TRADING_CONFIG["crypto"].copy()

    # [New] 파라미터 최적화 후보군 정의
    # 스캘핑용 (Agile)
    scalping_tp = [0.015, 0.02, 0.03]
    scalping_sl = [0.005, 0.01, 0.015]
    # 추세용 (Trend)
    trend_tp = [0.03, 0.05, 0.08, 0.12]
    trend_sl = [0.01, 0.02, 0.04]

    print(f"\n🚀 전략별 백테스팅 시작...")
    
    for strat_conf in strategies_to_test:
        strat_name = strat_conf["name"]
        strat_type = strat_conf["type"]
        entry_strategy = strat_conf["entry"]
        
        # 설정 패치 (TechnicalStrategy가 참조하는 값 수정)
        if entry_strategy:
            TRADING_CONFIG["crypto"]["entry_strategy"] = entry_strategy
        
        # 전략 타입에 따른 파라미터 후보군 선택
        if strat_type == "agile":
            tp_candidates = scalping_tp
            sl_candidates = scalping_sl
        else:
            tp_candidates = trend_tp
            sl_candidates = trend_sl
            
        # 전략 인스턴스 생성
        strategy = get_strategy_instance(strat_type, ML_CONFIG["lookback_window"])
        strategy = StrategyWrapper(strategy)

        # 수수료 설정
        fee_rate = TRADING_CONFIG["fees"].get("binance_fee_rate" if "binance" in api_config_key else "crypto_fee_rate", 0.001)

        # [New] 파라미터 조합별 테스트 (Grid Search)
        for tp, sl in itertools.product(tp_candidates, sl_candidates):
            total_pnl = 0
            total_trades = 0
            total_wins = 0

            for symbol, df in market_data.items():
                # Lookback 계산
                if timeframe == "1m": lookback = 3000
                elif timeframe in ["3m", "5m"]: lookback = 1000
                elif timeframe == "15m": lookback = 300
                else: lookback = 100
                
                # 테스트 기간
                train_period = 200
                test_len = len(df) - train_period 
                if test_len < 10: continue
                
                analyzer = WalkForwardAnalyzer(
                    df,
                    symbol=symbol,
                    train_period=train_period,
                    test_period=test_len,
                    slippage=0.001,
                    fee=fee_rate,
                    stop_loss=sl,  # 최적화 대상
                    take_profit=tp, # 최적화 대상
                    trailing_stop=conf.get("trailing_stop_percent", 0.0)
                )
                
                res = analyzer.run(strategy_type=strategy, lookback_window=lookback)
                
                if not res.empty:
                    sym_return = res['total_return'].sum()
                    sym_trades = res['trade_count'].sum()
                    sym_wins = (res['win_rate'] * res['trade_count']).sum()
                    
                    total_pnl += sym_return
                    total_trades += sym_trades
                    total_wins += sym_wins

            avg_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
            
            # 결과 저장 (파라미터 포함)
            results.append({
                "Exchange": api_config_key,
                "Strategy": strat_name,
                "Total PnL": total_pnl,
                "Trades": total_trades,
                "Win Rate": avg_win_rate,
                "TP": tp,
                "SL": sl
            })
            
            # 진행 상황 출력 (간략히)
            # print(f"   - {strat_name} (TP:{tp:.1%}, SL:{sl:.1%}) -> 수익: {total_pnl:,.0f}")

        # 해당 전략의 최고 성과 출력
        strat_results = [r for r in results if r["Strategy"] == strat_name and r["Exchange"] == api_config_key]
        if strat_results:
            best_strat = max(strat_results, key=lambda x: x["Total PnL"])
            print(f"   👉 {strat_name:<20} | 수익: {best_strat['Total PnL']:>10,.0f} | 승률: {best_strat['Win Rate']:>5.1f}% | TP: {best_strat['TP']:.1%} / SL: {best_strat['SL']:.1%}")

    # Config 원복
    TRADING_CONFIG["crypto"] = original_crypto_conf
    
    # 결과에 매핑 정보 추가 (env 업데이트용)
    for res in results:
        # 전략 이름에서 내부 코드 매핑
        s_name = res["Strategy"]
        if "Agile" in s_name: code = "agile"
        elif "Turtle" in s_name: code = "turtle_bollinger"
        elif "Heikin" in s_name: code = "heikin_ashi"
        elif "VolumeTrend" in s_name: code = "volume_trend"
        elif "MATrend" in s_name: code = "ma_trend"
        elif "EarlyBird" in s_name: code = "early_bird"
        elif "Breakout" in s_name: code = "breakout"
        elif "Combined" in s_name: code = "combined"
        elif "RSI_BB" in s_name: code = "rsi_bollinger"
        elif "Pullback" in s_name: code = "pullback"
        else: code = "breakout"
        res["StrategyCode"] = code

    return results

def main():
    # 로거 설정 (파일로만 저장)
    setup_logger("backtest_all", filename="backtest_all.log")
    
    all_results = []
    
    print("=" * 80)
    print("🧪 전체 전략 비교 백테스팅 (All Strategy Backtest)")
    print("=" * 80)
    
    # 1. Upbit
    if API_CONFIG["upbit"]:
        res = run_comparison("crypto", UpbitAPI, "UPBIT_API_KEY", "UPBIT_API_SECRET")
        all_results.extend(res)
        
    # 2. Binance Spot
    if API_CONFIG["binance_spot"]:
        res = run_comparison("binance_spot", BinanceAPI, "BINANCE_API_KEY", "BINANCE_API_SECRET")
        all_results.extend(res)

    # 3. Binance Futures
    if API_CONFIG["binance_futures"]:
        res = run_comparison("binance_futures", BinanceAPI, "BINANCE_API_KEY", "BINANCE_API_SECRET")
        all_results.extend(res)

    # 결과 출력
    if all_results:
        df_res = pd.DataFrame(all_results)
        # PnL 기준 내림차순 정렬
        df_res = df_res.sort_values(by="Total PnL", ascending=False)
        
        print("\n" + "="*80)
        print("🏆 백테스팅 전략 비교 결과 (수익금 순)")
        print("="*80)
        
        # 출력 포맷팅
        print(f"{'Exchange':<15} {'Strategy':<25} {'Total PnL':>15} {'Win Rate':>10} {'Trades':>8}")
        print("-" * 80)
        for _, row in df_res.iterrows():
            pnl_str = f"{row['Total PnL']:,.0f}"
            win_str = f"{row['Win Rate']:.1f}%"
            print(f"{row['Exchange']:<15} {row['Strategy']:<25} {pnl_str:>15} {win_str:>10} {row['Trades']:>8}")
        print("=" * 80)
        
        # 거래소별 최적 전략 선정 및 업데이트
        print(f"\n🌟 [거래소별 최적 전략 선정]")
        env_updates = {}
        report_lines = []
        
        for exchange in ["crypto", "binance_spot", "binance_futures"]:
            # 해당 거래소 결과만 필터링
            ex_results = df_res[df_res['Exchange'] == exchange]
            if ex_results.empty:
                continue
                
            # 수익금 기준 1위 선정
            best = ex_results.iloc[0]
            
            # 수익이 0 이하면 변경하지 않음 (안전장치)
            if best['Total PnL'] <= 0:
                print(f"   - {exchange}: ⚠️ 업데이트 스킵 (최고 수익 {best['Total PnL']:,.0f} <= 0)")
                continue
                
            print(f"   - {exchange}: {best['Strategy']} (수익 {best['Total PnL']:,.0f})")
            
            # Env 키 매핑
            if exchange == "crypto":
                env_key = "CRYPTO_ENTRY_STRATEGY"
            elif exchange == "binance_spot":
                env_key = "BINANCE_SPOT_ENTRY_STRATEGY"
            elif exchange == "binance_futures":
                env_key = "BINANCE_FUTURES_ENTRY_STRATEGY"
            
            env_updates[env_key] = best['StrategyCode']

            # [New] 최적 파라미터(익절/손절) 업데이트
            if exchange == "crypto":
                env_updates["CRYPTO_TAKE_PROFIT"] = str(best['TP'])
                env_updates["CRYPTO_STOP_LOSS"] = str(best['SL'])
            elif exchange == "binance_spot":
                env_updates["BINANCE_SPOT_TAKE_PROFIT"] = str(best['TP'])
                env_updates["BINANCE_SPOT_STOP_LOSS"] = str(best['SL'])
            elif exchange == "binance_futures":
                env_updates["BINANCE_FUTURES_TAKE_PROFIT"] = str(best['TP'])
                env_updates["BINANCE_FUTURES_STOP_LOSS"] = str(best['SL'])
            
            print(f"     └─ 최적 파라미터 적용: 익절 {best['TP']*100:.1f}% / 손절 {best['SL']*100:.1f}%")
            
            # 리포트 라인 추가
            report_lines.append(f"✅ *{exchange.upper()}*: {best['Strategy']}\n   └ 수익: {best['Total PnL']:,.0f} / 승률: {best['Win Rate']:.1f}%")
            report_lines.append(f"   └ 설정: TP {best['TP']*100:.1f}% / SL {best['SL']*100:.1f}%")

        if env_updates:
            # [New] 환경변수로 업데이트 여부 제어 (기본값: True)
            auto_update = os.getenv("AUTO_UPDATE_ENV", "true").lower() in ["true", "1", "yes", "on"]

            if auto_update:
                print("\n🔄 .env 파일 업데이트를 진행합니다...")
                update_env_file(env_updates)
                print("✅ 업데이트 완료! 봇을 재시작하면 적용됩니다.")
            else:
                print("\n🛑 [AUTO_UPDATE_ENV=False] .env 파일 업데이트를 건너뜁니다.")

            for k, v in env_updates.items():
                print(f"   👉 {k}={v}")
            
            # 텔레그램 전송
            if report_lines:
                status_msg = "설정이 갱신되었습니다." if auto_update else "설정 업데이트를 건너뛰었습니다."
                msg = f"🧪 *[전략 최적화 결과]*\n새로운 시장 상황에 맞춰 {status_msg}\n\n" + "\n".join(report_lines)
                send_telegram_report(msg)
        else:
            print("\nℹ️ 업데이트할 최적 전략이 없습니다.")
        
        # CSV 저장
        csv_file = f"backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_res.to_csv(csv_file, index=False)
        print(f"\n💾 상세 결과가 {csv_file} 파일로 저장되었습니다.")
    else:
        print("\n❌ 백테스팅 결과가 없습니다. API 설정이나 데이터를 확인하세요.")

if __name__ == "__main__":
    main()
