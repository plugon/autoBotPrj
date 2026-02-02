import json
import os
import pandas as pd
import sys
from datetime import datetime, timedelta

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import platform
    PLOT_AVAILABLE = True
except Exception as e:
    print(f"⚠️ 그래프 라이브러리(matplotlib)를 불러올 수 없습니다: {e}")
    print("   (그래프 기능이 비활성화됩니다. 'pip install matplotlib'로 설치 가능)")
    PLOT_AVAILABLE = False

def load_portfolio(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"파일 로드 오류 ({filepath}): {e}")
        return None

def analyze(filepath, name, start_date=None, end_date=None):
    print(f"\n{'='*20} {name} 성과 분석 {'='*20}")
    data = load_portfolio(filepath)
    
    if not data:
        print("❌ 데이터 파일이 없습니다. 봇이 한 번이라도 실행되었는지 확인하세요.")
        return None

    initial = data.get("initial_capital", 0)
    current_cash = data.get("current_capital", 0)
    positions = data.get("positions", {})
    entry_prices = data.get("entry_prices", {})
    history = data.get("trade_history", [])
    
    # 1. 자산 현황
    # (현재가 정보가 없으므로 매수가 기준으로 평가금액 추정)
    holdings_value = sum(positions[sym] * entry_prices.get(sym, 0) for sym in positions)
    total_equity = current_cash + holdings_value
    total_return = total_equity - initial
    return_pct = (total_return / initial * 100) if initial > 0 else 0
    
    print(f"💰 초기 자본: {initial:,.0f}원")
    print(f"💰 현재 자산: {total_equity:,.0f}원 (현금 {current_cash:,.0f}원 + 보유평가 {holdings_value:,.0f}원)")
    print(f"📊 누적 손익: {total_return:,.0f}원 ({return_pct:+.2f}%)")
    print(f"📦 보유 종목: {len(positions)}개")

    # 2. 거래 기록 분석
    if not history:
        print("\n⚠️ 아직 완료된 거래(매도) 내역이 없습니다.")
        return None

    df = pd.DataFrame(history)
    
    # 날짜 필터링
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if start_date:
            df = df[df['timestamp'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['timestamp'] < pd.to_datetime(end_date) + timedelta(days=1)]
            
    if df.empty:
        print(f"\n⚠️ 선택한 기간({start_date or '전체'} ~ {end_date or '전체'})에 거래 내역이 없습니다.")
        return None
    
    # 승/패 구분
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]
    
    win_count = len(wins)
    loss_count = len(losses)
    total_trades = len(df)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    
    avg_win = wins['pnl'].mean() if not wins.empty else 0
    avg_loss = losses['pnl'].mean() if not losses.empty else 0
    
    # Profit Factor (총 이익 / 총 손실)
    gross_profit = wins['pnl'].sum()
    gross_loss = abs(losses['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # MDD 계산
    df = df.sort_values('timestamp')
    df['cumulative_pnl'] = df['pnl'].cumsum()
    equity_curve = initial + df['cumulative_pnl']
    peak = equity_curve.cummax()
    if initial > 0:
        peak = peak.clip(lower=initial)
    
    drawdown = equity_curve - peak
    mdd = drawdown.min()
    mdd_pct = (drawdown / peak * 100).min() if initial > 0 else 0.0

    print(f"\n[거래 통계] ({start_date or '전체'} ~ {end_date or '전체'})")
    print(f"📝 총 거래 횟수: {total_trades}회")
    print(f"✅ 승률 (Win Rate): {win_rate:.2f}% ({win_count}승 {loss_count}패)")
    print(f"📈 평균 수익: {avg_win:,.0f}원")
    print(f"📉 평균 손실: {avg_loss:,.0f}원")
    print(f"⚖️ 손익비 (Profit Factor): {profit_factor:.2f}")
    print(f"🌊 최대 낙폭 (MDD): {mdd:,.0f}원 ({mdd_pct:.2f}%)")
    
    # 3. 진단 및 조언
    print(f"\n[AI 진단]")
    if mdd_pct < -10.0:
        print(f"⚠️ 경고: 최대 낙폭(MDD)이 -10%를 초과했습니다 ({mdd_pct:.2f}%). 리스크 관리가 시급합니다.")
        print("   👉 솔루션: 포지션 규모(Position Size)를 줄이거나 손절 라인을 타이트하게 잡으세요.")

    if profit_factor < 1.0:
        print("⚠️ 손실이 이익보다 큽니다. 현재 전략은 장기적으로 자산을 감소시킵니다.")
        print("   👉 솔루션: 손절(Stop Loss) 퍼센트를 줄이거나, 진입 조건을 더 까다롭게 설정하세요.")
    elif profit_factor < 1.5:
        print("⚠️ 수익이 나고 있으나 불안정합니다. 거래 수수료를 고려하면 실제로는 손실일 수 있습니다.")
        print("   👉 솔루션: 수수료를 감안하여 익절(Take Profit) 목표를 조금 더 높이세요.")
    else:
        print("🎉 훌륭한 성과입니다! 현재 전략을 유지하면서 투자금을 조금씩 늘려보세요.")

    return df

def plot_performance(df, title):
    if not PLOT_AVAILABLE:
        return

    # 한글 폰트 설정
    system_name = platform.system()
    if system_name == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif system_name == 'Darwin':
        plt.rc('font', family='AppleGothic')
    plt.rc('axes', unicode_minus=False)

    plt.figure(figsize=(12, 6))
    
    # 날짜순 정렬
    df = df.sort_values('timestamp')
    
    # 누적 수익금 (analyze 함수에서 이미 계산되어 있음)
    plt.plot(df['timestamp'], df['cumulative_pnl'], marker='o', linestyle='-', label='누적 손익')
    
    # 0선 표시
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    plt.title(f'{title} 누적 수익 곡선')
    plt.xlabel('날짜')
    plt.ylabel('누적 손익 (KRW)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("📅 거래 내역 필터링 (YYYY-MM-DD 형식, 엔터 입력 시 전체)")
    start_input = input("시작일: ").strip()
    end_input = input("종료일: ").strip()
    
    s_date = start_input if start_input else None
    e_date = end_input if end_input else None

    crypto_df = analyze("data/crypto_portfolio.json", "암호화폐(Crypto)", s_date, e_date)
    stock_df = analyze("data/stock_portfolio.json", "국내주식(Stock)", s_date, e_date)
    
    # CSV 내보내기
    print("\n" + "="*50)
    export = input("💾 거래 내역을 CSV 파일로 저장하시겠습니까? (y/n): ").strip().lower()
    if export == 'y':
        # 파일명에 기간 정보 추가
        period_str = ""
        if s_date or e_date:
            s_str = s_date.replace("-", "") if s_date else "ALL"
            e_str = e_date.replace("-", "") if e_date else "ALL"
            period_str = f"_{s_str}_to_{e_str}"

        if crypto_df is not None and not crypto_df.empty:
            f_name = f"trade_history_crypto{period_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            crypto_df.to_csv(f_name, index=False, encoding='utf-8-sig')
            print(f"✅ 암호화폐 거래 내역 저장 완료: {f_name}")
            
        if stock_df is not None and not stock_df.empty:
            f_name = f"trade_history_stock{period_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            stock_df.to_csv(f_name, index=False, encoding='utf-8-sig')
            print(f"✅ 주식 거래 내역 저장 완료: {f_name}")
            
        if (crypto_df is None or crypto_df.empty) and (stock_df is None or stock_df.empty):
            print("⚠️ 저장할 거래 내역이 없습니다.")
            
    # 그래프 그리기
    if PLOT_AVAILABLE:
        print("\n" + "="*50)
        show_plot = input("📈 수익률 그래프를 보시겠습니까? (y/n): ").strip().lower()
        if show_plot == 'y':
            if crypto_df is not None and not crypto_df.empty:
                plot_performance(crypto_df, "암호화폐(Crypto)")
            if stock_df is not None and not stock_df.empty:
                plot_performance(stock_df, "국내주식(Stock)")
            
            if (crypto_df is None or crypto_df.empty) and (stock_df is None or stock_df.empty):
                print("⚠️ 표시할 데이터가 없습니다.")
    
    input("\n엔터 키를 누르면 종료합니다...")
