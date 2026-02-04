import streamlit as st
import pandas as pd
import json
import os
import time
import sys
import subprocess
from datetime import datetime
import requests
import pyupbit  # 실시간 시세 조회를 위해 추가
from config.settings import STRATEGY_PRESETS

# 페이지 설정
st.set_page_config(
    page_title="자동매매 봇 대시보드",
    page_icon="📈",
    layout="wide"
)

st.title("🤖 자동매매 봇 현황 리포트")

# 데이터 로드 함수
def load_data(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        return None

def calculate_metrics(history):
    """거래 기록을 바탕으로 승률과 손익비를 계산"""
    if not history:
        return 0.0, 0.0, 0
    
    wins = [t['pnl'] for t in history if t['pnl'] > 0]
    losses = [t['pnl'] for t in history if t['pnl'] <= 0]
    
    win_rate = (len(wins) / len(history) * 100)
    
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
    
    return win_rate, profit_factor, len(history)

def get_bot_status():
    """봇 상태 파일 읽기"""
    status_file = "data/bot_status.json"
    if not os.path.exists(status_file):
        return {"status": "stopped", "timestamp": 0}
    
    try:
        with open(status_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        timestamp = data.get("timestamp", 0)
        
        # 15초 이상 업데이트 없으면 중지로 간주 (봇이 죽었거나 멈춤)
        if time.time() - timestamp > 15:
            data["status"] = "stopped"
            
        return data
    except:
        return {"status": "stopped", "timestamp": 0}

# 사이드바 설정
st.sidebar.header("설정")
auto_refresh = st.sidebar.checkbox("자동 새로고침 (3초)", value=True)

st.sidebar.divider()
st.sidebar.header("🤖 봇 제어")

# 상태 표시등
status_data = get_bot_status()
status = status_data.get("status", "stopped")
last_heartbeat = status_data.get("timestamp", 0)

if status == "running":
    st.sidebar.success(f"🟢 **봇 가동 중 (Running)**")
    st.sidebar.caption(f"Last heartbeat: {datetime.fromtimestamp(last_heartbeat).strftime('%H:%M:%S')}")
    # CPU 및 메모리 사용량 표시
    cpu = status_data.get("cpu", 0.0)
    memory = status_data.get("memory", 0.0)
    col_cpu, col_mem = st.sidebar.columns(2)
    col_cpu.metric("CPU", f"{cpu:.1f}%")
    col_mem.metric("Memory", f"{memory:.0f} MB")
elif status == "warming_up":
    st.sidebar.warning(f"🟡 **웜업 중 (Warming Up)**")
    
    # 웜업 진행률 표시
    w_curr = status_data.get("warmup_current", 0)
    w_total = status_data.get("warmup_total", 3)
    if w_total > 0:
        st.sidebar.progress(min(w_curr / w_total, 1.0))
        st.sidebar.caption(f"진행률: {w_curr}/{w_total}")

    st.sidebar.caption(f"Last heartbeat: {datetime.fromtimestamp(last_heartbeat).strftime('%H:%M:%S')}")
    # CPU 및 메모리 사용량 표시
    cpu = status_data.get("cpu", 0.0)
    memory = status_data.get("memory", 0.0)
    col_cpu, col_mem = st.sidebar.columns(2)
    col_cpu.metric("CPU", f"{cpu:.1f}%")
    col_mem.metric("Memory", f"{memory:.0f} MB")
elif status == "restarting":
    st.sidebar.warning(f"🟠 **재시작 중...**")
else:
    st.sidebar.error(f"🔴 **봇 중지됨 (Stopped)**")
    if last_heartbeat > 0:
        st.sidebar.caption(f"Last seen: {datetime.fromtimestamp(last_heartbeat).strftime('%H:%M:%S')}")

st.sidebar.divider()

# 사용 가능한 전략 목록 가져오기
strategy_options = list(STRATEGY_PRESETS.keys())
selected_strategy = st.sidebar.selectbox(
    "전략 변경",
    options=strategy_options,
    help="봇의 거래 전략을 실시간으로 변경합니다. 변경된 전략은 다음 거래부터 적용됩니다."
)

if st.sidebar.button("전략 적용하기"):
    command_file = "data/command.json"
    command_data = {
        "command": "change_strategy",
        "payload": selected_strategy,
        "timestamp": time.time()
    }
    with open(command_file, 'w', encoding='utf-8') as f:
        json.dump(command_data, f)
    st.sidebar.success(f"✅ '{selected_strategy}' 전략으로 변경 명령을 보냈습니다.")
    time.sleep(1) # 봇이 처리할 시간 확보

st.sidebar.divider()

if status == "stopped":
    if st.sidebar.button("▶️ 봇 시작", use_container_width=True):
        try:
            if getattr(sys, 'frozen', False):
                # EXE 환경: 현재 실행 파일(Dashboard.exe)과 같은 폴더의 TradingBot.exe 실행
                base_dir = os.path.dirname(sys.executable)
                bot_path = os.path.join(base_dir, "TradingBot.exe")
                if os.path.exists(bot_path):
                    subprocess.Popen([bot_path], cwd=base_dir)
                    st.sidebar.success("봇을 시작했습니다. 잠시 후 상태가 갱신됩니다.")
                    time.sleep(3)
                    st.rerun()
                else:
                    st.sidebar.error("TradingBot.exe를 찾을 수 없습니다.")
            else:
                # 개발 환경: python main.py 실행
                base_dir = os.path.dirname(os.path.abspath(__file__))
                main_py = os.path.join(base_dir, "main.py")
                subprocess.Popen([sys.executable, main_py], cwd=base_dir)
                st.sidebar.success("봇을 시작했습니다.")
                time.sleep(3)
                st.rerun()
        except Exception as e:
            st.sidebar.error(f"봇 시작 실패: {e}")

else:
    col1, col2 = st.sidebar.columns(2)

    if col1.button("🔄 봇 재시작"):
        command_file = "data/command.json"
        command_data = {
            "command": "restart_bot",
            "timestamp": time.time()
        }
        with open(command_file, 'w', encoding='utf-8') as f:
            json.dump(command_data, f)
        st.sidebar.warning("🔄 봇 재시작 명령을 보냈습니다.")

    if col2.button("🛑 봇 종료"):
        command_file = "data/command.json"
        command_data = {
            "command": "stop_bot",
            "timestamp": time.time()
        }
        with open(command_file, 'w', encoding='utf-8') as f:
            json.dump(command_data, f)
        st.sidebar.error("🛑 봇 종료 명령을 보냈습니다.")

st.sidebar.divider()

# 종료 옵션 (봇이 실행 중일 때만 표시)
stop_bot_on_exit = False
if status == "running":
    stop_bot_on_exit = st.sidebar.checkbox("🤖 실행 중인 봇도 함께 종료", value=False)

if st.sidebar.button("🚪 대시보드 종료", use_container_width=True):
    if stop_bot_on_exit and status == "running":
        command_file = "data/command.json"
        command_data = {"command": "stop_bot", "timestamp": time.time()}
        with open(command_file, 'w', encoding='utf-8') as f:
            json.dump(command_data, f)
        st.sidebar.error("🛑 봇 종료 명령을 보냈습니다.")
        time.sleep(1)

    st.sidebar.warning("대시보드를 종료합니다...")
    time.sleep(1)
    os._exit(0)

# [New] 전략별 성과 요약 테이블 (상단 배치)
st.subheader("📊 전략별 성과 요약")

summary_list = []
portfolio_files = {
    "🚀 Crypto (Upbit)": "data/crypto_portfolio.json",
    "🇰🇷 Stock (Korea)": "data/stock_portfolio.json",
    "🟡 Binance Spot": "data/binance_spot_portfolio.json",
    "🔴 Binance Futures": "data/binance_futures_portfolio.json"
}

for name, filepath in portfolio_files.items():
    p_data = load_data(filepath)
    if p_data:
        initial = p_data.get("initial_capital", 0)
        current_cash = p_data.get("current_capital", 0)
        positions = p_data.get("positions", {})
        entry_prices = p_data.get("entry_prices", {})
        
        # 추정 자산 (현재가 정보가 없으므로 평단가 기준)
        holdings_val = sum(positions[sym] * entry_prices.get(sym, 0) for sym in positions)
        total_est = current_cash + holdings_val
        
        # 누적 손익
        total_pnl = total_est - initial
        pnl_pct = (total_pnl / initial * 100) if initial > 0 else 0.0
        
        # 승률/손익비
        history = p_data.get("trade_history", [])
        win_rate, pf, trade_cnt = calculate_metrics(history)
        
        summary_list.append({
            "전략": name,
            "총 자산 (추정)": f"{total_est:,.0f}",
            "누적 손익": f"{total_pnl:,.0f} ({pnl_pct:+.1f}%)",
            "승률": f"{win_rate:.1f}%",
            "손익비": f"{pf:.2f}",
            "거래 횟수": f"{trade_cnt}회"
        })

if summary_list:
    st.dataframe(pd.DataFrame(summary_list), use_container_width=True, hide_index=True)
else:
    st.info("데이터가 없습니다.")

st.divider()

# 탭 생성
tab1, tab2, tab3, tab4 = st.tabs(["🚀 업비트 (Upbit)", "🇰🇷 국내주식 (Stock)", "🟡 바이낸스 현물", "🔴 바이낸스 선물"])

def display_portfolio(data, title, is_crypto=False):
    if not data:
        st.warning(f"{title} 데이터가 없습니다. 봇이 실행 중인지 확인하세요.")
        return

    # 0. 전략 정보 표시 (암호화폐인 경우)
    if is_crypto and "metadata" in data:
        meta = data.get("metadata", {})
        strategy = meta.get("strategy", "Unknown")
        timeframe = meta.get("timeframe", "Unknown")
        st.info(f"ℹ️ 현재 적용 전략: **{strategy.upper()}** (Timeframe: {timeframe})")

    # 1. 자산 현황 요약
    initial = data.get("initial_capital", 0)
    current_cash = data.get("current_capital", 0)
    
    # 현재 평가금액 계산 (보유 종목 가치 합산)
    positions = data.get("positions", {})
    entry_prices = data.get("entry_prices", {})
    
    # 현재가 조회 (암호화폐인 경우 pyupbit 사용)
    current_prices = {}
    if is_crypto and positions:
        try:
            tickers = list(positions.keys())
            # pyupbit로 현재가 일괄 조회
            prices = pyupbit.get_current_price(tickers)
            # 티커가 1개일 경우 float로 반환되므로 dict로 변환 처리
            if isinstance(prices, (float, int)):
                current_prices = {tickers[0]: prices}
            elif isinstance(prices, dict):
                current_prices = prices
        except Exception as e:
            st.error(f"⚠️ 시세 조회 실패: {e}")

    stock_value = 0
    for sym, qty in positions.items():
        # 현재가가 있으면 사용, 없으면 매수 평단가 사용 (보수적 계산)
        price = current_prices.get(sym, entry_prices.get(sym, 0))
        stock_value += price * qty
        
    total_equity = current_cash + stock_value
    total_pnl = total_equity - initial
    pnl_pct = (total_pnl / initial * 100) if initial > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 자산 (추정)", f"{total_equity:,.0f}원", f"{pnl_pct:+.2f}%")
    col2.metric("보유 현금", f"{current_cash:,.0f}원")
    col3.metric("보유 주식/코인 평가액", f"{stock_value:,.0f}원")
    col4.metric("누적 손익", f"{total_pnl:,.0f}원")

    st.divider()

    # 2. 현재 보유 종목
    st.subheader("📊 현재 보유 종목")
    if positions:
        pos_data = []
        for sym, qty in positions.items():
            entry = entry_prices.get(sym, 0)
            pos_data.append({
                "종목": sym,
                "보유수량": qty,
                "매수평단가": f"{entry:,.0f}원",
                "매수금액": f"{entry * qty:,.0f}원"
            })
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
    else:
        st.info("현재 보유 중인 종목이 없습니다.")

    st.divider()

    # 3. 수익률 그래프
    st.subheader("📈 누적 수익률 추이")
    history = data.get("trade_history", [])
    
    if history:
        try:
            df_graph = pd.DataFrame(history)
            df_graph['timestamp'] = pd.to_datetime(df_graph['timestamp'])
            df_graph = df_graph.sort_values('timestamp')
            
            # 누적 수익률 계산
            df_graph['cumulative_pnl'] = df_graph['pnl'].cumsum()
            initial_cap = data.get("initial_capital", 0)
            if initial_cap <= 0: initial_cap = 1  # 0으로 나누기 방지
            df_graph['return_rate'] = (df_graph['cumulative_pnl'] / initial_cap) * 100
            
            st.line_chart(df_graph.set_index('timestamp')['return_rate'])
        except Exception as e:
            st.error(f"그래프 생성 중 오류가 발생했습니다: {e}")
    else:
        st.info("거래 내역이 없어 그래프를 표시할 수 없습니다.")

    # [추가] 3-1. 일별 손익 및 상세 분석
    if history:
        st.divider()
        st.subheader("📊 상세 성과 분석")
        
        try:
            df_analysis = pd.DataFrame(history)
            df_analysis['timestamp'] = pd.to_datetime(df_analysis['timestamp'])
            df_analysis['date'] = df_analysis['timestamp'].dt.date
            
            # 1) 일별 손익 (Bar Chart)
            daily_pnl = df_analysis.groupby('date')['pnl'].sum()
            
            st.markdown("**📅 일별 손익 (Daily PnL)**")
            # 색상 구분을 위한 차트 데이터 생성 (양수: 파랑, 음수: 빨강)
            st.bar_chart(daily_pnl)
            
            # 2) 승률 분석
            wins = len(df_analysis[df_analysis['pnl'] > 0])
            losses = len(df_analysis[df_analysis['pnl'] <= 0])
            total = wins + losses
            win_rate = (wins / total * 100) if total > 0 else 0
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("총 거래 횟수", f"{total}회")
            col_b.metric("승리 (Win)", f"{wins}회", f"{win_rate:.1f}%")
            col_c.metric("패배 (Loss)", f"{losses}회")
            
        except Exception as e:
            st.error(f"상세 분석 중 오류: {e}")

    st.divider()

    # [추가] 3-2. 일별 자산 변동 및 MDD 차트
    daily_history = data.get("daily_history", [])
    if daily_history:
        st.subheader("📅 일별 자산 변동 및 MDD")
        
        try:
            df_daily = pd.DataFrame(daily_history)
            df_daily['date'] = pd.to_datetime(df_daily['date'])
            df_daily = df_daily.sort_values('date')
            df_daily.set_index('date', inplace=True)
            
            # 일별 수익률 계산
            df_daily['daily_return'] = df_daily['total_value'].pct_change() * 100
            df_daily['daily_return'] = df_daily['daily_return'].fillna(0)
            
            # MDD 계산 (Drawdown Series)
            df_daily['peak'] = df_daily['total_value'].cummax()
            df_daily['drawdown'] = (df_daily['total_value'] - df_daily['peak']) / df_daily['peak'] * 100
            
            # 차트 1: 자산 추이 & MDD (영역 차트)
            st.markdown("**📉 자산 추이 및 Drawdown**")
            
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                st.caption("자산 가치 (Total Value)")
                st.line_chart(df_daily['total_value'])
                
            with col_d2:
                st.caption("Drawdown (%)")
                st.area_chart(df_daily['drawdown'], color="#ff4b4b")

            # 차트 2: 일별 수익률 (Bar Chart)
            st.markdown("**📊 일별 수익률 (Daily Return %)**")
            st.bar_chart(df_daily['daily_return'])
            
        except Exception as e:
            st.error(f"일별 데이터 시각화 오류: {e}")

    st.divider()

    # 4. 최근 거래 내역
    st.subheader("📝 최근 거래 내역")
    if history:
        # 최신순 정렬
        history_rev = history[::-1]
        
        df_hist = pd.DataFrame(history_rev)
        
        # 컬럼 매핑
        cols_map = {
            'timestamp': '시간',
            'symbol': '종목',
            'type': '유형',
            'quantity': '수량',
            'entry_price': '진입가',
            'exit_price': '청산가',
            'pnl': '손익금',
            'pnl_percent': '수익률(%)'
        }
        
        # 존재하는 컬럼만 선택
        available_cols = [c for c in cols_map.keys() if c in df_hist.columns]
        df_display = df_hist[available_cols].rename(columns=cols_map)
        
        # 스타일링
        st.dataframe(
            df_display.style.format({
                '진입가': '{:,.0f}',
                '청산가': '{:,.0f}',
                '손익금': '{:,.0f}',
                '수익률(%)': '{:+.2f}'
            }, na_rep="-").map(
                lambda x: 'color: blue' if isinstance(x, (int, float)) and x < 0 else ('color: red' if isinstance(x, (int, float)) and x > 0 else ''), 
                subset=['손익금', '수익률(%)']
            ),
            use_container_width=True
        )
    else:
        st.info("아직 거래 내역이 없습니다.")

def display_logs():
    st.divider()
    col_head, col_btn = st.columns([6, 1])
    with col_head:
        st.subheader("📜 실시간 시스템 로그 (Live Logs)")
    log_dir = "logs"
    
    if not os.path.exists(log_dir):
        st.caption("로그 폴더가 없습니다.")
        return

    # 가장 최근 로그 파일 찾기
    log_files = [f for f in os.listdir(log_dir) if f.startswith("trading_") and f.endswith(".log")]
    if not log_files:
        st.caption("로그 파일이 없습니다.")
        return
    
    latest_log = sorted(log_files)[-1]
    log_path = os.path.join(log_dir, latest_log)
    
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            log_content = f.read()

        with col_btn:
            st.download_button(
                label="📥 다운로드",
                data=log_content,
                file_name=latest_log,
                mime="text/plain",
                use_container_width=True
            )
            
        # 마지막 50줄만 읽어서 표시
        lines = log_content.splitlines()[-50:]
            
        # 로그 컨테이너 스타일 (터미널 느낌)
        log_html = """
        <div style="
            height: 300px; 
            overflow-y: auto; 
            background-color: #1e1e1e; 
            color: #d4d4d4; 
            padding: 10px; 
            border-radius: 5px; 
            font-family: 'Consolas', 'Monaco', monospace; 
            font-size: 12px; 
            line-height: 1.4;
            border: 1px solid #333;
        ">
        """
        
        for line in lines:
            line = line.rstrip()
            if not line:
                continue
                
            color = "#d4d4d4" # 기본 텍스트 색상
            font_weight = "normal"
            
            if "ERROR" in line:
                color = "#f44336" # 빨간색
                font_weight = "bold"
            elif "WARNING" in line:
                color = "#ff9800" # 주황색
                font_weight = "bold"
            elif "INFO" in line:
                color = "#4caf50" # 초록색
            elif "DEBUG" in line:
                color = "#2196f3" # 파란색
                
            # HTML 이스케이프 처리
            line_esc = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            
            log_html += f'<div style="color: {color}; font-weight: {font_weight}; white-space: pre-wrap;">{line_esc}</div>'
            
        log_html += "</div>"
        
        st.caption(f"파일명: {latest_log}")
        st.markdown(log_html, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"로그 파일 읽기 오류: {e}")

def display_watchlist(data):
    """관심 종목(선정된 종목) 실시간 시세 표시"""
    if not data:
        return

    metadata = data.get("metadata", {})
    selected_symbols = metadata.get("selected_symbols", [])
    
    if not selected_symbols:
        return

    st.divider()
    st.subheader("👀 선정된 관심 종목 (Watchlist)")
    
    try:
        # 업비트 API로 일괄 조회 (효율성)
        url = "https://api.upbit.com/v1/ticker"
        markets = ",".join(selected_symbols)
        response = requests.get(url, params={"markets": markets}, timeout=2)
        
        if response.status_code == 200:
            tickers = response.json()
            
            # 5개씩 2줄로 표시 (최대 10개 가정)
            cols = st.columns(5)
            for i, ticker in enumerate(tickers):
                symbol = ticker['market']
                price = ticker['trade_price']
                change_rate = ticker['signed_change_rate'] * 100
                
                with cols[i % 5]:
                    st.metric(
                        label=symbol,
                        value=f"{price:,.0f}원",
                        delta=f"{change_rate:+.2f}%"
                    )
    except Exception as e:
        st.error(f"시세 조회 오류: {e}")

with tab1:
    data = load_data("data/crypto_portfolio.json")
    display_portfolio(data, "암호화폐", is_crypto=True)
    display_watchlist(data)

with tab2:
    data = load_data("data/stock_portfolio.json")
    display_portfolio(data, "국내주식", is_crypto=False)

with tab3:
    data = load_data("data/binance_spot_portfolio.json")
    display_portfolio(data, "바이낸스 현물", is_crypto=True)

with tab4:
    data = load_data("data/binance_futures_portfolio.json")
    display_portfolio(data, "바이낸스 선물", is_crypto=True)

# 로그 표시 (전체 탭 공통 하단)
display_logs()

# 푸터
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 자동 새로고침 로직 (마지막에 배치)
if auto_refresh:
    time.sleep(3)
    st.rerun()
