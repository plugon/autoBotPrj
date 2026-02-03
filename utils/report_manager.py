import logging
import pandas as pd
import requests
from datetime import datetime, timedelta
from trading.strategy import TechnicalStrategy
from trading.strategy_v2 import HeikinAshiStrategy
from trading.turtle_bollinger_strategy import TurtleBollingerStrategy
from utils.backtesting import WalkForwardAnalyzer
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

class ReportManager:
    """전략 성과 분석 및 리포팅 매니저"""
    
    def __init__(self, api):
        self.api = api
        self.strategies = {
            "Breakout": TechnicalStrategy(lookback_window=200),
            "HeikinAshi": HeikinAshiStrategy(lookback_window=200),
            "TurtleBollinger": TurtleBollingerStrategy(lookback_window=200)
        }

    def send_telegram_message(self, message: str):
        """텔레그램 메시지 전송"""
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            # 설정이 없으면 로그만 남기고 리턴
            return

        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                "chat_id": TELEGRAM_CHAT_ID, 
                "text": message, 
                "parse_mode": "Markdown"
            }
            response = requests.post(url, data=data, timeout=5)
            if response.status_code != 200:
                logger.error(f"텔레그램 전송 실패: {response.text}")
        except Exception as e:
            logger.error(f"텔레그램 전송 오류: {e}")

    def report_portfolio_status(self, portfolio, exchange_name="Crypto", api=None):
        """현재 포트폴리오 상태를 텔레그램으로 전송"""
        # [수정] 보유 종목이 없어도 현금 잔고 보고를 위해 체크 제거
        
        # 사용할 API 결정 (전달받은 api가 없으면 기본 self.api 사용)
        target_api = api if api else self.api
        
        # 거래소별 통화 및 소수점 설정
        is_binance = "BINANCE" in exchange_name.upper()
        currency = "USDT" if is_binance else "원"
        precision = 2 if is_binance else 0

        try:
            message = f"📊 *[{exchange_name}] 자산 현황 리포트*\n\n"
            total_pnl = 0
            total_value = 0
            
            if portfolio.positions:
                for symbol, quantity in portfolio.positions.items():
                    # 현재가 조회
                    current_price = target_api.get_price(symbol)
                    entry_price = portfolio.entry_prices.get(symbol, 0)
                    
                    # 가치 합산
                    total_value += current_price * quantity
                    
                    if entry_price > 0:
                        pnl_pct = (current_price - entry_price) / entry_price * 100
                        pnl_amount = (current_price - entry_price) * quantity
                        
                        # 이모지: 수익(빨강/상승), 손실(파랑/하락)
                        emoji = "🔴" if pnl_pct >= 0 else "🔵"
                        
                        message += f"{emoji} *{symbol}*\n"
                        message += f"   수익률: `{pnl_pct:+.2f}%`\n"
                        message += f"   평가손익: `{pnl_amount:+,.{precision}f}{currency}`\n"
                        
                        total_pnl += pnl_amount
            else:
                message += "📌 보유 중인 종목이 없습니다.\n"
            
            # 총 자산 현황 (예수금 포함)
            total_equity = total_value + portfolio.current_capital
            
            # [New] 금일 실현 손익 조회
            daily_pnl = portfolio.get_daily_realized_pnl()
            
            message += "\n" + "-"*20 + "\n"
            message += f"💵 *보유 현금*: `{portfolio.current_capital:,.{precision}f}{currency}`\n"
            message += f"📅 *금일 실현손익*: `{daily_pnl:+,.{precision}f}{currency}`\n"
            message += f"💰 *총 평가손익*: `{total_pnl:+,.{precision}f}{currency}`\n"
            message += f"📦 *총 추정자산*: `{total_equity:,.{precision}f}{currency}`"
            
            self.send_telegram_message(message)
            
        except Exception as e:
            logger.error(f"포트폴리오 리포트 생성 중 오류: {e}")

    def generate_daily_report(self, symbol: str = "BTC/KRW"):
        """
        최근 48시간 데이터를 기반으로 전략별 성과를 비교하고 리포트를 생성합니다.
        """
        logger.info(f"📊 전략 성과 비교 리포트 생성 중... ({symbol})")
        
        try:
            # 최근 48시간 + Lookback(200) 데이터 확보를 위해 넉넉히 조회
            # 15분봉 기준 48시간 = 192개. + 200개 = 약 400개.
            # 1시간봉 기준 48시간 = 48개.
            # 넉넉하게 1000개 조회
            df = self.api.get_ohlcv(symbol, timeframe="15m", count=1000)
            
            if df.empty:
                logger.warning("데이터 부족으로 리포트 생성 실패")
                return

            # 최근 48시간 데이터 슬라이싱 (백테스트용)
            # WalkForwardAnalyzer는 전체 데이터를 받아 내부에서 처리하므로 df 그대로 전달
            # 단, test_period를 48시간에 해당하는 캔들 수로 설정
            
            results = {}
            
            for name, strategy in self.strategies.items():
                # 백테스터 설정 (수수료, 슬리피지 포함)
                # test_period: 15분봉 기준 48시간 = 192개
                analyzer = WalkForwardAnalyzer(
                    df, 
                    train_period=200, 
                    test_period=192, 
                    slippage=0.001, 
                    fee=0.0005,
                    stop_loss=0.04, # 기본값
                    take_profit=0.12
                )
                
                # _backtest_period를 직접 호출하거나 run을 사용.
                # 여기서는 run()을 사용하여 최근 구간만 테스트하도록 유도하거나,
                # analyzer 내부 로직을 활용. run()은 전체 기간을 step별로 돕니다.
                # 최근 48시간만 딱 찝어서 하려면 _backtest_period를 직접 쓰는게 나을 수 있으나,
                # analyzer 구조상 run()을 돌리고 마지막 결과를 쓰는게 편함.
                
                # run()은 전체 데이터를 훑으므로, 데이터프레임을 최근 데이터로 잘라서 줌
                # 학습(200) + 테스트(192) = 392개 필요
                recent_df = df.tail(400) 
                analyzer.data = recent_df
                
                # 전략 주입 (WalkForwardAnalyzer 수정 없이 전략 객체를 직접 사용하도록 약간의 트릭 필요할 수 있음)
                # 현재 WalkForwardAnalyzer는 내부에서 Strategy를 새로 생성함.
                # 이를 우회하기 위해 analyzer._backtest_period 메서드를 직접 호출하여 단발성 테스트 수행
                
                # 테스트 데이터 준비 (최근 192개)
                test_data = recent_df.iloc[-192:]
                
                # 백테스트 실행
                res = analyzer._backtest_period(strategy, test_data, lookback=200)
                
                # 수익률(%) 환산
                initial_capital = 100000000
                return_pct = (res['total_return'] / initial_capital) * 100
                win_rate = res['win_rate'] * 100
                
                results[name] = {
                    'return': return_pct,
                    'win_rate': win_rate
                }

            # 결과 비교 및 알림 메시지 작성
            best_strategy = max(results, key=lambda k: results[k]['return'])
            best_return = results[best_strategy]['return']
            
            msg = f"📢 [일일 전략 리포트]\n"
            msg += f"대상: {symbol} (최근 48h)\n"
            for name, data in results.items():
                recommend = "✅" if name == best_strategy else ""
                msg += f"- {name}: 수익 {data['return']:+.2f}% / 승률 {data['win_rate']:.1f}% {recommend}\n"
            
            msg += f"\n🏆 추천: **{best_strategy}** (기대수익 {best_return:+.2f}%)"
            
            # 텔레그램 전송 (로그로 대체)
            logger.info("="*40)
            logger.info(msg)
            logger.info("="*40)
            self.send_telegram_message(msg)
            
        except Exception as e:
            logger.error(f"리포트 생성 중 오류: {e}")

    def send_trade_alert(self, symbol: str, side: str, price: float, quantity: float, 
                         pnl: float = 0.0, pnl_pct: float = 0.0, reason: str = "", leverage: int = None):
        """매매 알림 전송 (텔레그램)"""
        try:
            if "BUY" in side.upper() or "PYRAMIDING" in side.upper():
                title = "🔥 [불타기]" if "PYRAMIDING" in side.upper() else "🚀 [매수]"
                msg = f"{title} {symbol}\n"
                msg += f"가격: {price:,.0f}원\n"
                msg += f"수량: {quantity:.8f}\n"
                if reason:
                    msg += f"사유: {reason}"
            else:
                tag = "[매도]"
                reason_lower = str(reason).lower()
                if "emergency" in reason_lower:
                    tag = "🚨 [긴급매도]"
                elif "stop_loss" in reason_lower or "손절" in reason_lower:
                    tag = "💧 [손절실행]"
                elif "take_profit" in reason_lower or "익절" in reason_lower:
                    tag = "💰 [수익확정]"
                elif "trailing_stop" in reason_lower:
                    tag = "🛡️ [수익확정(TS)]"
                elif "break-even" in reason_lower or "본절" in reason_lower:
                    tag = "🛡️ [본절보존]"
                
                msg = f"{tag} {symbol}\n"
                msg += f"매도가: {price:,.0f}원\n"
                msg += f"수량: {quantity:.8f}\n"
                msg += f"손익: {pnl:+,.0f}원 ({pnl_pct:+.2f}%)\n"
                if leverage:
                    msg += f"레버리지: {leverage}x\n"
                msg += f"사유: {reason}"
            
            self.send_telegram_message(msg)
        except Exception as e:
            logger.error(f"알림 전송 오류: {e}")