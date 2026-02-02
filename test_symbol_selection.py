"""
종목 선정 로직 테스트 스크립트

기능:
1. 업비트 전 종목 중 24시간 거래대금 100억 원 이상인 종목을 필터링합니다.
2. 필터링된 종목들의 최근 1시간 변동률(ROC)을 계산합니다.
3. 변동률이 높은 순서대로 정렬하여 상위 종목을 출력합니다.
"""

import os
import time
import logging
from dotenv import load_dotenv
from api.crypto_api import UpbitAPI
from config.settings import VOLUME_CONFIG

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger("SymbolSelector")

def main():
    # 환경변수 로드
    load_dotenv()
    
    api_key = os.getenv("UPBIT_API_KEY")
    api_secret = os.getenv("UPBIT_API_SECRET")
    
    if not api_key or not api_secret:
        logger.error("❌ .env 파일에 UPBIT API 키가 설정되지 않았습니다.")
        return

    # API 연결
    api = UpbitAPI(api_key, api_secret)
    api.connect()
    
    print("\n🔍 전체 마켓 정보 조회 중...")
    try:
        markets = api.exchange.fetch_tickers()
    except Exception as e:
        logger.error(f"마켓 조회 실패: {e}")
        return

    # 1. 거래대금 필터링 (100억 이상)
    min_volume = 10_000_000_000  # 100억 원
    candidates = []
    
    print(f"📊 필터 기준: 24시간 거래대금 {min_volume/100_000_000:,.0f}억 원 이상")
    
    for symbol, ticker in markets.items():
        if "/KRW" in symbol and ticker.get('quoteVolume') is not None:
            volume_krw = ticker['quoteVolume']
            
            if volume_krw >= min_volume:
                if symbol not in VOLUME_CONFIG["exclude_symbols"]:
                    candidates.append((symbol, volume_krw))
    
    print(f"✅ 1차 필터링 통과: {len(candidates)}개 종목")
    
    # 2. 1시간 변동률(ROC) 계산 및 정렬
    print("\n🚀 1시간 변동률(ROC) 분석 중... (API 호출 제한으로 시간이 걸릴 수 있습니다)")
    scored_candidates = []
    
    for i, (symbol, volume) in enumerate(candidates):
        # 진행 상황 표시
        print(f"\r   [{i+1}/{len(candidates)}] {symbol} 분석 중...", end="")
        
        try:
            # 1시간봉 2개 조회 (직전 캔들과 현재 캔들)
            df = api.get_ohlcv(symbol, timeframe="1h", count=2)
            
            if not df.empty:
                # 현재 캔들의 시가 vs 현재가 비교 (실시간 모멘텀)
                # df.iloc[-1]은 현재 진행 중인 캔들
                current_open = df.iloc[-1]['open']
                current_close = df.iloc[-1]['close']
                
                if current_open > 0:
                    roc = (current_close - current_open) / current_open * 100 # 퍼센트 단위
                    scored_candidates.append({
                        'symbol': symbol,
                        'volume': volume,
                        'roc': roc,
                        'price': current_close
                    })
            
            # Rate Limit 준수 (초당 요청 제한 고려)
            time.sleep(0.1)
            
        except Exception as e:
            logger.warning(f"\n⚠️ {symbol} 데이터 조회 실패: {e}")

    print("\n\n✅ 분석 완료! 변동률 상위 종목을 정렬합니다.")
    
    # 변동률 내림차순 정렬
    scored_candidates.sort(key=lambda x: x['roc'], reverse=True)
    
    # 결과 출력
    print(f"\n{'='*75}")
    print(f"{'순위':<5} {'종목':<10} {'현재가':<15} {'1시간 변동률':<15} {'거래대금(24h)':<15}")
    print(f"{'='*75}")
    
    top_n = min(len(scored_candidates), 20) # 상위 20개만 출력
    
    for i in range(top_n):
        item = scored_candidates[i]
        vol_str = f"{item['volume']/100_000_000:,.0f}억"
        roc_str = f"{item['roc']:+.2f}%"
        
        # 상위 10개(선정 대상)는 강조 표시
        mark = "👉" if i < 10 else "  "
        
        # 색상 효과 (터미널 지원 시)
        color_reset = "\033[0m"
        color_red = "\033[91m" if item['roc'] > 0 else "\033[94m"
        
        print(f"{mark} {i+1:<3} {item['symbol']:<10} {item['price']:<15,.0f} {color_red}{roc_str:<15}{color_reset} {vol_str:<15}")
        
    print(f"{'='*75}")
    print(f"👉 상위 10개 종목이 봇의 감시 대상(crypto_symbols)으로 선정됩니다.")

if __name__ == "__main__":
    main()
