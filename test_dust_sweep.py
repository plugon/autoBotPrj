"""
소액 코인(먼지) 정리 테스트 스크립트

기능:
1. 보유 중인 코인 중 평가금액 5,000원 미만인 '먼지 코인'을 조회합니다.
2. 선택한 코인에 대해 5,000원어치를 시장가로 추가 매수합니다.
3. 매수 체결 후 합산된 수량을 전량 시장가로 매도합니다.

주의:
* 계좌에 최소 5,000원 이상의 예수금(KRW)이 있어야 작동합니다.
* 매수/매도 과정에서 수수료(약 0.1%) 및 슬리피지가 발생할 수 있습니다.
"""

import os
import time
import logging
from dotenv import load_dotenv
from api.crypto_api import UpbitAPI

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # 깔끔한 출력을 위해 포맷 단순화
)
logger = logging.getLogger("DustSweeper")

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
    
    print("\n🔍 보유 코인 조회 중...")
    positions = api.get_positions()
    
    if not positions:
        print("보유 중인 코인이 없습니다.")
        return

    print(f"\n{'='*60}")
    print(f"{'No.':<5} {'종목':<10} {'수량':<15} {'현재가(추정)':<15} {'평가금액':<15}")
    print(f"{'='*60}")

    dust_coins = []
    
    for i, p in enumerate(positions):
        symbol = p['symbol']
        qty = p['quantity']
        current_price = api.get_price(symbol)
        value = qty * current_price
        
        # 5000원 미만인 경우 표시
        is_dust = value < 5000
        mark = "🧹" if is_dust else "  "
        
        print(f"{mark} {i+1:<3} {symbol:<10} {qty:<15.8f} {current_price:<15,.0f} {value:<15,.0f}")
        
        if is_dust:
            dust_coins.append((symbol, qty, value))

    print(f"{'='*60}\n")

    if not dust_coins:
        print("✅ 5,000원 미만의 소액 코인(먼지)이 없습니다.")
        return

    print(f"🧹 정리 가능한 먼지 코인: {len(dust_coins)}개")
    choice = input("어떤 코인을 정리하시겠습니까? (종목코드 예: XRP/KRW, 전체는 'all', 종료는 'q'): ").strip().upper()
    
    if choice == 'Q':
        return
    
    targets = []
    if choice == 'ALL':
        targets = dust_coins
    else:
        # 입력한 종목 찾기
        target = next((item for item in dust_coins if item[0] == choice), None)
        if target:
            targets = [target]
        else:
            print("❌ 목록에 없는 종목이거나 5,000원 이상인 코인입니다.")
            return

    # 정리 로직 실행
    for symbol, quantity, value in targets:
        print(f"\n🚀 [{symbol}] 먼지 털기 시작 (현재 가치: {value:,.0f}원)")
        
        # 1. 잔액 확인
        balance = api.get_balance()
        krw_free = float(balance.get("free", {}).get("KRW", 0))
        
        if krw_free < 5000:
            logger.error(f"❌ 잔액 부족 ({krw_free:,.0f}원). 최소 5,000원이 필요합니다.")
            continue

        # 2. 추가 매수 (5000원)
        print(f"   👉 1단계: 5,000원 시장가 매수 시도...")
        if api.buy(symbol, 5000):
            print("   ✅ 매수 주문 완료. 체결 대기 (2초)...")
            time.sleep(2)
            
            # 3. 수량 재조회 (매수된 수량 포함)
            positions = api.get_positions()
            new_quantity = 0
            for p in positions:
                if p['symbol'] == symbol:
                    new_quantity = p['quantity']
                    break
            
            if new_quantity > 0:
                # 4. 전량 매도
                print(f"   👉 2단계: 전량 매도 시도 ({new_quantity:.8f} {symbol})...")
                if api.sell(symbol, new_quantity):
                    print(f"   ✅ {symbol} 정리 완료!")
                else:
                    print(f"   ❌ 매도 실패")
            else:
                print("   ❌ 매수 후 수량 확인 실패")
        else:
            print("   ❌ 매수 실패")

if __name__ == "__main__":
    main()