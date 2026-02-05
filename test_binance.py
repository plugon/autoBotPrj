import ccxt
from dotenv import load_dotenv
import os

load_dotenv()
load_dotenv(".env_secret") # [New] 시크릿 파일 로드 추가

def test_connection():
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    # 1. 객체 생성
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'} # 우선 현물로 테스트
    })

    try:
        # 2. 잔액 조회 (API 키 권한 확인의 척도)
        balance = exchange.fetch_balance()
        print("✅ 연결 성공!")
        print(f"💰 가용 USDT: {balance.get('USDT', {}).get('free', 0)}")
        
        # 3. 선물 권한 확인 (선택 사항)
        exchange.options['defaultType'] = 'future'
        f_balance = exchange.fetch_balance()
        print("✅ 선물 API 접근 권한 확인 완료!")
        
    except Exception as e:
        print(f"❌ 연결 실패: {e}")

if __name__ == "__main__":
    test_connection()