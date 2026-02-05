import os
import sys
import logging
import time
from dotenv import load_dotenv

# 프로젝트 루트 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.crypto_api import BinanceAPI
from utils.logger import setup_logger

def main():
    # 로거 설정 (콘솔 출력)
    setup_logger("test_fallback", logging.INFO)
    logger = logging.getLogger("test_fallback")
    
    # 환경변수 로드
    load_dotenv()
    load_dotenv(".env_secret")

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        logger.error("❌ .env 파일에 BINANCE_API_KEY 또는 BINANCE_API_SECRET이 없습니다.")
        return

    logger.info("🔌 바이낸스 API 연결 중...")
    try:
        # Spot 계정으로 연결 (선물도 동일 로직)
        api = BinanceAPI(api_key, api_secret, account_type='spot')
        api.connect()
    except Exception as e:
        logger.error(f"❌ API 연결 실패: {e}")
        return

    symbol = "BTC/USDT"
    
    # ---------------------------------------------------------
    # [TEST 1] 웹소켓 미연결 상태에서 가격 조회 (REST API Fallback)
    # ---------------------------------------------------------
    logger.info(f"\n🧪 [TEST 1] 웹소켓 미연결 상태에서 가격 조회 (REST API Fallback)")
    logger.info(f"   - 현재 웹소켓 상태(is_ws_ready): {api.is_ws_ready}")
    
    start_time = time.time()
    price = api.get_price(symbol)
    duration = time.time() - start_time
    
    if price > 0:
        logger.info(f"   ✅ 조회 성공: {symbol} = {price:,.2f} (소요시간: {duration:.4f}초)")
        logger.info("   -> 웹소켓 없이 REST API를 통해 데이터를 가져왔습니다.")
    else:
        logger.error(f"   ❌ 조회 실패: 가격이 0입니다.")

    # ---------------------------------------------------------
    # [TEST 2] 웹소켓 연결 후 끊김 시뮬레이션
    # ---------------------------------------------------------
    logger.info(f"\n🧪 [TEST 2] 웹소켓 연결 후 끊김 시뮬레이션")
    
    # 웹소켓 연결
    logger.info("   📡 웹소켓 구독 시작...")
    api.subscribe_websocket([symbol])
    
    # 데이터 수신 대기 (최대 10초)
    wait_count = 0
    while not api.is_ws_ready and wait_count < 10:
        time.sleep(1)
        wait_count += 1
        print(".", end="", flush=True)
    print()
    
    if api.is_ws_ready:
        logger.info("   ✅ 웹소켓 연결 완료 (Ready)")
        
        # 웹소켓으로 가격 조회 (캐시된 값)
        start_time = time.time()
        ws_price = api.get_price(symbol)
        duration = time.time() - start_time
        logger.info(f"   - 웹소켓 가격: {ws_price:,.2f} (소요시간: {duration:.6f}초 - 매우 빠름)")
        
        # 강제 끊김 시뮬레이션
        logger.info("   ✂️ 웹소켓 연결 상태 강제 해제 (Simulating Disconnect)...")
        api.is_ws_ready = False # 강제로 '준비 안 됨' 상태로 변경하여 Fallback 유도
        
        # 다시 조회 (Fallback 동작 확인)
        logger.info("   🔄 끊김 상태에서 가격 재조회 (Fallback 동작 확인)...")
        start_time = time.time()
        fallback_price = api.get_price(symbol)
        duration = time.time() - start_time
        
        if fallback_price > 0:
            logger.info(f"   ✅ Fallback 조회 성공: {symbol} = {fallback_price:,.2f} (소요시간: {duration:.4f}초)")
            logger.info("   -> 웹소켓 상태가 False일 때 REST API로 자동 전환되었습니다.")
        else:
            logger.error("   ❌ Fallback 조회 실패")
            
    else:
        logger.warning("   ⚠️ 웹소켓 연결 시간 초과로 TEST 2를 건너뜁니다.")

    # 종료
    api.disconnect()
    logger.info("\n🏁 테스트 완료")

if __name__ == "__main__":
    main()
