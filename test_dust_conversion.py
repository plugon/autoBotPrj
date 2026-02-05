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
    setup_logger("test_dust", logging.INFO)
    logger = logging.getLogger("test_dust")
    
    # 환경변수 로드
    load_dotenv()
    load_dotenv(".env_secret")

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        logger.error("❌ .env 파일에 BINANCE_API_KEY 또는 BINANCE_API_SECRET이 없습니다.")
        return

    logger.info("🔌 바이낸스 현물 API 연결 중...")
    try:
        # Spot 계정으로 연결
        api = BinanceAPI(api_key, api_secret, account_type='spot')
        api.connect()
    except Exception as e:
        logger.error(f"❌ API 연결 실패: {e}")
        return

    logger.info("🧹 소액 잔고(Dust) BNB 변환 테스트 시작...")
    
    # 테스트를 위해 쿨타임 강제 초기화
    api.last_dust_conversion = 0 
    
    # 변환 시도 (인자 없이 호출하면 전체 조회 후 변환)
    # 주의: 바이낸스 API는 Dust 변환에 쿨타임(보통 6시간 또는 1시간) 제한이 있습니다.
    result = api.convert_dust_to_bnb()
    
    if result:
        if 'totalTransfered' in result:
            logger.info(f"✅ 변환 성공! 총 {result['totalTransfered']} BNB로 변환되었습니다.")
            logger.info(f"상세 결과: {result}")
        else:
            logger.info("ℹ️ 변환된 내역이 없거나 결과 형식이 다릅니다.")
    else:
        logger.info("⚠️ 변환 실패 또는 변환할 자산이 없습니다. (위 로그 확인)")

if __name__ == "__main__":
    main()
