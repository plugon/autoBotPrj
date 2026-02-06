import requests
from datetime import datetime
from email.utils import parsedate_to_datetime
import time

def check_time():
    print("="*60)
    print("🕒 시스템 시간 정밀 검증")
    print("="*60)
    
    # 1. 로컬 시간
    local_now = datetime.now()
    local_utc = datetime.utcnow()
    print(f"💻 내 컴퓨터 시간 (Local): {local_now}")
    print(f"💻 내 컴퓨터 시간 (UTC)  : {local_utc}")
    
    # 2. 서버 시간 (Google)
    try:
        print("\n🌍 구글 서버 시간 조회 중...")
        response = requests.head("https://www.google.com", timeout=5)
        if 'Date' in response.headers:
            server_time = parsedate_to_datetime(response.headers['Date']).replace(tzinfo=None)
            print(f"🌍 구글 서버 시간 (UTC)  : {server_time}")
            
            diff = (server_time - local_utc).total_seconds()
            print(f"\n⏱️ 시간 차이: {diff:.2f}초")
            
            if abs(diff) > 60:
                print("\n❌ [경고] 시간이 맞지 않습니다! 동기화가 필요합니다.")
                print("   -> 윈도우 설정 > 시간 및 언어 > '지금 동기화' 버튼을 눌러주세요.")
            else:
                print("\n✅ [정상] 시스템 시간이 정확합니다.")
        else:
            print("⚠️ 서버 응답에 Date 헤더가 없습니다.")
            
    except Exception as e:
        print(f"❌ 서버 연결 실패: {e}")

    print("="*60)
    input("엔터 키를 누르면 종료합니다...")

if __name__ == "__main__":
    check_time()
