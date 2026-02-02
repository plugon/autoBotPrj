import os
import requests
from dotenv import load_dotenv

def main():
    """
    텔레그램 봇 알림 테스트 스크립트
    .env 파일에 설정된 TELEGRAM_BOT_TOKEN과 TELEGRAM_CHAT_ID를 사용하여 테스트 메시지를 전송합니다.
    """
    # .env 파일 로드
    load_dotenv()
    
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    print("=" * 50)
    print("📡 텔레그램 알림 테스트")
    print("=" * 50)
    
    if not token:
        print("❌ 오류: .env 파일에 TELEGRAM_BOT_TOKEN이 설정되지 않았습니다.")
        return
    if not chat_id:
        print("❌ 오류: .env 파일에 TELEGRAM_CHAT_ID가 설정되지 않았습니다.")
        return
        
    print(f"🔹 Bot Token: {token[:6]}******")
    print(f"🔹 Chat ID  : {chat_id}")
    
    message = "🔔 [테스트] 텔레그램 알림이 정상적으로 작동합니다.\n봇 연결 상태: ✅ 양호"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {
        "chat_id": chat_id, 
        "text": message
    }
    
    try:
        print("\n🚀 메시지 전송 시도 중...")
        response = requests.post(url, data=data, timeout=10)
        
        if response.status_code == 200:
            print("✅ 전송 성공! 텔레그램 메시지를 확인해주세요.")
        else:
            print(f"❌ 전송 실패 (Status: {response.status_code})")
            print(f"   응답: {response.text}")
            
    except Exception as e:
        print(f"❌ 전송 중 오류 발생: {e}")

if __name__ == "__main__":
    main()