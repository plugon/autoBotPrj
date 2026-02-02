import sys
import os
from streamlit.web import cli as stcli

def resolve_path(path):
    """EXE 내부의 리소스 경로를 찾습니다."""
    if getattr(sys, "frozen", False):
        # PyInstaller로 패키징된 경우 임시 폴더 경로 사용
        basedir = sys._MEIPASS
    else:
        basedir = os.path.dirname(__file__)
    return os.path.join(basedir, path)

if __name__ == "__main__":
    # dashboard.py의 절대 경로 계산
    dashboard_script = resolve_path("dashboard.py")
    
    # streamlit run 명령 실행 시뮬레이션
    sys.argv = [
        "streamlit",
        "run",
        dashboard_script,
        "--global.developmentMode=false",
    ]
    
    sys.exit(stcli.main())