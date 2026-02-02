@echo off
chcp 65001
echo ========================================================
echo  Dashboard EXE 변환 스크립트
echo ========================================================

echo 1. PyInstaller 설치 확인...
pip install pyinstaller

echo.
echo 2. EXE 생성 중... (Streamlit 라이브러리가 커서 시간이 좀 걸립니다)
:: --collect-all streamlit: 스트림릿 관련 모든 파일 포함
:: --add-data: 대시보드 소스코드 포함

pyinstaller --noconfirm --onefile --console --clean --name "TradingDashboard" ^
    --collect-all streamlit ^
    --collect-all altair ^
    --collect-all pandas ^
    --add-data "dashboard.py;." ^
    --hidden-import=streamlit ^
    run_dashboard.py

echo.
echo ========================================================
echo  빌드 완료!
echo ========================================================
echo.
echo [생성된 파일 위치]
echo  - 실행 파일: dist\TradingDashboard.exe
echo.
pause