@echo off
chcp 65001
echo ========================================================
echo  자동매매 봇 실행 스크립트
echo ========================================================

cd /d "%~dp0"

:: 가상환경 확인 및 활성화
if not exist venv (
    echo 🔨 가상환경[venv]이 없습니다. 새로 생성하고 패키지를 설치합니다...
    python -m venv venv
    call venv\Scripts\activate
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
)

:: 봇 실행
echo.
echo 🚀 봇을 시작합니다... (종료하려면 Ctrl+C)
echo.
python main.py
pause