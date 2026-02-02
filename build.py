import os
import shutil
import sys
import subprocess

def check_and_setup_venv():
    """가상환경 확인 및 자동 설정"""
    # 현재 실행 중인 Python이 가상환경인지 확인
    if sys.prefix != sys.base_prefix:
        return  # 이미 가상환경임

    print("⚠️ 가상환경(venv)이 활성화되지 않았습니다.")
    
    work_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(work_dir, "venv")
    
    # OS별 실행 파일 경로
    if sys.platform == "win32":
        python_executable = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        python_executable = os.path.join(venv_dir, "bin", "python")

    # 1. 가상환경 생성
    if not os.path.exists(python_executable):
        print(f"🔨 가상환경을 생성합니다: {venv_dir}")
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])
        
        # 2. pip 업그레이드 및 필수 패키지 설치
        print("📦 패키지 설치 중...")
        subprocess.check_call([python_executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        
        # requirements.txt 설치
        req_path = os.path.join(work_dir, "requirements.txt")
        if os.path.exists(req_path):
            print(f"📄 requirements.txt 설치 중...")
            subprocess.check_call([python_executable, "-m", "pip", "install", "-r", req_path])
        
        # PyInstaller는 빌드 필수이므로 명시적 설치
        subprocess.check_call([python_executable, "-m", "pip", "install", "pyinstaller"])

    # 3. 가상환경으로 재실행
    print(f"🔄 가상환경({venv_dir})으로 전환하여 빌드를 시작합니다...\n")
    subprocess.check_call([python_executable] + sys.argv)
    sys.exit()

def build_exe():
    # 현재 스크립트 위치를 작업 디렉토리로 설정
    work_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(work_dir)

    print(f"작업 디렉토리: {work_dir}")

    # PyInstaller 임포트 (가상환경 내에서 실행됨을 보장)
    try:
        import PyInstaller.__main__
        from PyInstaller.utils.hooks import collect_data_files, copy_metadata, collect_submodules
    except ImportError:
        print("PyInstaller가 설치되어 있지 않습니다. 설치를 시작합니다...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        import PyInstaller.__main__
        from PyInstaller.utils.hooks import collect_data_files, copy_metadata, collect_submodules

    # 1. 기존 빌드 폴더 정리
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    if os.path.exists("build"):
        shutil.rmtree("build")
    
    os.makedirs("dist", exist_ok=True)

    # 2. Trading Bot 빌드 (main.py)
    print("\n" + "=" * 50)
    print("🤖 Trading Bot (main.py) 빌드 중...")
    print("=" * 50)
    
    # psutil 설치 확인 (시스템 리소스 모니터링용)
    try:
        import psutil
    except ImportError:
        print("psutil이 설치되어 있지 않습니다. 설치를 시작합니다...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])

    main_args = [
        'main.py',
        '--name=TradingBot',    # 실행 파일 이름
        '--onefile',            # 단일 파일로 생성
        '--clean',              # 캐시 정리
        '--log-level=INFO',
        # 머신러닝 라이브러리 관련 히든 임포트 (에러 방지)
        '--hidden-import=sklearn.utils._typedefs',
        '--hidden-import=sklearn.neighbors._partition_nodes',
        '--hidden-import=sklearn.tree',
        '--hidden-import=sklearn.ensemble',
    ]
    
    # 아이콘 파일이 있으면 적용 (icon.ico)
    if os.path.exists("icon.ico"):
        print("🎨 TradingBot 아이콘 적용: icon.ico")
        main_args.append('--icon=icon.ico')
    
    PyInstaller.__main__.run(main_args)

    # 3. Dashboard 빌드 (run_dashboard.py)
    print("\n" + "=" * 50)
    print("📈 Dashboard (run_dashboard.py) 빌드 중...")
    print("=" * 50)

    # Streamlit 설치 확인
    try:
        import streamlit
    except ImportError:
        print("Streamlit이 설치되어 있지 않습니다. 설치를 시작합니다...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])

    # Streamlit 관련 데이터 및 메타데이터 수집
    datas = []
    datas.append(('dashboard.py', '.'))  # dashboard.py를 실행 파일 내부에 포함
    
    try:
        datas += collect_data_files('streamlit')
    except Exception as e:
        print(f"⚠️ Streamlit 데이터 파일 수집 실패: {e}")

    try:
        datas += copy_metadata('streamlit')
    except Exception as e:
        print(f"⚠️ Streamlit 메타데이터 수집 실패 (무시됨): {e}")
    
    # Streamlit 관련 히든 임포트 수집
    hidden_imports = collect_submodules('streamlit')
    hidden_imports.append('streamlit.web.cli')
    hidden_imports.append('pyupbit')
    hidden_imports.append('pandas')
    hidden_imports.append('config')
    hidden_imports.append('config.settings')
    
    dashboard_args = [
        'run_dashboard.py',
        '--name=Dashboard',
        '--onefile',
        '--clean',
        '--noconsole',
    ]
    
    # 아이콘 파일이 있으면 적용 (icon.ico)
    if os.path.exists("icon.ico"):
        print("🎨 Dashboard 아이콘 적용: icon.ico")
        dashboard_args.append('--icon=icon.ico')
    
    for hidden in hidden_imports:
        dashboard_args.append(f'--hidden-import={hidden}')
        
    for src, dest in datas:
        # 윈도우 경로 구분자(;) 처리
        dashboard_args.append(f'--add-data={src}{os.pathsep}{dest}')
        
    PyInstaller.__main__.run(dashboard_args)

    # 4. Analyze Performance 빌드 (analyze_performance.py)
    print("\n" + "=" * 50)
    print("📊 Analyze Performance (analyze_performance.py) 빌드 중...")
    print("=" * 50)
    
    # Matplotlib 설치 확인
    try:
        import matplotlib
    except ImportError:
        print("Matplotlib이 설치되어 있지 않습니다. 설치를 시작합니다...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    
    analyze_args = [
        'analyze_performance.py',
        '--name=AnalyzePerformance',
        '--onefile',
        '--clean',
    ]
    
    if os.path.exists("icon.ico"):
        print("🎨 AnalyzePerformance 아이콘 적용: icon.ico")
        analyze_args.append('--icon=icon.ico')
    
    PyInstaller.__main__.run(analyze_args)

    # 5. 배포 파일 및 폴더 정리
    print("\n" + "=" * 50)
    print("� 배포 패키지 구성 중...")
    print("=" * 50)

    # 필수 데이터 폴더 생성 (EXE 실행 시 필요)
    folders_to_create = ['data', 'logs', 'models', 'config']
    for folder in folders_to_create:
        path = os.path.join("dist", folder)
        os.makedirs(path, exist_ok=True)
        print(f"폴더 생성: {path}")

    # .env 파일 처리 (API 키)
    if os.path.exists(".env"):
        shutil.copy(".env", "dist/.env")
        print(".env 파일 복사 완료")
    else:
        # .env 템플릿 생성
        env_path = os.path.join("dist", ".env")
        with open(env_path, "w", encoding="utf-8") as f:
            f.write("# API 키 설정\n")
            f.write("UPBIT_API_KEY=your_key_here\n")
            f.write("UPBIT_API_SECRET=your_secret_here\n")
            f.write("\n# --- 암호화폐 거래 전략 설정 (재빌드 없이 수정 가능) ---\n")
            f.write("# 1. 전략 프리셋 선택 (아래 중 하나 선택)\n")
            f.write("# 사용 가능한 전략: scalping(초단타), short_term(단기), mid_term(중기), long_term(장기)\n")
            f.write("CRYPTO_STRATEGY_PRESET=scalping\n\n")
            
            f.write("# 2. 기본 설정\n")
            f.write("CRYPTO_INITIAL_CAPITAL=300000\n")
            f.write("CRYPTO_MAX_POSITIONS=3\n")
            f.write("CRYPTO_MIN_ORDER_AMOUNT=5000\n")
            f.write("MAX_SYMBOLS=10\n\n")

            f.write("# 3. 상세 설정 (프리셋 값을 덮어쓰고 싶을 때만 주석 해제 후 사용)\n")
            f.write("#CRYPTO_MAX_POSITION_SIZE=0.2\n")
            f.write("#CRYPTO_STOP_LOSS=0.012\n")
            f.write("#CRYPTO_TAKE_PROFIT=0.02\n")
            f.write("#CRYPTO_TRAILING_STOP=0.008\n")
            f.write("#CRYPTO_TIMEFRAME=1m\n")
            
            f.write("\n# 4. 머신러닝 설정\n")
            f.write("# 사용 가능한 모델: lstm, random_forest, xgboost\n")
            f.write("MODEL_TYPE=lstm\n")
            
            f.write("\n# 5. API 활성화 설정 (True/False)\n")
            f.write("ENABLE_UPBIT=True\n")
            f.write("ENABLE_BINANCE=True\n")
            f.write("ENABLE_SHINHAN=False\n")
            f.write("ENABLE_KIWOOM=False\n")
            f.write("ENABLE_DAISHIN=False\n")
        print(".env 템플릿 파일 생성 완료")

    # README 복사
    if os.path.exists("README.md"):
        shutil.copy("README.md", "dist/README.md")

    print("\n" + "=" * 50)
    print("✅ 빌드 완료!")
    print(f"결과물 위치: {os.path.join(work_dir, 'dist')}")
    print("-" * 50)
    print("1. dist/TradingBot.exe : 자동매매 봇 실행 파일")
    print("2. dist/Dashboard.exe  : 대시보드 실행 파일")
    print("3. dist/AnalyzePerformance.exe : 성과 분석 실행 파일")
    print("4. dist/.env           : API 키 설정 파일 (다른 PC에서 실행 시 수정 필요)")
    print("=" * 50)

if __name__ == "__main__":
    # --reset-venv 옵션이 있으면 가상환경 폴더 삭제 (초기화)
    if "--reset-venv" in sys.argv:
        # 현재 가상환경 내부가 아닐 때만 삭제 시도
        if sys.prefix == sys.base_prefix:
            sys.argv.remove("--reset-venv")
            work_dir = os.path.dirname(os.path.abspath(__file__))
            venv_dir = os.path.join(work_dir, "venv")
            
            if os.path.exists(venv_dir):
                print(f"🗑️ 기존 가상환경 폴더를 삭제합니다: {venv_dir}")
                try:
                    shutil.rmtree(venv_dir)
                    print("✅ 삭제 완료. 잠시 후 가상환경을 다시 생성합니다.")
                except Exception as e:
                    print(f"❌ 삭제 실패: {e}\n   (폴더가 사용 중이거나 권한이 없습니다. 수동으로 삭제해주세요.)")

    check_and_setup_venv()
    build_exe()
