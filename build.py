# -*- coding: utf-8 -*-
import os
import shutil
import sys
import subprocess

def check_and_setup_venv():
    if sys.prefix != sys.base_prefix: return
    work_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(work_dir, "venv")
    python_exe = os.path.join(venv_dir, "Scripts", "python.exe") if sys.platform == "win32" else os.path.join(venv_dir, "bin", "python")

    if not os.path.exists(python_exe):
        print(f"🔨 가상환경 생성 중...")
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])

    print(f"📦 패키지 설치 및 업데이트 확인 중...")
    required = ["pip", "setuptools", "wheel", "pyinstaller", "ccxt", "pyupbit", "python-dotenv", 
                "pandas", "numpy", "tensorflow", "tf2onnx", "onnxruntime", "scikit-learn", 
                "psutil", "matplotlib", "streamlit", "websocket-client", "apscheduler", "ta", "certifi"]
    subprocess.check_call([python_exe, "-m", "pip", "install", "--upgrade"] + required)

    # [Fix] 스크립트 절대 경로 사용 및 인터럽트 예외 처리 (Traceback 방지)
    script_path = os.path.abspath(sys.argv[0])
    args = sys.argv[1:]
    
    try:
        subprocess.check_call([python_exe, script_path] + args, cwd=work_dir)
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
        
    sys.exit()

def build_exe():
    work_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(work_dir)
    import PyInstaller.__main__
    from PyInstaller.utils.hooks import collect_submodules, collect_data_files, copy_metadata

    for folder in ["dist", "build"]:
        if os.path.exists(folder): shutil.rmtree(folder)
    os.makedirs("dist", exist_ok=True)

    icon_args = [f'--icon=icon.ico'] if os.path.exists("icon.ico") else []

    # 1. Trading Bot 빌드
    print("\n🤖 1/3: Trading Bot 빌드 시작")
    tf_datas = collect_data_files('tensorflow')
    main_args = ['main.py', '--name=TradingBot', '--onefile', '--clean'] + icon_args
    for hi in ['tensorflow', 'onnxruntime', 'tf2onnx', 'sklearn.utils._typedefs', 'websocket', 'apscheduler', 'ta', 'certifi']:
        main_args.append(f'--hidden-import={hi}')
    for src, dest in tf_datas:
        main_args.append(f'--add-data={src}{os.pathsep}{dest}')
    PyInstaller.__main__.run(main_args)

    # 2. Dashboard 빌드 (문법 오류 해결)
    print("\n📈 2/3: Dashboard 빌드 시작")
    dash_args = ['run_dashboard.py', '--name=Dashboard', '--onefile', '--noconsole', '--clean'] + icon_args
    
    # 스트림릿 데이터/메타데이터/히든임포트 수집
    st_datas = collect_data_files('streamlit')
    st_meta = copy_metadata('streamlit')
    st_hidden = collect_submodules('streamlit')

    if os.path.exists('dashboard.py'):
        dash_args.append(f'--add-data=dashboard.py{os.pathsep}.')
    
    # 수집된 데이터 추가 (오류 수정 지점)
    for src, dest in st_datas: dash_args.append(f'--add-data={src}{os.pathsep}{dest}')
    for m_src, m_dest in st_meta: dash_args.append(f'--add-data={m_src}{os.pathsep}{m_dest}')
    for h in st_hidden + ['streamlit.web.cli', 'pyupbit', 'config', 'websocket']:
        dash_args.append(f'--hidden-import={h}')
    
    PyInstaller.__main__.run(dash_args)

    # 3. Analyze Performance 빌드
    print("\n📊 3/3: Analyze Performance 빌드 시작")
    PyInstaller.__main__.run(['analyze_performance.py', '--name=AnalyzePerformance', '--onefile', '--clean'] + icon_args)

    # 4. Backtester 빌드 (추가)
    print("\n🧪 4/4: Backtester 빌드 시작")
    backtest_args = ['run_backtest_all.py', '--name=Backtester', '--onefile', '--clean'] + icon_args
    for hi in ['tensorflow', 'onnxruntime', 'tf2onnx', 'sklearn.utils._typedefs', 'websocket']:
        backtest_args.append(f'--hidden-import={hi}')
    # 텐서플로우 데이터 추가 (필요 시)
    for src, dest in tf_datas: backtest_args.append(f'--add-data={src}{os.pathsep}{dest}')
    PyInstaller.__main__.run(backtest_args)

    # 마무리 작업
    for folder in ['data', 'logs', 'models', 'config']: os.makedirs(os.path.join("dist", folder), exist_ok=True)
    if os.path.exists(".env"): shutil.copy(".env", "dist/.env")
    if os.path.exists(".env_secret"): shutil.copy(".env_secret", "dist/.env_secret") # [New] 시크릿 파일 복사
    print("\n✅ 빌드 완료! dist 폴더를 확인하세요.")

if __name__ == "__main__":
    if "--reset-venv" in sys.argv:
        if os.path.exists("venv"): shutil.rmtree("venv")
        sys.argv.remove("--reset-venv")
    check_and_setup_venv()
    build_exe()