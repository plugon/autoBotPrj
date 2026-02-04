# -*- coding: utf-8 -*-
import os
import shutil
import sys
import subprocess

def check_and_setup_venv():
    if sys.prefix != sys.base_prefix: return
    work_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(work_dir, "venv")
    # Mac/Linux uses bin/python
    python_exe = os.path.join(venv_dir, "bin", "python")

    if not os.path.exists(python_exe):
        print(f"🔨 가상환경 생성 및 패키지 설치 중...")
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])
        required = ["pip", "setuptools", "wheel", "pyinstaller", "ccxt", "pyupbit", "python-dotenv", 
                    "pandas", "numpy", "tensorflow", "tf2onnx", "onnxruntime", "scikit-learn", 
                    "psutil", "matplotlib", "streamlit", "websocket-client"]
        subprocess.check_call([python_exe, "-m", "pip", "install", "--upgrade"] + required)

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

def build_mac():
    work_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(work_dir)
    import PyInstaller.__main__
    from PyInstaller.utils.hooks import collect_submodules, collect_data_files, copy_metadata

    for folder in ["dist_mac", "build"]:
        if os.path.exists(folder): shutil.rmtree(folder)
    os.makedirs("dist_mac", exist_ok=True)

    # Mac uses .icns for icons
    icon_args = [f'--icon=icon.icns'] if os.path.exists("icon.icns") else []

    # 1. Trading Bot 빌드
    print("\n🤖 1/3: Trading Bot 빌드 시작")
    tf_datas = collect_data_files('tensorflow')
    main_args = ['main.py', '--name=TradingBot', '--onefile', '--clean', '--distpath=dist_mac'] + icon_args
    for hi in ['tensorflow', 'onnxruntime', 'tf2onnx', 'sklearn.utils._typedefs', 'websocket']:
        main_args.append(f'--hidden-import={hi}')
    for src, dest in tf_datas:
        main_args.append(f'--add-data={src}{os.pathsep}{dest}')
    PyInstaller.__main__.run(main_args)

    # 2. Dashboard 빌드
    print("\n📈 2/3: Dashboard 빌드 시작")
    dash_args = ['run_dashboard.py', '--name=Dashboard', '--onefile', '--noconsole', '--clean', '--distpath=dist_mac'] + icon_args
    
    st_datas = collect_data_files('streamlit')
    st_meta = copy_metadata('streamlit')
    st_hidden = collect_submodules('streamlit')

    if os.path.exists('dashboard.py'):
        dash_args.append(f'--add-data=dashboard.py{os.pathsep}.')
    
    for src, dest in st_datas: dash_args.append(f'--add-data={src}{os.pathsep}{dest}')
    for m_src, m_dest in st_meta: dash_args.append(f'--add-data={m_src}{os.pathsep}{m_dest}')
    for h in st_hidden + ['streamlit.web.cli', 'pyupbit', 'config', 'websocket']:
        dash_args.append(f'--hidden-import={h}')
    
    PyInstaller.__main__.run(dash_args)

    # 3. Analyze Performance 빌드
    print("\n📊 3/3: Analyze Performance 빌드 시작")
    PyInstaller.__main__.run(['analyze_performance.py', '--name=AnalyzePerformance', '--onefile', '--clean', '--distpath=dist_mac'] + icon_args)

    # 마무리 작업
    for folder in ['data', 'logs', 'models', 'config']: os.makedirs(os.path.join("dist_mac", folder), exist_ok=True)
    if os.path.exists(".env"): shutil.copy(".env", "dist_mac/.env")
    
    # 실행 권한 부여 (Mac/Linux)
    try:
        subprocess.check_call(['chmod', '+x', 'dist_mac/TradingBot'])
        subprocess.check_call(['chmod', '+x', 'dist_mac/Dashboard'])
        subprocess.check_call(['chmod', '+x', 'dist_mac/AnalyzePerformance'])
    except Exception as e:
        print(f"⚠️ 실행 권한 설정 실패: {e}")

    print("\n✅ 빌드 완료! dist_mac 폴더를 확인하세요.")

if __name__ == "__main__":
    if "--reset-venv" in sys.argv:
        if os.path.exists("venv"): shutil.rmtree("venv")
        sys.argv.remove("--reset-venv")
    check_and_setup_venv()
    build_mac()