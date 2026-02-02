import logging
import logging.handlers
import os
import time
import zipfile
from datetime import datetime

LOG_DIR = "logs"

def compress_old_logs(days=7):
    """7일 이상 지난 로그 파일 압축"""
    if not os.path.exists(LOG_DIR):
        return
        
    now = time.time()
    cutoff = now - (days * 86400)
    
    for filename in os.listdir(LOG_DIR):
        file_path = os.path.join(LOG_DIR, filename)
        
        # 파일이고, .zip이 아닌 경우
        if os.path.isfile(file_path) and not filename.endswith(".zip"):
            # 로그 파일인 경우 (.log가 포함된 파일)
            if "log" in filename:
                try:
                    # 수정 시간이 cutoff보다 오래된 경우
                    if os.path.getmtime(file_path) < cutoff:
                        zip_path = file_path + ".zip"
                        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                            zf.write(file_path, filename)
                        os.remove(file_path)
                except Exception:
                    pass

def _zip_namer(name):
    """로테이션된 로그 파일명에 .zip 확장자 추가"""
    return name + ".zip"

def _zip_rotator(source, dest):
    """로그 파일을 압축하여 로테이션"""
    with open(source, "rb") as f_in:
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(source, os.path.basename(source))
    os.remove(source)

def setup_logger(name: str = "trading_bot", 
                 log_level: int = logging.INFO,
                 filename: str = None) -> logging.Logger:
    """로거 설정"""
    
    # 로그 디렉토리 생성
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # [New] 오래된 로그 압축 (설정 시 1회 수행)
    compress_old_logs(days=7)
    
    # 로거 생성
    logger = logging.getLogger(name)
    # [Change] 핸들러별 레벨 제어를 위해 로거 자체는 최저 레벨(DEBUG)로 설정
    logger.setLevel(logging.DEBUG)
    
    # 기존 핸들러 제거 (중복 방지)
    if logger.handlers:
        logger.handlers.clear()
    
    # 포매터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 파일 핸들러 설정
    if filename:
        # 특정 파일명이 주어지면 (예: 백테스트) 일반 FileHandler 사용
        log_file = os.path.join(LOG_DIR, filename)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
    else:
        # 메인 봇 로그: 날짜별 분리 (TimedRotatingFileHandler)
        log_file = os.path.join(LOG_DIR, f"{name}.log")
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file,
            when='midnight',
            interval=1,
            backupCount=60, # 60일치 보관 (압축된 파일은 별도 관리)
            encoding='utf-8'
        )
        file_handler.suffix = "%Y%m%d" # 파일명 뒤에 날짜 붙임
        file_handler.setLevel(log_level)

        # [New] 디버그 전용 로그 파일 (bot_debug.log)
        debug_log_file = os.path.join(LOG_DIR, "bot_debug.log")
        debug_handler = logging.handlers.RotatingFileHandler(
            debug_log_file,
            maxBytes=20*1024*1024, # 20MB
            backupCount=5,
            encoding='utf-8'
        )
        debug_handler.namer = _zip_namer
        debug_handler.rotator = _zip_rotator
        debug_handler.setLevel(logging.DEBUG) # 항상 DEBUG 수집
        debug_handler.setFormatter(formatter)
        logger.addHandler(debug_handler)

    file_handler.setFormatter(formatter)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) # [Request] 콘솔은 INFO 고정
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
