# 📈 AI Hybrid Auto Trading Bot (Stock & Crypto)

한국 주식(신한/키움/대신)과 암호화폐(Upbit/Binance)를 동시에 지원하는 하이브리드 자동매매 봇입니다.
머신러닝(Random Forest/LSTM) 기반의 예측 모델과 기술적 분석 전략을 결합하여 운영됩니다.

## 📂 프로젝트 구조 (Project Structure)

```
stock/
├── config/                 # 설정
│   └── settings.py        # 전역 설정
├── api/                   # API 연결
│   ├── base_api.py       # API 기본 클래스
│   ├── shinhan_api.py    # 신한투자 API
│   ├── kiwoom_api.py     # 키움증권 API
│   ├── daishin_api.py    # 대신증권 API
│   └── crypto_api.py     # 암호화폐 API (업비트, 바이낸스)
├── models/                # 머신러닝 모델
│   └── ml_model.py       # ML 예측 모델
├── trading/               # 거래 로직
│   ├── portfolio.py      # 포트폴리오 관리
│   ├── strategy.py       # 거래 전략
│   └── risk_manager.py   # 위험 관리
├── utils/                 # 유틸리티
│   └── logger.py         # 로깅 설정
├── logs/                  # 로그 파일
├── data/                  # 데이터 저장
├── main.py               # 메인 실행 파일
├── requirements.txt      # 필요 라이브러리
├── .env                  # 환경 변수
└── README.md            # 문서
```

## 주요 기능

### 1. API 연결
- **신한투자 API**: 한국주식 거래
- **키움증권 API**: 한국주식 거래
- **대신증권 API**: 한국주식 거래
- **암호화폐 API**: 업비트, 바이낸스를 통한 암호화폐 거래

### 2. 머신러닝 기반 예측
- **LSTM, Random Forest 모델**: 시계열 데이터 분석
- **자동 모델 학습**: 매일 자정에 최신 데이터로 재학습
- **신뢰도 기반 거래**: 모델 신뢰도에 따른 선택적 거래

### 3. 거래 전략
- **머신러닝 전략**: ML 모델의 예측을 기반한 거래
- **기술적 지표 전략**: RSI, MACD 등 기술적 지표 활용

### 4. 포트폴리오 관리
- **포지션 추적**: 각 종목별 수량, 매입가 추적
- **손익 계산**: 실현/미실현 손익 계산
- **포트폴리오 요약**: 보유 자산 상태 모니터링

### 5. 위험 관리
- **손실제한(Stop Loss)**: 설정한 손실률 이상 시 자동 매도
- **수익실현(Take Profit)**: 목표 수익률 도달 시 자동 매도
- **포지션 크기 제한**: 최대 포지션 비율 설정

### 6. 실시간 모니터링
- **정기 점검**: 설정된 주기로 시장 모니터링
- **자동 거래**: 신호 발생 시 자동 주문 실행
- **로깅**: 모든 거래 내역 및 오류 기록

## 설치 및 실행

### 1. 필수 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 2. API 설정

`.env` 파일에 API 키와 시크릿을 입력합니다:

```
SHINHAN_API_KEY=your_key
SHINHAN_API_SECRET=your_secret
UPBIT_API_KEY=your_key
UPBIT_API_SECRET=your_secret
...
```

### 2. 설정 파일 수정

`config/settings.py`에서 다음을 설정합니다:

```python
# 거래 종목
TRADING_CONFIG = {
    "korean_stocks": {
        "symbols": ["005930", "000660"],  # 원하는 종목 추가
    },
    "crypto": {
        "symbols": ["BTC/KRW", "ETH/KRW"],  # 원하는 암호화폐 추가
    }
}

# 거래 금액
TRADING_CONFIG["korean_stocks"]["initial_capital"] = 10000000  # 1천만원
TRADING_CONFIG["crypto"]["initial_capital"] = 5000000  # 500만원
```

### 3. 증권사 선택

다중 증권사를 동시에 사용 가능합니다. `main.py`에서 필요한 API를 활성화하세요:

```python
# 신한투자
self.shinhan_api = ShinhanAPI(SHINHAN_API_KEY, SHINHAN_API_SECRET, SHINHAN_ACCOUNT)

# 키움증권
self.kiwoom_api = KiwoomAPI(KIWOOM_API_KEY, KIWOOM_API_SECRET, KIWOOM_ACCOUNT)

# 대신증권
self.daishin_api = DaishinAPI(DAISHIN_API_KEY, DAISHIN_API_SECRET, DAISHIN_ACCOUNT)
```

### 4. 봇 시작

```bash
python main.py
```

## 주요 클래스

### AutoTradingBot
메인 봇 클래스로 전체 거래 프로세스를 관리합니다.

**주요 메서드:**
- `initialize_apis()`: API 초기화
- `train_ml_model()`: ML 모델 학습
- `monitor_and_trade()`: 모니터링 및 거래 실행
- `start()`: 봇 시작
- `stop()`: 봇 종료

### MLPredictor
머신러닝 기반 가격 예측 모델입니다.

**주요 메서드:**
- `train()`: 모델 학습
- `predict()`: 다음 기간 가격 예측
- `predict_direction()`: 가격 방향성 예측

### Portfolio
포트폴리오 관리 클래스입니다.

**주요 메서드:**
- `add_position()`: 포지션 추가
- `close_position()`: 포지션 종료
- `get_total_value()`: 포트폴리오 총 가치
- `get_statistics()`: 포트폴리오 통계

### RiskManager
위험 관리 클래스입니다.

**주요 메서드:**
- `set_stop_loss()`: 손실제한 설정
- `set_take_profit()`: 수익실현 설정
- `check_exit_conditions()`: 매도 조건 확인

## 개발 팁

### 1. 모델 성능 개선
- 학습 데이터 량 증가
- 하이퍼파라미터 튜닝
- 다양한 기술적 지표 추가

### 2. 거래 전략 추가
새로운 전략을 추가하려면 `trading/strategy.py`의 `TradingStrategy` 클래스를 상속합니다:

```python
class MyStrategy(TradingStrategy):
    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[Signal]:
        # 거래 신호 생성 로직
        return Signal(...)
```

### 3. 백테스팅
과거 데이터로 전략을 검증하려면:

```python
from backtesting.walk_forward import WalkForwardAnalyzer

analyzer = WalkForwardAnalyzer(data, train_period=252, test_period=63)
results = analyzer.run()
print(results)
```

## 주의사항

⚠️ **실제 거래 전에:**
- 데모 계정에서 충분히 테스트하세요
- 작은 금액으로 시작하세요
- 시장 상황을 항상 모니터링하세요
- API 키는 절대 코드에 직접 입력하지 마세요
- 정기적으로 로그를 확인하세요

## 문제 해결

### API 연결 실패
- API 키와 시크릿이 올바른지 확인하세요
- 네트워크 연결을 확인하세요
- API 서비스 상태를 확인하세요

### 거래가 실행되지 않음
- 자본이 충분한지 확인하세요
- 거래 시간대가 맞는지 확인하세요
- 로그 파일에서 오류 메시지를 확인하세요

## 라이선스

이 프로젝트는 학습용으로 제공됩니다.


 필수 복사 항목 (이건 꼭 가져가야 해요!)
data 폴더 (가장 중요)

이 안에 있는 crypto_portfolio.json과 stock_portfolio.json 파일에 현재 보유 중인 종목, 평단가, 매수 수량, 거래 내역이 모두 저장되어 있습니다.
이 폴더를 안 가져가면 봇은 "아무것도 안 들고 있다"고 생각해서 새로 시작하게 됩니다.
.env 파일

API 키(업비트 등)와 봇 설정값이 들어있습니다.
새 PC에서 다시 설정하기 귀찮다면 이 파일을 그대로 복사해 가세요.
📂 선택 복사 항목 (필수는 아님)
models 폴더

봇이 학습한 AI 모델 파일(*.pkl)들이 들어있습니다.
가져가면 새 PC에서 봇을 켰을 때 처음부터 다시 학습하지 않아도 되어 시작 속도가 빨라집니다. (안 가져가면 봇이 켜질 때 자동으로 다시 학습합니다.)
logs 폴더

과거의 실행 기록(로그)입니다. 기록을 보관하고 싶다면 가져가세요.
🚀 새 PC로 이사하는 순서
새 PC에 폴더를 하나 만듭니다 (예: MyBot).
TradingBot.exe와 Dashboard.exe를 그 폴더에 넣습니다.
기존 PC에서 가져온 data 폴더와 .env 파일을 같은 폴더에 붙여넣습니다.
TradingBot.exe를 실행하면 기존 매수 내역을 그대로 인식하고 매매를 이어갑니다.
