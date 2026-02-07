# 📘 자동매매 시스템 기술 분석 및 상세 보고서 (초보자 및 개발자 겸용)

## 1. 시스템 아키텍처 및 개요 (System Overview)

이 시스템은 **"24시간 쉬지 않고 주식과 코인을 대신 사고팔아주는 로봇"**입니다.
사용자가 전략(언제 살지, 언제 팔지)을 정해주면, 로봇이 거래소(업비트, 바이낸스 등)를 계속 감시하다가 조건이 맞을 때 자동으로 주문을 넣습니다.

### 1.1 전체 구조도 (어떻게 연결되어 있나요?)

시스템은 크게 4가지 부품으로 이루어져 있습니다.

1.  **대시보드 (`dashboard.py`)**: **리모컨**입니다. 현재 자산이 얼마인지 보고, 전략을 바꾸거나 봇을 끄고 켤 수 있습니다.
2.  **메인 엔진 (`main.py`)**: **로봇의 몸체**입니다. 실제로 일을 하는 곳입니다. 대시보드의 명령을 듣고, 거래소에 주문을 보냅니다.
3.  **전략 두뇌 (`strategy.py`)**: **로봇의 뇌**입니다. "지금 살까? 말까?"를 계산합니다.
4.  **데이터 저장소 (`data/`)**: **로봇의 수첩**입니다. 내가 뭘 샀는지, 얼마에 샀는지 적어두는 곳입니다.

```mermaid
graph TD
    User[사용자] -->|설정 변경| Config[config/settings.py & .env]
    User -->|모니터링/제어| Dashboard[dashboard.py]
    User[사용자] -->|1. 화면 보고 조작| Dashboard[대시보드 (dashboard.py)]
    
    Dashboard -->|명령 전달 (JSON)| CommandJSON[data/command.json]
    Dashboard <--|상태 읽기 (JSON)| StatusJSON[data/bot_status.json]
    Dashboard <--|자산 읽기 (JSON)| PortfolioJSON[data/*_portfolio.json]
    Dashboard -->|2. 명령 내림 (파일로 저장)| CommandJSON[명령 파일 (data/command.json)]
    Dashboard <--|3. 상태 확인 (파일 읽기)| StatusJSON[상태 파일 (data/bot_status.json)]
    
    
    Main -->|API 호출 (REST/WS)| Exchange[거래소 API (Upbit/Binance/Stock)]
    Main -->|전략 계산| Strategy[trading/strategy.py]
    Main -->|로그 기록| Logs[logs/*.log]
    Main -->|6. 가격 조회 & 주문| Exchange[거래소 (업비트/바이낸스)]
    Main -->|7. 살까 말까 질문| Strategy[전략 두뇌 (trading/strategy.py)]
    
    Main -->|6. 가격 조회 & 주문| Exchange[거래소 (업비트/바이낸스)]
    Main -->|7. 살까 말까 질문| Strategy[전략 두뇌 (trading/strategy.py)]
    
    Backtest[run_backtest_current.py] -->|과거 데이터 분석| Exchange
    Backtest -->|전략 검증| Strategy
    Strategy -->|8. 매수/매도 신호| Main
```

---

## 2. 데이터 및 파라미터 흐름 (Data & Parameter Flow)

시스템 내에서 설정값과 데이터가 어떻게 흘러가는지 설명합니다.

### 2.1 파라미터 주입 및 갱신 메커니즘
1.  **정적 설정 로드 (`config/settings.py`)**:
    *   `TRADING_CONFIG`: 거래소별(Upbit, Binance Spot/Futures, Stock) 설정을 담은 딕셔너리입니다.
    *   `STRATEGY_PRESETS`: 'scalping', 'mid_term' 등 사전 정의된 전략 파라미터 집합입니다.
    *   `.env` 파일: API Key, Secret 등 민감 정보와 `CRYPTO_STRATEGY_PRESET` 같은 환경 변수를 로드합니다.
2.  **초기화 (Initialization - `main.py`)**:
    *   `AutoTradingBot.__init__`: `TRADING_CONFIG`를 참조하여 `Portfolio`, `RiskManager`, `Strategy` 객체를 초기화합니다.
    *   `initialize_apis()`: `.env` 정보를 바탕으로 거래소 API 객체(`UpbitAPI`, `BinanceAPI` 등)를 생성하고 연결을 테스트합니다.
3.  **동적 변경 (Hot Reload & Command)**:
    *   **파일 감시 (`check_env_updates`)**: `.env` 파일의 수정 시간(mtime)을 주기적으로 확인하여, 변경 시 메모리 상의 설정을 즉시 갱신합니다. (K값, 익절률 등 즉시 반영)
    *   **대시보드 명령 (`_check_for_commands`)**: `dashboard.py`에서 전략 변경 시 `data/command.json`을 생성합니다. 메인 루프가 이를 감지하여 `update_strategy_config()`를 호출, 실행 중인 봇의 전략(타임프레임, 익절률 등)을 즉시 변경합니다.

### 2.2 데이터 지속성 (Persistence Layer) - JSON 스키마
로컬 파일 시스템을 데이터베이스처럼 활용하여 봇의 상태를 영구 저장합니다.

#### A. `data/*_portfolio.json` (자산 장부)
```json
{
  "initial_capital": 1000000,      // 초기 자본금
  "current_capital": 1050000,      // 현재 예수금 (평가금 미포함)
  "positions": {                   // 보유 종목 및 수량
    "BTC/KRW": 0.0015,
    "ETH/KRW": 0.5
  },
  "entry_prices": {                // 평단가
    "BTC/KRW": 50000000,
    "ETH/KRW": 3000000
  },
  "trade_history": [               // 매매 이력 (대시보드 차트용)
    {
      "timestamp": "2023-10-27 10:00:00",
      "symbol": "BTC/KRW",
      "type": "SELL",
      "pnl": 50000,
      "pnl_percent": 5.0
    }
  ],
  "metadata": {                    // 현재 적용된 전략 메타데이터
    "strategy": "scalping",
    "timeframe": "1m",
    "selected_symbols": ["BTC/KRW", "ETH/KRW"]
  }
}
```

#### B. `data/bot_status.json` (상태 정보)
```json
{
  "status": "running",             // running, stopped, warming_up
  "timestamp": 1698370000.123,     // 마지막 생존 신호 (Heartbeat)
  "cpu": 15.5,                     // CPU 사용률 (%)
  "memory": 120.5,                 // 메모리 사용량 (MB)
  "warmup_current": 3,             // 현재 웜업 카운트
  "warmup_total": 3                // 목표 웜업 카운트
}
```


-## 2. 데이터 및 파라미터 흐름 (Data & Parameter Flow) +## 2. 데이터 및 파라미터 흐름 (설정은 어떻게 바뀌나요?)

-시스템 내에서 설정값과 데이터가 어떻게 흘러가는지 설명합니다. +로봇이 어떻게 설정을 읽고 행동을 바꾸는지 설명합니다.

-### 2.1 파라미터 주입 및 갱신 메커니즘 -1. 정적 설정 로드 (config/settings.py):

TRADING_CONFIG: 거래소별(Upbit, Binance Spot/Futures, Stock) 설정을 담은 딕셔너리입니다.
STRATEGY_PRESETS: 'scalping', 'mid_term' 등 사전 정의된 전략 파라미터 집합입니다.
.env 파일: API Key, Secret 등 민감 정보와 CRYPTO_STRATEGY_PRESET 같은 환경 변수를 로드합니다. -2. 초기화 (Initialization - main.py):
AutoTradingBot.__init__: TRADING_CONFIG를 참조하여 Portfolio, RiskManager, Strategy 객체를 초기화합니다.
initialize_apis(): .env 정보를 바탕으로 거래소 API 객체(UpbitAPI, BinanceAPI 등)를 생성하고 연결을 테스트합니다. -3. 동적 변경 (Hot Reload & Command):
파일 감시 (check_env_updates): .env 파일의 수정 시간(mtime)을 주기적으로 확인하여, 변경 시 메모리 상의 설정을 즉시 갱신합니다. (K값, 익절률 등 즉시 반영)
대시보드 명령 (_check_for_commands): dashboard.py에서 전략 변경 시 data/command.json을 생성합니다. 메인 루프가 이를 감지하여 update_strategy_config()를 호출, 실행 중인 봇의 전략(타임프레임, 익절률 등)을 즉시 변경합니다. +### 2.1 설정이 적용되는 과정 +1. 시작할 때: 로봇을 켜면 config/settings.py와 .env 파일(비밀번호 파일)을 읽어서 기억합니다. +2. 실행 중에 바꿀 때:
파일 수정: 사용자가 .env 파일을 메모장으로 열어서 CRYPTO_K_VALUE=0.5 처럼 숫자를 바꾸고 저장하면, 로봇이 "어? 파일이 바뀌었네?" 하고 알아채서 즉시 새로운 값을 적용합니다. (이걸 Hot Reload라고 합니다.)
대시보드 버튼: 대시보드에서 '전략 변경' 버튼을 누르면 data/command.json이라는 파일에 쪽지를 남깁니다. 로봇은 3초마다 이 쪽지함을 확인하다가 쪽지가 있으면 전략을 바꿉니다.
-### 2.2 데이터 지속성 (Persistence Layer) - JSON 스키마 -로컬 파일 시스템을 데이터베이스처럼 활용하여 봇의 상태를 영구 저장합니다. +### 2.2 데이터 저장 방식 (꺼져도 기억하는 이유) +로봇은 컴퓨터가 꺼지거나 프로그램이 종료되어도 내가 뭘 가지고 있는지 잊어버리면 안 됩니다. 그래서 JSON 파일이라는 텍스트 파일에 장부를 기록합니다.

-* data/*_portfolio.json: 핵심 장부 데이터.

positions: 현재 보유 종목 및 수량 ({"BTC/KRW": 0.01}).
entry_prices: 평단가 정보.
trade_history: 매매 이력 리스트 (대시보드 수익률 차트용).
metadata: 현재 적용된 전략 메타데이터
"strategy": "scalping",
"timeframe": "1m",
"selected_symbols": ["BTC/KRW", "ETH/KRW"]
} -} -```
-#### B. data/bot_status.json (상태 정보) -```json -{

"status": "running", // running, stopped, warming_up
"timestamp": 1698370000.123, // 마지막 생존 신호 (Heartbeat)
"cpu": 15.5, // CPU 사용률 (%)
"memory": 120.5, // 메모리 사용량 (MB)
"warmup_current": 3, // 현재 웜업 카운트
"warmup_total": 3 // 목표 웜업 카운트 -} -``` +* data/crypto_portfolio.json (자산 장부):
positions: "비트코인 0.5개, 이더리움 10개 가지고 있음"
entry_prices: "비트코인은 5천만원에 샀고, 이더리움은 300만원에 샀음" (평단가)
이 파일이 지워지면 로봇은 "나는 아무것도 안 가지고 있다"고 생각하게 되므로 절대 지우면 안 됩니다.


---

## 3. 핵심 프로그램 상세 분석

### 3.1 메인 엔진 (`main.py`)

`AutoTradingBot` 클래스가 전체 라이프사이클을 관리합니다.

#### 3.1.1 주요 실행 루프 (`monitor_and_trade`)
스케줄러에 의해 주기적(기본 3초)으로 호출되며, `trade_lock`을 사용하여 중복 실행을 방지합니다.

1.  **웜업 체크 (Warm-up)**: 초기 실행 시 데이터 수집 안정화를 위해 첫 3회 루프는 매매를 건너뜁니다.
2.  **설정 갱신 (Hot Reload)**: `.env` 변경 사항을 확인합니다.
3.  **거래소별 처리**: `_trade_korean_stocks`, `_trade_upbit`, `_trade_binance_spot`, `_trade_binance_futures`를 순차 실행합니다.

#### 3.1.2 암호화폐 거래 로직 (`_process_crypto_trading`)
가장 복잡한 로직을 담고 있으며, **보유 종목 관리(매도)**와 **신규 진입(매수)** 두 단계로 나뉩니다.

*   **Phase 1: 보유 종목 관리 (Selling Logic)**
    1.  **OCO 감시 (Binance Spot)**: `oco_monitoring_symbols`에 등록된 종목의 미체결 주문 상태를 확인합니다. 주문이 사라졌다면(체결 또는 취소), 잔고를 재조회하여 전량 매도되었는지 확인합니다. 매도 완료 시 포트폴리오에서 제거하고, 취소되어 잔고가 남아있다면 즉시 봇의 실시간 감시 모드로 전환하여 매도 기회를 다시 포착합니다.
    2.  **비상 손절 (Emergency Stop)**: 평단가 대비 **-5% 이상 급락**하는 상황이 발생하면, 현재 전략의 판단이나 보조지표의 상태와 무관하게 **무조건 시장가로 매도**합니다. 이는 ATR 정보가 유실되거나 API 오류로 인해 정상적인 손절 로직이 작동하지 않을 때를 대비한 최후의 물리적 안전장치입니다.
    3.  **리스크 관리 체크**: `RiskManager.check_exit_conditions`를 호출하여 설정된 익절가(Take Profit), 손절가(Stop Loss), 트레일링 스탑(Trailing Stop) 가격에 도달했는지 확인합니다.
    4.  **전략 매도 신호**: `Strategy.generate_signal` 메소드를 호출하여 기술적 지표(RSI 과매수, 볼린저밴드 이탈 등)에 의한 매도 신호(`SELL`)가 발생했는지 확인합니다.
    5.  **매도 실행**: `_execute_sell` 메소드를 통해 주문을 전송합니다. 이때 최소 주문 금액(업비트 5,000원, 바이낸스 10 USDT) 미만인 경우 'Dust(먼지)'로 간주하여 매도하지 않고 포트폴리오에서 제외 처리하거나(Upbit), BNB로 변환을 시도합니다(Binance).

*   **Phase 2: 신규 진입 (Buying Logic)**
    1.  **종목 선정**: `VOLUME_CONFIG`에 의해 선정된 거래량 상위 종목(`crypto_symbols`)을 순회합니다.
    2.  **보유 제한 체크**: `max_positions`를 초과하면 신규 매수를 중단합니다.
    3.  **MTF 필터 (Multi-Timeframe)**: 알트코인 매매 시, 1시간봉 EMA50 위에 있는지 확인하여 대세 상승장에서만 진입합니다.
    4.  **데이터 수집**: `_get_latest_ohlcv`를 통해 캔들 데이터를 가져옵니다. (캐싱 적용됨)
    5.  **신호 생성**: `Strategy.generate_signal`을 호출하여 `BUY` 신호를 확인합니다.
    6.  **자금 관리**: `RiskManager` 또는 전략에서 계산된 `suggested_quantity`를 기반으로 매수 금액을 산정합니다.
    7.  **매수 실행**: 잔액 확인 -> 주문 전송 -> 포트폴리오 등록 -> `RiskManager`에 손절/익절가 등록 순으로 진행됩니다.

#### 3.1.3 자동 최적화 (`optimize_strategy_params`)
매일 아침 또는 봇 시작 시 실행됩니다.
1.  최근 7일간의 데이터를 로드합니다.
2.  K값(0.4~0.6), 익절률(3~10%), 손절률(1~5%)의 조합(Grid Search)으로 백테스팅을 수행합니다.
3.  가장 높은 승률을 기록한 파라미터 조합을 찾아 메모리와 `.env` 파일에 반영합니다.

#### 3.1.4 K값 미니 최적화 (`find_best_k`)
4시간마다 실행되는 단기 최적화 로직입니다. 시장의 단기 변동성 변화에 빠르게 적응하기 위함입니다.
1.  최근 200개의 캔들 데이터를 로드합니다.
2.  K값 0.3부터 0.8까지 0.05 단위로 시뮬레이션(백테스팅)을 수행합니다.
3.  가장 높은 수익률을 기록한 K값을 찾아 메모리 상의 설정(`TRADING_CONFIG`)에 즉시 반영합니다.

#### 3.1.5 병렬 머신러닝 학습 (`train_ml_model`)
`concurrent.futures.ProcessPoolExecutor`를 활용하여 학습 속도를 극대화합니다.
1.  **데이터 수집**: API를 통해 각 종목의 OHLCV 데이터를 가져옵니다.
2.  **병렬 처리 (Multiprocessing)**: CPU 코어 수만큼 워커 프로세스를 생성하여 종목별로 독립적인 학습 태스크(`_train_model_task`)를 할당합니다. 각 프로세스는 독립된 메모리 공간을 가지므로 GIL(Global Interpreter Lock)의 영향을 받지 않고 고속으로 연산합니다.
3.  **지표 선행 계산**: 각 워커는 학습 전에 RSI, MACD, Bollinger Bands 등의 보조지표를 미리 계산하여 데이터프레임에 추가합니다. 이는 학습 및 백테스팅 속도를 획기적으로 높여줍니다.
4.  **검증 및 저장**: 각 태스크는 `WalkForwardAnalyzer`를 사용하여 최근 데이터에 대한 전진 분석(Walk-Forward Analysis)을 수행합니다. 검증 결과 수익이 긍정적(Total Return > -10000)인 경우에만 모델을 파일로 저장하여, 성능이 입증된 모델만 실전 매매에 투입되도록 합니다.

---

### 3.2 전략 모듈 (`trading/strategy.py`)

매매 의사결정을 담당하는 핵심 로직입니다. `generate_signal` 메소드가 핵심입니다.

#### 3.2.1 `TechnicalStrategy` 상세 로직
다양한 기술적 지표를 조합하여 신호를 생성합니다.

*   **공통 필터 (Filters)**:
    *   **변동성 필터**: ATR이 가격의 일정 비율(예: 0.8%) 미만이면 횡보장으로 판단하여 `HOLD`.
    *   **추세 필터**: 현재가가 EMA 50(지수이동평균)보다 낮으면 하락장으로 판단하여 매수 금지.
    *   **ADX 필터**: ADX 지표가 20 미만이거나 하락 중이면 추세가 약하다고 판단하여 진입 보류.

*   **주요 전략 모드 (`entry_strategy`)**:
    *   **`breakout` (변동성 돌파)**:
        *   `Range = 전일 고가 - 전일 저가`
        *   `Target = 당일 시가 + (Range * K)`
        *   조건: `현재가 > Target` AND `현재가 > EMA50`
    *   **`combined` (복합 전략)**:
        *   RSI 과매도(<35) + 볼린저밴드 하단 터치 + MACD 골든크로스 등 여러 지표를 종합 판단.
        *   각 신호마다 가중치(Confidence)를 부여하여 평균 신뢰도가 임계값(0.5)을 넘으면 매수.
    *   **`bb_breakout` (볼린저 밴드 돌파)**:
        *   조건: `현재가 > 상단 밴드` AND `거래량 급등(1.2배)` AND `밴드폭 확장`
        *   추세 추종형 전략으로, 강한 상승세 초입을 포착합니다.
    *   **`pullback` (눌림목)**:
        *   조건: `현재가 > EMA100` (장기 상승세) AND `RSI < 50` (단기 조정) AND `RSI 상승 반전`
    *   **`dynamic_breakout` (동적 돌파)**:
        *   조건: `현재가 > EMA50` (상승 추세 확인) AND `현재가 > 볼린저밴드 상단` (강한 모멘텀 발생) AND `ADX > 20` (추세 강도 확보). 상승장의 초입이나 가속 구간을 포착하는 전략입니다.
    *   **`rsi_bollinger` (역추세)**:
        *   매수: `RSI < 30` (과매도 상태) AND `현재가 < 볼린저밴드 하단` (통계적으로 매우 낮은 가격). 과매도 구간에서의 기술적 반등을 노립니다.
        *   매도: `RSI > 70` (과매수) OR `현재가 > 볼린저밴드 상단` (통계적 고점).

#### 3.2.2 `MLStrategy` 상세 로직
머신러닝 모델(`models/ml_model.py`)을 사용하여 예측합니다.
*   **입력**: 최근 OHLCV 데이터 (Lookback Window: 60~200).
*   **전처리**: MinMaxScaler를 사용하여 0~1 사이로 정규화.
*   **모델 아키텍처 (LSTM)**:
    *   **Layer 1**: LSTM (64 units, return_sequences=True) + L2 Regularization + Dropout(0.2)
    *   **Layer 2**: LSTM (32 units) + L2 Regularization + Dropout(0.2)
    *   **Output**: Dense(1) - 다음 시점의 종가(Close Price) 예측 (Regression)
    *   **Optimizer**: Adam (Learning Rate: 0.001)
    *   **Loss Function**: MSE (Mean Squared Error)
*   **출력**: 다음 캔들의 종가 예측.
*   **신호 생성**:
    *   `예측가 > 현재가 * 1.001` (0.1% 이상 상승 예측) -> `UP` -> `BUY`
    *   `예측가 < 현재가 * 0.999` (0.1% 이상 하락 예측) -> `DOWN` -> `SELL`
    *   신뢰도(Confidence)는 최근 변동성이 낮을수록 높게 책정됩니다.

---

### 3.3 대시보드 (`dashboard.py`)
Streamlit 프레임워크를 사용한 웹 UI입니다.

*   **실시간 데이터 연동**: `data/*.json` 파일을 매번 다시 읽어(Polling) 최신 상태를 보여줍니다.
*   **봇 제어**: '전략 적용하기', '봇 재시작' 등의 버튼을 누르면 `data/command.json` 파일을 생성하여 `main.py`에 비동기적으로 명령을 전달합니다.
*   **로그 뷰어**: `logs/` 폴더의 최신 로그 파일을 읽어 화면에 출력하며, 에러/경고 메시지에 색상 강조(HTML/CSS Injection)를 적용합니다.


-## 3. 핵심 프로그램 상세 분석 +## 3. 핵심 프로그램 상세 분석 (로봇 해부)

-### 3.1 메인 엔진 (main.py) +### 3.1 메인 엔진 (main.py) - 로봇의 하루 일과

-AutoTradingBot 클래스가 전체 라이프사이클을 관리합니다. +이 파일이 실행되면 로봇은 아래 행동을 무한히 반복합니다. (monitor_and_trade 함수)

-#### 3.1.1 주요 실행 루프 (monitor_and_trade) -스케줄러에 의해 주기적(기본 3초)으로 호출되며, trade_lock을 사용하여 중복 실행을 방지합니다. +1. 준비 운동 (Warm-up): 처음 켜지면 바로 매매하지 않고 3번 정도 시장 가격만 지켜봅니다. 데이터가 충분히 모일 때까지 기다리는 것입니다. +2. 쪽지 확인: 대시보드에서 온 명령(종료, 재시작 등)이 있는지 확인합니다. +3. 가격 확인: 업비트나 바이낸스 서버에 "지금 비트코인 얼마야?" 하고 물어봅니다. +4. 보유 종목 관리 (매도 판단):

내가 가진 코인이 목표 수익률(예: 5%)을 넘었나? -> 익절 (이익 실현)
내가 가진 코인이 너무 많이 떨어졌나(예: -3%)? -> 손절 (손실 방어)
갑자기 -5% 이상 폭락했나? -> 비상 탈출 (무조건 시장가 매도) +5. 신규 종목 탐색 (매수 판단):
살 만한 좋은 코인이 있나? -> 전략 두뇌(Strategy)에게 물어봅니다.
두뇌가 "사라(BUY)"고 하면 -> 내 지갑에 돈이 있는지 확인하고 주문을 넣습니다. +6. 잠자기: 3초 동안 쉽니다. (너무 자주 물어보면 거래소에서 차단당하기 때문입니다.)
-1. 웜업 체크 (Warm-up): 초기 실행 시 데이터 수집 안정화를 위해 첫 3회 루프는 매매를 건너뜁니다. -2. 설정 갱신 (Hot Reload): .env 변경 사항을 확인합니다. -3. 거래소별 처리: _trade_korean_stocks, _trade_upbit, _trade_binance_spot, _trade_binance_futures를 순차 실행합니다. +### 3.2 전략 모듈 (trading/strategy.py) - 로봇의 판단 기준

-#### 3.1.2 암호화폐 거래 로직 (_process_crypto_trading) -가장 복잡한 로직을 담고 있으며, **보유 종목 관리(매도)**와 신규 진입(매수) 두 단계로 나뉩니다. +"언제 살 것인가?"를 결정하는 수학 공식들이 들어있습니다.

-* Phase 1: 보유 종목 관리 (Selling Logic)

OCO 감시 (Binance Spot): oco_monitoring_symbols에 등록된 종목의 미체결 주문 상태를 확인합니다. 주문이 사라졌다면(체결 또는 취소), 잔고를 재조회하여 전량 매도되었는지 확인합니다. 매도 완료 시 포트폴리오에서 제거하고, 취소되어 잔고가 남아있다면 즉시 봇의 실시간 감시 모드로 전환하여 매도 기회를 다시 포착합니다.
비상 손절 (Emergency Stop): 평단가 대비 -5% 이상 급락하는 상황이 발생하면, 현재 전략의 판단이나 보조지표의 상태와 무관하게 무조건 시장가로 매도합니다. 이는 ATR 정보가 유실되거나 API 오류로 인해 정상적인 손절 로직이 작동하지 않을 때를 대비한 최후의 물리적 안전장치입니다.
리스크 관리 체크: RiskManager.check_exit_conditions를 호출하여 설정된 익절가(Take Profit), 손절가(Stop Loss), 트레일링 스탑(Trailing Stop) 가격에 도달했는지 확인합니다.
전략 매도 신호: Strategy.generate_signal 메소드를 호출하여 기술적 지표(RSI 과매수, 볼린저밴드 이탈 등)에 의한 매도 신호(SELL)가 발생했는지 확인합니다.
매도 실행: _execute_sell 메소드를 통해 주문을 전송합니다. 이때 최소 주문 금액(업비트 5,000원, 바이낸스 10 USDT) 미만인 경우 'Dust(먼지)'로 간주하여 매도하지 않고 포트폴리오에서 제외 처리하거나(Upbit), BNB로 변환을 시도합니다(Binance). +#### 대표적인 전략 설명 (쉽게 풀이)
-* Phase 2: 신규 진입 (Buying Logic)

종목 선정: VOLUME_CONFIG에 의해 선정된 거래량 상위 종목(crypto_symbols)을 순회합니다.
보유 제한 체크: max_positions를 초과하면 신규 매수를 중단합니다.
MTF 필터 (Multi-Timeframe): 알트코인 매매 시, 1시간봉 EMA50 위에 있는지 확인하여 대세 상승장에서만 진입합니다.
데이터 수집: _get_latest_ohlcv를 통해 캔들 데이터를 가져옵니다. (캐싱 적용됨)
신호 생성: Strategy.generate_signal을 호출하여 BUY 신호를 확인합니다.
자금 관리: RiskManager 또는 전략에서 계산된 suggested_quantity를 기반으로 매수 금액을 산정합니다.
매수 실행: 잔액 확인 -> 주문 전송 -> 포트폴리오 등록 -> RiskManager에 손절/익절가 등록 순으로 진행됩니다. +1. 변동성 돌파 (Volatility Breakout):
원리: "어제 많이 움직였던 폭만큼 오늘 오르면, 오늘도 계속 오를 것이다"라고 믿는 전략입니다.
비유: 달리기 선수가 출발선을 힘차게 박차고 나가면 계속 달릴 것이라고 예상하고 같이 뛰는 것과 같습니다.
공식: 오늘 시가 + (어제 최고가 - 어제 최저가) * K 가격을 넘으면 매수합니다.
-#### 3.1.3 자동 최적화 (optimize_strategy_params) -매일 아침 또는 봇 시작 시 실행됩니다. -1. 최근 7일간의 데이터를 로드합니다. -2. K값(0.4~0.6), 익절률(3~10%), 손절률(1~5%)의 조합(Grid Search)으로 백테스팅을 수행합니다. -3. 가장 높은 승률을 기록한 파라미터 조합을 찾아 메모리와 .env 파일에 반영합니다. +2. RSI (상대 강도 지수):

원리: "너무 많이 팔려서 가격이 비정상적으로 싸졌다"고 판단될 때 사는 전략입니다.
비유: 백화점 세일 기간에 물건이 너무 싸지면 사람들이 몰려와서 다시 가격이 오를 것을 노리는 것입니다.
조건: RSI 숫자가 30 밑으로 떨어지면 "과매도(너무 많이 팔림)" 상태라 보고 매수합니다.
-#### 3.1.4 K값 미니 최적화 (find_best_k) -4시간마다 실행되는 단기 최적화 로직입니다. 시장의 단기 변동성 변화에 빠르게 적응하기 위함입니다. -1. 최근 200개의 캔들 데이터를 로드합니다. -2. K값 0.3부터 0.8까지 0.05 단위로 시뮬레이션(백테스팅)을 수행합니다. -3. 가장 높은 수익률을 기록한 K값을 찾아 메모리 상의 설정(TRADING_CONFIG)에 즉시 반영합니다.
-#### 3.1.5 병렬 머신러닝 학습 (train_ml_model) -concurrent.futures.ProcessPoolExecutor를 활용하여 학습 속도를 극대화합니다. -1. 데이터 수집: API를 통해 각 종목의 OHLCV 데이터를 가져옵니다. -2. 병렬 처리 (Multiprocessing): CPU 코어 수만큼 워커 프로세스를 생성하여 종목별로 독립적인 학습 태스크(_train_model_task)를 할당합니다. 각 프로세스는 독립된 메모리 공간을 가지므로 GIL(Global Interpreter Lock)의 영향을 받지 않고 고속으로 연산합니다. -3. 지표 선행 계산: 각 워커는 학습 전에 RSI, MACD, Bollinger Bands 등의 보조지표를 미리 계산하여 데이터프레임에 추가합니다. 이는 학습 및 백테스팅 속도를 획기적으로 높여줍니다. -4. 검증 및 저장: 각 태스크는 WalkForwardAnalyzer를 사용하여 최근 데이터에 대한 전진 분석(Walk-Forward Analysis)을 수행합니다. 검증 결과 수익이 긍정적(Total Return > -10000)인 경우에만 모델을 파일로 저장하여, 성능이 입증된 모델만 실전 매매에 투입되도록 합니다. +3. 볼린저 밴드 (Bollinger Bands):

원리: 가격은 보통 일정한 밴드(띠) 안에서 움직이는데, 이 밴드를 뚫고 올라가면 "뭔가 큰 일이 터졌다(상승 추세)"고 보고 따라 사는 것입니다.
-### 3.2 전략 모듈 (trading/strategy.py) +## 4. 리스크 관리 (돈을 지키는 방법)

-매매 의사결정을 담당하는 핵심 로직입니다. generate_signal 메소드가 핵심입니다. +돈을 버는 것보다 잃지 않는 것이 더 중요합니다. RiskManager가 하는 일입니다.

-#### 3.2.1 TechnicalStrategy 상세 로직 -다양한 기술적 지표를 조합하여 신호를 생성합니다. +### 4.1 자금 관리 (얼마나 살까?) +* 몰빵 금지: 내 돈이 100만원이 있어도 한 종목에 100만원을 다 쓰지 않습니다. +* 설정: max_position_size = 0.2로 설정하면, 한 종목에 최대 20만원(20%)까지만 삽니다. 이렇게 하면 코인 하나가 상장 폐지되어도 내 전 재산은 날아가지 않습니다.

-* 공통 필터 (Filters):

변동성 필터: ATR이 가격의 일정 비율(예: 0.8%) 미만이면 횡보장으로 판단하여 HOLD.
추세 필터: 현재가가 EMA 50(지수이동평균)보다 낮으면 하락장으로 판단하여 매수 금지.
ADX 필터: ADX 지표가 20 미만이거나 하락 중이면 추세가 약하다고 판단하여 진입 보류.
-* 주요 전략 모드 (entry_strategy):

breakout (변동성 돌파):
plaintext
   *   `Range = 전일 고가 - 전일 저가`
plaintext
   *   `Target = 당일 시가 + (Range * K)`
plaintext
   *   조건: `현재가 > Target` AND `현재가 > EMA50`
combined (복합 전략):
plaintext
   *   RSI 과매도(<35) + 볼린저밴드 하단 터치 + MACD 골든크로스 등 여러 지표를 종합 판단.
plaintext
   *   각 신호마다 가중치(Confidence)를 부여하여 평균 신뢰도가 임계값(0.5)을 넘으면 매수.
bb_breakout (볼린저 밴드 돌파):
plaintext
   *   조건: `현재가 > 상단 밴드` AND `거래량 급등(1.2배)` AND `밴드폭 확장`
plaintext
   *   추세 추종형 전략으로, 강한 상승세 초입을 포착합니다.
pullback (눌림목):
plaintext
   *   조건: `현재가 > EMA100` (장기 상승세) AND `RSI < 50` (단기 조정) AND `RSI 상승 반전`
dynamic_breakout (동적 돌파):
plaintext
   *   조건: `현재가 > EMA50` (상승 추세 확인) AND `현재가 > 볼린저밴드 상단` (강한 모멘텀 발생) AND `ADX > 20` (추세 강도 확보). 상승장의 초입이나 가속 구간을 포착하는 전략입니다.
rsi_bollinger (역추세):
plaintext
   *   매수: `RSI < 30` (과매도 상태) AND `현재가 < 볼린저밴드 하단` (통계적으로 매우 낮은 가격). 과매도 구간에서의 기술적 반등을 노립니다.
plaintext
   *   매도: `RSI > 70` (과매수) OR `현재가 > 볼린저밴드 상단` (통계적 고점).
-#### 3.2.2 MLStrategy 상세 로직 -머신러닝 모델(models/ml_model.py)을 사용하여 예측합니다. -* 입력: 최근 OHLCV 데이터 (Lookback Window: 60~200). -* 전처리: MinMaxScaler를 사용하여 0~1 사이로 정규화. -* 모델 아키텍처 (LSTM):

Layer 1: LSTM (64 units, return_sequences=True) + L2 Regularization + Dropout(0.2)
Layer 2: LSTM (32 units) + L2 Regularization + Dropout(0.2)
Output: Dense(1) - 다음 시점의 종가(Close Price) 예측 (Regression)
Optimizer: Adam (Learning Rate: 0.001)
Loss Function: MSE (Mean Squared Error) -* 출력: 다음 캔들의 종가 예측. -* 신호 생성:
예측가 > 현재가 * 1.001 (0.1% 이상 상승 예측) -> UP -> BUY
예측가 < 현재가 * 0.999 (0.1% 이상 하락 예측) -> DOWN -> SELL
신뢰도(Confidence)는 최근 변동성이 낮을수록 높게 책정됩니다. +### 4.2 손절과 익절 (언제 팔까?) +* 익절 (Take Profit): 욕심부리지 않고 미리 정한 목표(예: 5%)에 도달하면 기계적으로 팝니다. +* 손절 (Stop Loss): "내가 틀렸다"는 것을 인정하고, 손실이 커지기 전에(예: -3%) 잘라냅니다. +* 트레일링 스탑 (Trailing Stop): 가격이 오르면 손절 기준도 같이 올립니다.
예: 100원에 사서 손절가를 90원으로 잡았는데, 가격이 120원으로 오르면 손절가를 110원으로 올립니다. 나중에 가격이 떨어져도 110원에 팔리니까 이익을 챙길 수 있습니다.
-### 3.3 대시보드 (dashboard.py) -Streamlit 프레임워크를 사용한 웹 UI입니다. +## 5. 주요 설정값 설명 (config/settings.py)

-* 실시간 데이터 연동: data/*.json 파일을 매번 다시 읽어(Polling) 최신 상태를 보여줍니다. -* 봇 제어: '전략 적용하기', '봇 재시작' 등의 버튼을 누르면 data/command.json 파일을 생성하여 main.py에 비동기적으로 명령을 전달합니다. -* 로그 뷰어: logs/ 폴더의 최신 로그 파일을 읽어 화면에 출력하며, 에러/경고 메시지에 색상 강조(HTML/CSS Injection)를 적용합니다. +이 숫자들을 고치면 로봇의 성격이 바뀝니다.

-| 파라미터 | 설명 | 권장값/범위 | +| 설정 이름 | 설명 | 추천값 | | :--- | :--- | :--- | -| initial_capital | 봇이 운용할 가상의 초기 자본금. 수익률 계산의 기준이 됨. | 실제 잔고보다 약간 적게 설정 | -| max_position_size | 단일 종목 최대 투자 비중 (0.1 = 10%). | 0.1 ~ 0.3 | -| take_profit_percent | 목표 수익률. 도달 시 시장가 매도. | 스캘핑: 0.01~0.03, 스윙: 0.05~0.1 | -| stop_loss_percent | 고정 손절률. 0으로 설정 시 ATR 기반 동적 손절 사용. | 0 (권장) | -| trailing_stop_percent | 고점 대비 하락 시 이익 실현 비율. | 익절률의 1/3 ~ 1/2 수준 | -| k_value | 변동성 돌파 전략의 민감도 계수. | 0.4 (공격적) ~ 0.7 (보수적) | -| atr_multiplier | 동적 손절가 계산 시 ATR의 배수. (손절가 = 현재가 - ATR * N) | 2.0 ~ 3.0 | +| initial_capital | 로봇한테 "너는 이만큼의 돈이 있다고 생각하고 굴려"라고 알려주는 가상의 자본금입니다. | 실제 통장 잔고보다 약간 적게 | +| take_profit_percent | 익절률. "이만큼 벌면 팔아라". 0.05는 5%를 의미합니다. | 단타: 0.02 (2%)스윙: 0.10 (10%) | +| stop_loss_percent | 손절률. "이만큼 잃으면 무조건 팔아라". 0으로 두면 로봇이 시장 상황(ATR)을 보고 알아서 정합니다. | 0 (자동) 또는 0.03 (3%) | +| k_value | 돌파 계수. 변동성 돌파 전략에서 얼마나 빨리 살지 결정합니다. - 낮으면(0.4): 빨리 사지만 가짜 신호에 속을 수 있음. - 높으면(0.7): 늦게 사지만 확실할 때만 삼. | 0.5 ~ 0.6 |
---

## 4. 리스크 관리 시스템 (Risk Management)

`RiskManager` 클래스와 `main.py`의 협력으로 자본을 보호합니다.

### 4.1 자금 관리 (Position Sizing)
*   **터틀 트레이딩 방식**:
    *   `1 Unit Risk = 전체 자본의 1%`
    *   `Stop Distance = ATR * Multiplier (기본 2.0)`
    *   `매수 수량 = (전체 자본 * 0.01) / Stop Distance`
    *   변동성(ATR)이 클수록 매수 수량을 줄여 리스크를 일정하게 유지합니다.

### 4.2 동적 레버리지 (Dynamic Leverage - Binance Futures)
*   선물 거래 시 변동성에 따라 레버리지를 자동 조절합니다.
*   `Leverage = 목표 변동성(2%) / 현재 변동성(ATR%)`
*   변동성이 낮을 때 레버리지를 높이고, 변동성이 높을 때 낮춰 청산 위험을 관리합니다. (최대 10배 제한)

### 4.3 청산 로직 (Exit Logic)
*   **익절 (Take Profit)**: `진입가 * (1 + take_profit_percent)` 도달 시 매도.
*   **손절 (Stop Loss)**:
    *   고정 손절: `진입가 * (1 - stop_loss_percent)`
    *   동적 손절(ATR): `진입가 - (ATR * Multiplier)` (변동성에 따라 유동적)
*   **트레일링 스탑 (Trailing Stop)**:
    *   가격이 상승하면 손절가도 함께 상승시킵니다.
    *   `최고가 대비 trailing_stop_percent` 하락 시 이익 실현.

### 4.4 청산 위험 관리 (Binance Futures)
*   **`_check_liquidation_safety`**: 바이낸스 선물 거래 시, 주기적으로 청산가(Liquidation Price)와 현재가의 거리를 모니터링합니다.
*   **강제 청산**: 청산가까지의 거리가 **20% 미만**으로 좁혀지면(`distance_pct < 0.20`), 즉시 포지션을 시장가로 종료하여 거래소에 의한 강제 청산(Liquidation)을 방지하고 남은 증거금을 보호합니다.

---

## 5. 주요 파라미터 상세 (`config/settings.py`)

| 파라미터 | 설명 | 권장값/범위 |
| :--- | :--- | :--- |
| `initial_capital` | 봇이 운용할 가상의 초기 자본금. 수익률 계산의 기준이 됨. | 실제 잔고보다 약간 적게 설정 |
| `max_position_size` | 단일 종목 최대 투자 비중 (0.1 = 10%). | 0.1 ~ 0.3 |
| `take_profit_percent` | 목표 수익률. 도달 시 시장가 매도. | 스캘핑: 0.01~0.03, 스윙: 0.05~0.1 |
| `stop_loss_percent` | 고정 손절률. 0으로 설정 시 ATR 기반 동적 손절 사용. | 0 (권장) |
| `trailing_stop_percent` | 고점 대비 하락 시 이익 실현 비율. | 익절률의 1/3 ~ 1/2 수준 |
| `k_value` | 변동성 돌파 전략의 민감도 계수. | 0.4 (공격적) ~ 0.7 (보수적) |
| `atr_multiplier` | 동적 손절가 계산 시 ATR의 배수. (손절가 = 현재가 - ATR * N) | 2.0 ~ 3.0 |

---

## 6. 유지보수 및 확장 가이드

### 6.1 새로운 전략 추가 프로세스
1.  **클래스 생성**: `trading/strategy.py`에 `TradingStrategy`를 상속받는 새 클래스(예: `MyNewStrategy`)를 작성합니다.
2.  **로직 구현**: `generate_signal` 메소드를 오버라이딩하여 매수/매도 로직을 구현합니다. `Signal` 객체를 반환해야 합니다.
3.  **팩토리 등록**: `main.py`의 `_initialize_strategies` 메소드와 `run_backtest_current.py`의 `get_strategy` 함수에 새 전략 클래스를 연결하는 코드를 추가합니다.
4.  **설정 추가**: `config/settings.py`의 `STRATEGY_PRESETS`에 새 전략용 파라미터 프리셋을 정의합니다.

### 6.2 백테스팅을 통한 검증
1.  `config/settings.py`에서 테스트할 전략과 타임프레임을 설정합니다.
2.  `run_backtest_current.py`를 실행합니다.
3.  로그에 출력된 승률(Win Rate), 손익비(Profit Factor), 총 수익금(Total Return)을 확인합니다.
4.  `WalkForwardAnalyzer`는 데이터를 학습/테스트 구간으로 나누어 전진 분석하므로, 과적합(Overfitting) 여부를 판단하는 데 유용합니다.

### 6.3 로그 분석 및 디버깅
*   **매수 안 함**: `logs/trading_bot.log`에서 `[HOLD]` 또는 `진입 보류` 메시지를 찾으세요. 필터(ADX, EMA 등) 조건 미달인 경우가 많습니다.
*   **API 오류**: `ERROR` 레벨 로그를 확인하세요. `401 Unauthorized`(키 오류), `429 Too Many Requests`(요청 제한 초과) 등이 기록됩니다.
*   **데이터 부족**: `[WAIT]` 또는 `데이터 부족` 로그는 봇 구동 초기(Warm-up)에 발생하며, 시간이 지나면 해소됩니다.

-## 6. 유지보수 및 확장 가이드 +## 6. 고급 기능 (알아두면 좋은 것들)

-### 6.1 새로운 전략 추가 프로세스 -1. 클래스 생성: trading/strategy.py에 TradingStrategy를 상속받는 새 클래스(예: MyNewStrategy)를 작성합니다. -2. 로직 구현: generate_signal 메소드를 오버라이딩하여 매수/매도 로직을 구현합니다. Signal 객체를 반환해야 합니다. -3. 팩토리 등록: main.py의 _initialize_strategies 메소드와 run_backtest_current.py의 get_strategy 함수에 새 전략 클래스를 연결하는 코드를 추가합니다. -4. 설정 추가: config/settings.py의 STRATEGY_PRESETS에 새 전략용 파라미터 프리셋을 정의합니다. +### 6.1 더스트(Dust) 처리 +* 문제: 코인을 팔 때 수수료 때문에 0.000001개 같은 찌꺼기가 남아서 "잔액 부족"으로 매도가 안 되는 경우가 있습니다. +* 해결: 로봇이 "이건 너무 적은 금액(5000원 미만)이라 못 파네"라고 판단하면, 장부에서 그냥 지워버리거나(업비트), 거래소 기능을 이용해 BNB 코인으로 바꿔버립니다(바이낸스).

-### 6.2 백테스팅을 통한 검증 -1. config/settings.py에서 테스트할 전략과 타임프레임을 설정합니다. -2. run_backtest_current.py를 실행합니다. -3. 로그에 출력된 승률(Win Rate), 손익비(Profit Factor), 총 수익금(Total Return)을 확인합니다. -4. WalkForwardAnalyzer는 데이터를 학습/테스트 구간으로 나누어 전진 분석하므로, 과적합(Overfitting) 여부를 판단하는 데 유용합니다. +### 6.2 OCO 주문 (바이낸스 전용) +* 기능: 매수하자마자 "익절 주문"과 "손절 주문"을 동시에 미리 걸어둡니다. +* 장점: 로봇이 갑자기 꺼지거나 인터넷이 끊겨도, 거래소 서버에 이미 주문이 들어가 있으므로 안전하게 익절/손절이 됩니다.

-### 6.3 로그 분석 및 디버깅 -* 매수 안 함: logs/trading_bot.log에서 [HOLD] 또는 진입 보류 메시지를 찾으세요. 필터(ADX, EMA 등) 조건 미달인 경우가 많습니다. -* API 오류: ERROR 레벨 로그를 확인하세요. 401 Unauthorized(키 오류), 429 Too Many Requests(요청 제한 초과) 등이 기록됩니다. -* 데이터 부족: [WAIT] 또는 데이터 부족 로그는 봇 구동 초기(Warm-up)에 발생하며, 시간이 지나면 해소됩니다.



## 7. 고급 기능 및 안전장치 (Advanced Features & Safety)

시스템 안정성을 위한 특수 기능들입니다.

### 7.1 소액 잔고 처리 (Dust Handling)
*   **문제**: 매매 수수료 공제 후 남은 극소량의 코인(Dust)이 포트폴리오에 남아 매도 주문이 거부되는 현상.
*   **해결**: `_execute_sell` 및 `_sync_portfolio`에서 최소 주문 금액(예: 5,000원, 10 USDT) 미만의 잔고는 'Dust'로 분류하여 포트폴리오에서 자동 제외하거나(Upbit), BNB로 변환을 시도합니다(Binance).

### 7.2 슬리피지 감지 (Slippage Protection)
*   매도 주문 체결 시, 직전 현재가와 실제 체결가를 비교합니다.
*   괴리율이 0.5% 이상 발생하면 경고 로그를 남기고 텔레그램으로 알림을 보내, 시장가 매도의 위험성을 사용자에게 알립니다.

### 7.3 API Fallback 메커니즘
*   웹소켓 연결이 불안정하여 실시간 가격 데이터가 수신되지 않을 경우(`current_price == 0`), 즉시 REST API를 호출하여 최신 가격을 조회합니다. 이를 통해 매매 판단이 멈추지 않도록 보장합니다.

### 7.4 OCO 주문 (One-Cancels-the-Other)
*   **Binance Spot 전용**: 매수 체결 즉시 익절 주문과 손절 주문을 거래소 서버에 동시에 등록합니다.
*   봇이 꺼져도 거래소 서버 차원에서 익절/손절이 집행되므로 안전성이 매우 높습니다.


-## 7. 고급 기능 및 안전장치 (Advanced Features & Safety)
-시스템 안정성을 위한 특수 기능들입니다.
-### 7.1 소액 잔고 처리 (Dust Handling) -* 문제: 매매 수수료 공제 후 남은 극소량의 코인(Dust)이 포트폴리오에 남아 매도 주문이 거부되는 현상. -* 해결: _execute_sell 및 _sync_portfolio에서 최소 주문 금액(예: 5,000원, 10 USDT) 미만의 잔고는 'Dust'로 분류하여 포트폴리오에서 자동 제외하거나(Upbit), BNB로 변환을 시도합니다(Binance).
-### 7.2 슬리피지 감지 (Slippage Protection) -* 매도 주문 체결 시, 직전 현재가와 실제 체결가를 비교합니다. -* 괴리율이 0.5% 이상 발생하면 경고 로그를 남기고 텔레그램으로 알림을 보내, 시장가 매도의 위험성을 사용자에게 알립니다.
-### 7.3 API Fallback 메커니즘 -* 웹소켓 연결이 불안정하여 실시간 가격 데이터가 수신되지 않을 경우(current_price == 0), 즉시 REST API를 호출하여 최신 가격을 조회합니다. 이를 통해 매매 판단이 멈추지 않도록 보장합니다.
-### 7.4 OCO 주문 (One-Cancels-the-Other) -* Binance Spot 전용: 매수 체결 즉시 익절 주문과 손절 주문을 거래소 서버에 동시에 등록합니다. -* 봇이 꺼져도 거래소 서버 차원에서 익절/손절이 집행되므로 안전성이 매우 높습니다. +### 6.3 슬리피지(Slippage) 감지 +* 상황: 내가 "100원에 팔아줘" 했는데, 가격이 급락해서 "98원"에 팔리는 현상입니다. +* 기능: 로봇이 이걸 감지해서 "어? 생각보다 너무 싸게 팔렸는데?" 하고 경고 메시지를 보내줍니다.