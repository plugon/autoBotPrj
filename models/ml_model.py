import numpy as np
import pandas as pd
import logging
import os
import joblib
import warnings
import matplotlib.pyplot as plt
import requests
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ML_CONFIG

# TensorFlow 로그 레벨 조정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    from tensorflow.keras.regularizers import l2
    HAS_TF = True
except ImportError:
    HAS_TF = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

class MLPredictor:
    def __init__(self, lookback_window=60, model_type="random_forest"):
        self.lookback_window = lookback_window
        self.model_type = model_type
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
    def _build_lstm_model(self, input_shape):
        """
        [Request] 시계열 예측에 최적화된 고성능 LSTM 모델 구축
        1. 다층 LSTM 구조 (Stacked LSTM)
        2. 규제화 (Dropout + L2 Regularization)
        3. 학습 최적화 (Adam + MSE)
        4. 데이터 쉐이프 자동 조정 (input_shape 인자 활용)
        """
        if not HAS_TF:
            logger.error("TensorFlow not installed.")
            return None

        model = Sequential()
        
        # 1. 다층 LSTM 구조 (Stacked LSTM) - 첫 번째 레이어
        # return_sequences=True: 다음 LSTM 레이어로 시퀀스 전달
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape,
                       kernel_regularizer=l2(0.001))) # 2. L2 규제화
        model.add(Dropout(0.2)) # 2. Dropout
        
        # 두 번째 LSTM 레이어
        model.add(LSTM(32, return_sequences=False,
                       kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.2))
        
        # 출력층 (다음 종가 예측 - Regression)
        model.add(Dense(1))
        
        # 3. 학습 최적화 설정
        # Optimizer: Adam (lr=0.001)
        # Loss: MSE (Mean Squared Error)
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse')
        
        return model

    def prepare_data(self, data):
        """데이터 전처리 (Scaling + Windowing)"""
        if len(data) < self.lookback_window + 1:
            return None, None

        # 종가 기준 예측
        close_prices = data['close'].values.reshape(-1, 1)
        
        # Scaling
        if not self.is_trained:
            scaled_data = self.scaler.fit_transform(close_prices)
        else:
            scaled_data = self.scaler.transform(close_prices)
        
        X, y = [], []
        for i in range(self.lookback_window, len(scaled_data)):
            X.append(scaled_data[i-self.lookback_window:i, 0])
            y.append(scaled_data[i, 0])
            
        X, y = np.array(X), np.array(y)
        
        if self.model_type == "lstm":
            # LSTM input shape: (samples, time steps, features)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
        return X, y

    def train(self, data, epochs=5, batch_size=64, **kwargs):
        """모델 학습"""
        try:
            X, y = self.prepare_data(data)
            if X is None or len(X) == 0:
                logger.warning("학습 데이터 부족 ")
                return

            if self.model_type == "lstm":
                if not HAS_TF:
                    logger.error("TensorFlow 미설치로 LSTM 학습 불가")
                    return
                logger.error("TensorFlow로 학     습시작")
                # 4. 데이터 쉐이프 자동 조정
                input_shape = (X.shape[1], 1)
                self.model = self._build_lstm_model(input_shape)
                
                # [New] 모델 구조 요약 로그 출력
                self.model.summary(print_fn=logger.info)
                
                # 3. 학습 최적화 설정 (ReduceLROnPlateau)
                # 학습 정체 시 학습률 자동 감소
                reduce_lr = ReduceLROnPlateau(
                    monitor='loss', 
                    factor=0.5, 
                    patience=5, 
                    min_lr=0.00001,
                    verbose=0
                )

                # [New] EarlyStopping (과적합 방지)
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=3, # [Request] 학습 정체 시 즉시 중단 (10 -> 3)
                    restore_best_weights=True,
                    verbose=0
                )
                
                val_split = ML_CONFIG.get("validation_ratio", 0.1)
                
                self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, 
                               validation_split=val_split, callbacks=[reduce_lr, early_stopping])
                
            else: # random_forest (Classifier)
                logger.error("Trandom_forest로 학습시작")
                # RF는 방향성(0/1) 예측으로 변환 필요
                y_class = (y > X[:, -1]).astype(int)
                
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
                self.model.fit(X, y_class)
                
            self.is_trained = True
            
            # [New] 학습 후 평가 실행
            self.evaluate(data)
            
            # [New] 시각화 및 텔레그램 전송 (LSTM인 경우)
            if self.model_type == "lstm":
                self._visualize_and_send(X, y)
            
        except Exception as e:
            logger.error(f"모델 학습 중 오류: {e}")

    def evaluate(self, data):
        """[New] 모델 성능 평가 (Loss, Accuracy)"""
        if not self.is_trained or self.model is None:
            return

        try:
            X, y = self.prepare_data(data)
            if X is None or len(X) == 0:
                return

            if self.model_type == "lstm":
                # 1. Loss (MSE)
                loss = self.model.evaluate(X, y, verbose=0)
                
                # 2. Directional Accuracy (방향성 정확도)
                # 예측값 생성
                pred_scaled = self.model.predict(X, verbose=0)
                
                # 직전 가격 (X의 마지막 스텝 값)
                prev_prices = X[:, -1, 0]
                
                # 실제 방향: y > prev_prices
                actual_dir = (y > prev_prices).astype(int)
                # 예측 방향: pred > prev_prices
                pred_dir = (pred_scaled.flatten() > prev_prices).astype(int)
                
                accuracy = np.mean(actual_dir == pred_dir)
                
                logger.info(f"📊 [LSTM 평가] Loss(MSE): {loss:.6f} | 방향성 정확도: {accuracy*100:.2f}%")
                
            else: # random_forest
                y_class = (y > X[:, -1]).astype(int)
                accuracy = self.model.score(X, y_class)
                logger.info(f"📊 [RF 평가] Accuracy: {accuracy*100:.2f}%")
                
        except Exception as e:
            logger.error(f"모델 평가 중 오류: {e}")

    def _visualize_and_send(self, X, y):
        """[New] 학습 결과 시각화 및 텔레그램 전송"""
        try:
            # Headless 모드 설정 (서버 환경 호환)
            plt.switch_backend('Agg')
            
            # 예측
            pred_scaled = self.model.predict(X, verbose=0)
            
            # 역변환 (스케일링 복구)
            real_y = self.scaler.inverse_transform(y.reshape(-1, 1))
            real_pred = self.scaler.inverse_transform(pred_scaled)
            
            # 그래프 생성
            plt.figure(figsize=(10, 5))
            
            # 최근 100개만 표시 (가독성)
            display_len = min(len(real_y), 100)
            
            plt.plot(real_y[-display_len:], label='Actual', color='blue', alpha=0.6)
            plt.plot(real_pred[-display_len:], label='Predicted', color='red', linestyle='--', alpha=0.8)
            
            plt.title('LSTM Model Prediction (Recent 100)')
            plt.xlabel('Time Steps')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 이미지 저장
            save_dir = "data/plots"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            save_path = os.path.join(save_dir, "lstm_prediction.png")
            plt.savefig(save_path)
            plt.close()
            
            # 텔레그램 전송
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
                with open(save_path, 'rb') as f:
                    files = {'photo': f}
                    data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': '📊 LSTM 학습 결과 (Actual vs Predicted)'}
                    requests.post(url, data=data, files=files, timeout=5)
                    
        except Exception as e:
            logger.error(f"시각화 전송 오류: {e}")

    def predict_direction(self, data, current_price):
        """다음 캔들 방향 예측"""
        if not self.is_trained or self.model is None:
            return "HOLD"
            
        try:
            # 데이터 준비 (마지막 윈도우)
            if len(data) < self.lookback_window:
                return "HOLD"
                
            close_prices = data['close'].values.reshape(-1, 1)
            scaled_data = self.scaler.transform(close_prices)
            
            last_window = scaled_data[-self.lookback_window:].reshape(1, -1)
            
            if self.model_type == "lstm":
                last_window = np.reshape(last_window, (1, self.lookback_window, 1))
                predicted_scaled = self.model.predict(last_window, verbose=0)[0][0]
                predicted_price = self.scaler.inverse_transform([[predicted_scaled]])[0][0]
                
                # 0.1% 이상 변동 시 방향성 제시
                if predicted_price > current_price * 1.001:
                    return "UP"
                elif predicted_price < current_price * 0.999:
                    return "DOWN"
                else:
                    return "HOLD"
            else:
                prediction = self.model.predict(last_window)[0]
                return "UP" if prediction == 1 else "DOWN"
                
        except Exception as e:
            logger.error(f"예측 중 오류: {e}")
            return "HOLD"

    def save_model(self, path):
        if not self.is_trained: return
        try:
            if self.model_type == "lstm":
                self.model.save(path.replace(".pkl", ".h5"))
                joblib.dump(self.scaler, path.replace(".pkl", "_scaler.pkl"))
            else:
                joblib.dump(self.model, path)
                joblib.dump(self.scaler, path.replace(".pkl", "_scaler.pkl"))
        except Exception as e:
            logger.error(f"모델 저장 오류: {e}")