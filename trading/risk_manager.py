import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class RiskManager:
    """위험 관리"""
    
    def __init__(self, take_profit_percent: float = 0.15,
                 atr_multiplier: float = 2.0,
                 trailing_stop_percent: float = 0.02):
        self.take_profit_percent = take_profit_percent
        self.atr_multiplier = atr_multiplier
        self.trailing_stop_percent = trailing_stop_percent
        self.stop_loss_prices: Dict[str, float] = {}
        self.take_profit_prices: Dict[str, float] = {}
        self.highest_prices: Dict[str, float] = {}
        self.atr_values: Dict[str, float] = {}  # 진입 시점의 ATR 저장
        self.entry_prices: Dict[str, float] = {} # 진입가 저장 (Break-even용)
    
    def set_stop_loss(self, symbol: str, entry_price: float, atr_value: float = 0.0, custom_stop_loss: Optional[float] = None) -> float:
        """손실 제한 가격 설정 (터틀 트레이딩 2N 룰 적용)"""
        self.entry_prices[symbol] = entry_price
        
        # ATR 저장 (트레일링 스탑용)
        if atr_value > 0:
            self.atr_values[symbol] = atr_value

        if custom_stop_loss is None:
            if atr_value > 0:
                # ATR 기반 2N 손절가 자동 계산
                stop_loss_price = entry_price - (atr_value * self.atr_multiplier)
            else:
                logger.warning(f"⚠️ {symbol} ATR 정보 없음. 비상 손절(-5%)을 적용합니다.")
                stop_loss_price = entry_price * 0.95
        else:
            stop_loss_price = custom_stop_loss

        implied_pct = (entry_price - stop_loss_price) / entry_price * 100 if entry_price > 0 else 0
        logger.info(
            f"{symbol} ATR 가변 손절 설정: {entry_price:,.0f} → {stop_loss_price:,.0f} "
            f"(-{implied_pct:.2f}%)"
        )
            
        self.stop_loss_prices[symbol] = stop_loss_price
        
        # 트레일링 스탑 초기화 (진입가를 초기 최고가로 설정)
        self.highest_prices[symbol] = entry_price
        return stop_loss_price
    
    def set_take_profit(self, symbol: str, entry_price: float, fee_rate: float = 0.0, atr_value: float = 0.0) -> float:
        """수익 실현 가격 설정 (수수료 고려)"""
        target_pct = self.take_profit_percent

        # [New] 동적 익절 로직 (ATR 기반 가변 익절)
        if atr_value > 0 and entry_price > 0:
            # 변동성이 크면 익절 목표 상향 (3 * ATR 기준 - 수익 극대화)
            dynamic_pct = (atr_value * 3.0) / entry_price
            
            # 최소 0.5% ~ 최대 20% 범위 내에서 조정
            dynamic_pct = max(0.005, min(dynamic_pct, 0.20))
            
            logger.info(f"⚖️ [Dynamic TP] {symbol} 변동성(ATR) 반영: 기본 {target_pct*100:.1f}% -> 조정 {dynamic_pct*100:.1f}%")
            target_pct = dynamic_pct

        # 목표 수익률에 수수료율을 더해서 목표가 상향 조정
        take_profit_price = entry_price * (1 + target_pct + fee_rate)
        self.take_profit_prices[symbol] = take_profit_price
        logger.info(
            f"{symbol} 수익 실현 설정: {entry_price} → {take_profit_price} "
            f"({target_pct*100:.1f}%)"
        )
        return take_profit_price
    
    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """손실 제한 확인"""
        if symbol not in self.stop_loss_prices:
            return False
        
        # [Fix] 손절가가 0이거나 음수면 체크 스킵 (유효하지 않은 설정)
        if self.stop_loss_prices[symbol] <= 0:
            return False
        
        if current_price <= self.stop_loss_prices[symbol]:
            logger.warning(
                f"{symbol} 손실 제한 도달: {current_price} "
                f"<= {self.stop_loss_prices[symbol]}"
            )
            return True
        
        return False
    
    def check_take_profit(self, symbol: str, current_price: float) -> bool:
        """수익 실현 확인"""
        if symbol not in self.take_profit_prices:
            return False
        
        # [터틀 트레이딩] 추세가 강해 트레일링 스탑 라인이 익절가를 넘어선 경우,
        # 익절을 보류하고 트레일링 스탑을 따라가며 수익을 극대화함
        if symbol in self.highest_prices and symbol in self.atr_values:
            trailing_stop_price = self.highest_prices[symbol] - (self.atr_values[symbol] * self.atr_multiplier)
            if trailing_stop_price > self.take_profit_prices[symbol]:
                # 로그는 너무 자주 찍히지 않게 디버그 레벨이나 조건부로 처리하는 것이 좋음
                # logger.debug(f"{symbol} 강한 추세: 트레일링 스탑이 익절가 상회. 익절 보류.")
                return False

        # [Fix] 익절가가 0이거나 음수면 체크 스킵 (유효하지 않은 설정)
        if self.take_profit_prices[symbol] <= 0:
            return False

        if current_price >= self.take_profit_prices[symbol]:
            logger.warning(
                f"{symbol} 수익 실현 도달: {current_price} "
                f">= {self.take_profit_prices[symbol]}"
            )
            return True
        
        return False
    
    def check_trailing_stop(self, symbol: str, current_price: float) -> bool:
        """트레일링 스탑 확인 및 손절가 상향"""
        if symbol not in self.highest_prices or symbol not in self.atr_values:
            return False
        
        # 최고가 갱신 (현재가가 더 높으면 갱신하고 트레일링 스탑 미발동)
        if current_price > self.highest_prices[symbol]:
            self.highest_prices[symbol] = current_price
            
            # [로그 상세화] 손절가 상향 조정 (Ratcheting)
            if self.trailing_stop_percent > 0:
                new_stop_price = self.highest_prices[symbol] * (1 - self.trailing_stop_percent)
            else:
                new_stop_price = self.highest_prices[symbol] - (self.atr_values[symbol] * self.atr_multiplier)
            
            current_sl = self.stop_loss_prices.get(symbol, 0)
            # 기존 손절가보다 높을 때만 업데이트 (상향 조정)
            if new_stop_price > current_sl:
                self.stop_loss_prices[symbol] = new_stop_price
                logger.info(f"🛡️ 보호선 상향: {symbol} {current_sl:,.0f}원 -> {new_stop_price:,.0f}원 (최고가 갱신)")
            
            return False
            
        # 트레일링 스탑 가격 계산
        # 설정된 퍼센트가 있으면 우선 사용, 없으면 ATR 기반 사용
        if self.trailing_stop_percent > 0:
            trailing_stop_price = self.highest_prices[symbol] * (1 - self.trailing_stop_percent)
        else:
            trailing_stop_price = self.highest_prices[symbol] - (self.atr_values[symbol] * self.atr_multiplier)
        
        if current_price <= trailing_stop_price:
            logger.warning(
                f"{symbol} 트레일링 스탑 도달: {current_price} "
                f"<= {trailing_stop_price:.0f} (최고가: {self.highest_prices[symbol]})"
            )
            return True
        
        return False
    
    def check_exit_conditions(self, symbol: str, current_price: float) -> Optional[str]:
        """
        매도 조건 확인
        
        Returns:
            "stop_loss", "take_profit", "trailing_stop", 또는 None
        """
        # 0. Break-even (본절 보존) 로직
        # 수익률이 +4% 이상이고, 현재 손절가가 본절+0.5%보다 낮으면 손절가 상향
        if symbol in self.entry_prices and symbol in self.stop_loss_prices:
            entry_price = self.entry_prices[symbol]
            profit_pct = (current_price - entry_price) / entry_price
            target_sl = entry_price * 1.005
            
            if profit_pct >= 0.04 and self.stop_loss_prices[symbol] < target_sl:
                self.stop_loss_prices[symbol] = target_sl
                logger.info(f"🛡️ {symbol} 수익 4% 도달: 손절가를 본절+0.5%({target_sl:,.0f})로 상향 (Break-even)")

        # 1. 손절(Stop Loss) 최우선 체크 (자산 보호)
        if self.check_stop_loss(symbol, current_price):
            return "stop_loss"
        elif self.check_take_profit(symbol, current_price):
            return "take_profit"
        elif self.check_trailing_stop(symbol, current_price):
            return "trailing_stop"
        else:
            return None
    
    def remove_position(self, symbol: str):
        """포지션 정보 제거"""
        if symbol in self.stop_loss_prices:
            del self.stop_loss_prices[symbol]
        if symbol in self.take_profit_prices:
            del self.take_profit_prices[symbol]
        if symbol in self.highest_prices:
            del self.highest_prices[symbol]
        if symbol in self.atr_values:
            del self.atr_values[symbol]
        if symbol in self.entry_prices:
            del self.entry_prices[symbol]

    def calculate_volatility_index(self, atr_current: float, atr_avg: float) -> float:
        """변동성 지표(Volatility Index) 계산"""
        if atr_avg <= 0:
            return 1.0
        return atr_current / atr_avg

    def get_dynamic_leverage(self, symbol: str, atr_current: float, atr_avg: float, 
                             base_leverage: int, max_leverage_limit: int, 
                             current_price: float, prev_close: float) -> int:
        """
        동적 레버리지 계산 (Inverse Volatility Scaling)
        - Volatility_Index > 1.5: 50% 축소
        - 0.8 <= Index <= 1.2: 유지
        - Index < 0.7: 150% 확대 (최대 10배)
        - Panic Mode: 급락 시 1배
        """
        # 1. Panic Mode (Flash Crash 감지: 전봉 대비 5% 이상 하락)
        if prev_close > 0 and (prev_close - current_price) / prev_close >= 0.05:
            logger.warning(f"🚨 [PANIC] {symbol} 급락 감지(Flash Crash)! 레버리지를 1배로 고정합니다.")
            return 1

        vol_index = self.calculate_volatility_index(atr_current, atr_avg)
        new_leverage = base_leverage

        if vol_index > 1.5:
            new_leverage = int(base_leverage * 0.5)
            logger.info(f"📉 [Risk] 고변동성(Idx:{vol_index:.2f}) -> 레버리지 축소 ({base_leverage}x -> {new_leverage}x)")
        elif vol_index < 0.7:
            new_leverage = int(base_leverage * 1.5)
            new_leverage = min(new_leverage, 10) # 알고리즘상 최대 10배 제한
            logger.info(f"📈 [Risk] 저변동성(Idx:{vol_index:.2f}) -> 레버리지 확대 ({base_leverage}x -> {new_leverage}x)")
        
        # Safety Rail: Hard Cap
        if new_leverage > max_leverage_limit:
            logger.warning(f"⚠️ [Safety] 계산된 레버리지({new_leverage}x)가 한도({max_leverage_limit}x)를 초과하여 조정합니다.")
            new_leverage = max_leverage_limit
            
        return max(1, new_leverage)
