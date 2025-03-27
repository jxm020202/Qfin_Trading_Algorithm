from typing import Dict
import numpy as np

class TradingAlgorithm:
    def __init__(self):
        self.positions: Dict[str, int] = {}
        self.price_history: Dict[str, list] = {}
        
        # Adjusted parameters
        self.ma_fast = 5         # Faster MA to catch moves quicker
        self.ma_slow = 15        # Shorter slow MA
        self.position_size = 5    # Smaller initial position
        self.min_trend = 0.001   # 0.1% minimum trend strength - less conservative
        self.cooldown = 5        # Shorter cooldown
        
        # State tracking
        self.last_trade: Dict[str, int] = {}
        self.last_signal: Dict[str, int] = {}

    def calculate_moving_averages(self, prices: list) -> tuple:
        """Calculate moving averages."""
        if len(prices) < self.ma_slow:
            return None, None
        
        # Calculate moving averages from the entire window    
        fast_ma = np.mean(prices[-self.ma_fast:])  # Recent prices for fast MA
        slow_ma = np.mean(prices)  # All prices for slow MA
        return fast_ma, slow_ma

    def getOrders(self, current_data: Dict[str, Dict[str, float]], order_data: Dict[str, int]) -> Dict[str, int]:
        for product, data in current_data.items():
            # Initialize tracking for new products
            if product not in self.price_history:
                self.price_history[product] = []
                self.last_trade[product] = -self.cooldown
                self.last_signal[product] = 0
            
            # Calculate mid price and update history
            mid_price = (data['Bid'] + data['Ask']) / 2
            self.price_history[product].append(mid_price)
            
            # Process when we have enough history
            if len(self.price_history[product]) > self.ma_slow * 2:
                # Get current position and timestamp
                current_position = self.positions.get(product, 0)
                current_timestamp = data.get('Timestamp', 0)
                
                # Honor cooldown period
                if current_timestamp - self.last_trade[product] < self.cooldown:
                    # Remove oldest price and continue
                    self.price_history[product].pop(0)
                    continue
                
                # Calculate MAs
                fast_ma, slow_ma = self.calculate_moving_averages(self.price_history[product])
                if fast_ma is None or slow_ma is None:
                    # Remove oldest price and continue
                    self.price_history[product].pop(0)
                    continue
                
                # Calculate trend strength
                trend_strength = (fast_ma - slow_ma) / slow_ma
                
                # Generate signal
                signal = 0
                
                # Only trade if trend is strong enough
                if abs(trend_strength) > self.min_trend:
                    if trend_strength > 0 and current_position <= 0:  # Strong uptrend
                        signal = self.position_size
                    elif trend_strength < 0 and current_position >= 0:  # Strong downtrend
                        signal = -self.position_size
                
                # Only trade if it's a new signal direction
                if signal != 0 and signal == self.last_signal[product]:
                    signal = 0
                
                # Apply the signal
                if signal != 0:
                    order_data[product] = signal
                    self.last_trade[product] = current_timestamp
                    self.last_signal[product] = signal
                
                # Remove the oldest price after processing
                self.price_history[product].pop(0)
                
        return order_data

# Leave this stuff as it is
team_algorithm = TradingAlgorithm()

def getOrders(current_data, positions):
    team_algorithm.positions = positions
    order_data = {product: 0 for product in current_data}
    return team_algorithm.getOrders(current_data, order_data) 