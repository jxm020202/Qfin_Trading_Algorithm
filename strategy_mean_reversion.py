from typing import Dict
import numpy as np

class TradingAlgorithm:
    def __init__(self):
        self.positions: Dict[str, int] = {}
        self.price_history: Dict[str, list] = {}  # Store price history for each product
        self.window = 20  # Moving average window
        self.threshold = 0.01  # 1% deviation threshold

    def calculate_moving_average(self, prices: list) -> float:
        """Calculate moving average over the specified window."""
        if len(prices) < self.window:
            return None
        return np.mean(prices)

    def getOrders(self, current_data: Dict[str, Dict[str, float]], order_data: Dict[str, int]) -> Dict[str, int]:
        for product, data in current_data.items():
            # Initialize price history for new products
            if product not in self.price_history:
                self.price_history[product] = []
            
            # Calculate mid price
            mid_price = (data['Bid'] + data['Ask']) / 2
            self.price_history[product].append(mid_price)
            
            # Process when we have enough history
            if len(self.price_history[product]) > self.window:
                # Calculate moving average with the current window
                ma = self.calculate_moving_average(self.price_history[product])
                
                if ma is None:
                    continue
                    
                # Get current position
                current_position = self.positions.get(product, 0)
                
                # IMPORTANT: Use the OLDEST price for comparison (index 0)
                # This matches the profitable approach from basic.py
                oldest_price = self.price_history[product][0]
                
                # Calculate deviation from mean using oldest price
                deviation = (oldest_price - ma) / ma
                
                # Generate signals
                if deviation > self.threshold and current_position >= 0:
                    # Oldest price is significantly above MA - sell signal
                    order_data[product] = -1
                elif deviation < -self.threshold and current_position <= 0:
                    # Oldest price is significantly below MA - buy signal
                    order_data[product] = 1
                
                # Remove the oldest price AFTER using it
                self.price_history[product].pop(0)
                
        return order_data

# Leave this stuff as it is
team_algorithm = TradingAlgorithm()

def getOrders(current_data, positions):
    team_algorithm.positions = positions
    order_data = {product: 0 for product in current_data}
    return team_algorithm.getOrders(current_data, order_data) 