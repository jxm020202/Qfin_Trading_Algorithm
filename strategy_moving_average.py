from typing import Dict
import numpy as np

class TradingAlgorithm:
    def __init__(self):
        self.positions: Dict[str, int] = {}
        self.price_history: Dict[str, list] = {}  # Store price history for each product
        self.short_window = 20
        self.long_window = 50
        self.buffer = 0.001  # 0.1% buffer to avoid whipsaws

    def calculate_moving_averages(self, prices: list) -> tuple:
        """Calculate short-term and long-term moving averages."""
        if len(prices) < self.long_window:
            return None, None
        
        # For the short MA, we use the most recent prices
        short_ma = np.mean(prices[-self.short_window:])
        # For the long MA, we use all prices
        long_ma = np.mean(prices)
        return short_ma, long_ma

    def getOrders(self, current_data: Dict[str, Dict[str, float]], order_data: Dict[str, int]) -> Dict[str, int]:
        for product, data in current_data.items():
            # Initialize price history for new products
            if product not in self.price_history:
                self.price_history[product] = []
            
            # Calculate mid price
            mid_price = (data['Bid'] + data['Ask']) / 2
            self.price_history[product].append(mid_price)
            
            # Process when we have enough history
            if len(self.price_history[product]) > self.long_window:
                # Calculate moving averages
                short_ma, long_ma = self.calculate_moving_averages(self.price_history[product])
                
                if short_ma is None or long_ma is None:
                    continue
                    
                # Get current position
                current_position = self.positions.get(product, 0)
                
                # Generate signals
                if short_ma > long_ma * (1 + self.buffer) and current_position <= 0:
                    # Buy signal - short-term average above long-term
                    order_data[product] = 1
                elif short_ma < long_ma * (1 - self.buffer) and current_position >= 0:
                    # Sell signal - short-term average below long-term
                    order_data[product] = -1
                
                # Remove the oldest price after processing
                self.price_history[product].pop(0)
                
        return order_data

# Leave this stuff as it is
team_algorithm = TradingAlgorithm()

def getOrders(current_data, positions):
    team_algorithm.positions = positions
    order_data = {product: 0 for product in current_data}
    return team_algorithm.getOrders(current_data, order_data) 