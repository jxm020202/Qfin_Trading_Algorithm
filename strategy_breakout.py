from typing import Dict
import numpy as np

class TradingAlgorithm:
    def __init__(self):
        self.positions: Dict[str, int] = {}
        self.price_history: Dict[str, list] = {}  # Store price history for each product
        self.lookback = 10  # Number of periods to look back

    def calculate_high_low(self, prices: list) -> tuple:
        """Calculate highest and lowest prices over entire window."""
        if len(prices) < self.lookback:
            return None, None
        return max(prices), min(prices)

    def getOrders(self, current_data: Dict[str, Dict[str, float]], order_data: Dict[str, int]) -> Dict[str, int]:
        for product, data in current_data.items():
            # Initialize price history for new products
            if product not in self.price_history:
                self.price_history[product] = []
            
            # Calculate mid price
            mid_price = (data['Bid'] + data['Ask']) / 2
            self.price_history[product].append(mid_price)
            
            # Process when we have enough history
            if len(self.price_history[product]) > self.lookback:
                # Calculate high and low over the entire window
                recent_high, recent_low = self.calculate_high_low(self.price_history[product])
                
                if recent_high is None or recent_low is None:
                    # Remove oldest price and continue
                    self.price_history[product].pop(0)
                    continue
                    
                # Get current position
                current_position = self.positions.get(product, 0)
                
                # Generate signals using current price
                if mid_price > recent_high * 0.99 and current_position <= 0:
                    # Price is near or above recent high - buy signal
                    order_data[product] = 1
                elif mid_price < recent_low * 1.01 and current_position >= 0:
                    # Price is near or below recent low - sell signal
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