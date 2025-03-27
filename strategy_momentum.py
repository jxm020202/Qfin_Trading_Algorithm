from typing import Dict
import numpy as np

class TradingAlgorithm:
    def __init__(self):
        self.positions: Dict[str, int] = {}
        self.price_history: Dict[str, list] = {}  # Store price history for each product
        self.lookback = 20  # Number of periods to look back
        self.threshold = 0.05  # Momentum threshold in percentage

    def calculate_momentum(self, prices: list) -> float:
        """Calculate price momentum comparing oldest to newest price."""
        if len(prices) < 2:  # Need at least 2 prices
            return 0
        # Compare the oldest price (index 0) to the newest price (index -1)
        return (prices[-1] - prices[0]) / prices[0] * 100

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
                # Calculate momentum using all prices in the window
                momentum = self.calculate_momentum(self.price_history[product])
                
                # Get current position
                current_position = self.positions.get(product, 0)
                
                # Generate signals
                if momentum > self.threshold and current_position <= 0:
                    # Strong upward momentum - buy signal
                    order_data[product] = 1
                elif momentum < -self.threshold and current_position >= 0:
                    # Strong downward momentum - sell signal
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