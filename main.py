from typing import Dict
import numpy as np
import pandas as pd

class TradingAlgorithm:
    def __init__(self):
        self.positions: Dict[str, int] = {}  # Tracks positions per asset
        self.price_data = {}  # For each asset, store a dict with 'mid_prices', 'moving_avg'
        self.last_trade = {}  # Track last trade time to prevent overtrading
        
        # Asset-specific parameters
        self.THRESHOLDS = {
            "UEC": .7,  # Lower threshold for UEC due to lower volatility
            "SOBER": 3  # Higher threshold for SOBER due to higher volatility
        }
        self.WINDOW_SIZE = {
            "UEC": 40,
            "SOBER": 20
        }
        # Minimum time between trades (in timestamps)
        self.MIN_TRADE_INTERVAL = {
            "UEC": 15,
            "SOBER": 10
        }
        # Trade sizes
        self.POSITION_SMALL = {
            "UEC": 3,
            "SOBER": 3
        }
        self.POSITION_LARGE = {
            "UEC": 18,
            "SOBER": 8
        }

    def initialize_product(self, product: str) -> None:
        if product not in self.price_data:
            self.price_data[product] = {
                'mid_prices': [],
                'moving_avg': None
            }
            self.last_trade[product] = -float('inf')
    
    def update_indicators(self, product: str) -> None:
        window = self.WINDOW_SIZE.get(product, 20)
        prices = self.price_data[product]['mid_prices']
        if len(prices) >= window:
            self.price_data[product]['moving_avg'] = np.mean(prices)
    
    def getOrders(self, current_data: Dict[str, Dict[str, float]], order_data: Dict[str, int]) -> Dict[str, int]:
        for product, data in current_data.items():
            # Initialize if needed
            self.initialize_product(product)
            
            # Compute mid-price
            mid_price = (data["Bid"] + data["Ask"]) / 2.0
            self.price_data[product]['mid_prices'].append(mid_price)
            
            # Get current timestamp
            current_time = data["Timestamp"]
            
            # Check if enough time has passed since last trade
            if current_time - self.last_trade[product] < self.MIN_TRADE_INTERVAL.get(product, 5):
                order_data[product] = 0
                continue
            
            window = self.WINDOW_SIZE.get(product, 20)
            
            if len(self.price_data[product]['mid_prices']) >= window:
                self.update_indicators(product)
                moving_avg = self.price_data[product]['moving_avg']
                
                # Calculate deviation using the OLDEST price
                oldest_price = self.price_data[product]['mid_prices'][0]
                deviation = oldest_price - moving_avg
                
                threshold = self.THRESHOLDS.get(product, 1.0)
                current_position = self.positions.get(product, 0)
                
                # FIXED: Corrected signal direction for mean reversion
                # If oldest price is ABOVE moving average, price likely to fall (SELL)
                # If oldest price is BELOW moving average, price likely to rise (BUY)
                if deviation > threshold:  # Price above MA - likely to fall
                    if deviation > threshold * 1.5:
                        order_data[product] = -self.POSITION_LARGE.get(product, 2)
                    else:
                        order_data[product] = -self.POSITION_SMALL.get(product, 1)
                    self.last_trade[product] = current_time
                elif deviation < -threshold:  # Price below MA - likely to rise
                    if deviation < -threshold * 1.5:
                        order_data[product] = self.POSITION_LARGE.get(product, 2)
                    else:
                        order_data[product] = self.POSITION_SMALL.get(product, 1)
                    self.last_trade[product] = current_time
                else:
                    order_data[product] = 0
                
                # Remove the oldest price after using it
                self.price_data[product]['mid_prices'].pop(0)
            else:
                order_data[product] = 0
        
        return order_data

# Global instance
team_algorithm = TradingAlgorithm()

def getOrders(current_data: Dict[str, Dict[str, float]], positions: Dict[str, int]) -> Dict[str, int]:
    team_algorithm.positions = positions
    return team_algorithm.getOrders(current_data, {product: 0 for product in current_data})


