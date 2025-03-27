from typing import Dict
import numpy as np
import pandas as pd

class TradingAlgorithm:
    # Core strategy parameters
    THRESHOLDS = {
        "SOBER_expanded": 4,
        "UEC_expanded": .1
    }
    WINDOW_SIZE = 20
    POSITION_SMALL = 1
    POSITION_LARGE = 5
    MULT_NORMAL = 1.0
    MULT_AGGRESSIVE = 1.5

    def __init__(self):
        self.positions: Dict[str, int] = {}
        self.price_data = {}

    def initialize_product(self, product: str) -> None:
        if product not in self.price_data:
            self.price_data[product] = {
                'mid_prices': [],
                'moving_avg': 0
            }

    def getOrders(self, current_data: Dict[str, Dict[str, float]], order_data: Dict[str, int]) -> Dict[str, int]:
        for product in current_data:
            # Initialize if needed
            self.initialize_product(product)
            
            # Update price history
            mid_price = (current_data[product]["Bid"] + current_data[product]["Ask"]) / 2
            self.price_data[product]['mid_prices'].append(mid_price)
            
            # Maintain window size
            if len(self.price_data[product]['mid_prices']) > self.WINDOW_SIZE:
                # Calculate moving average before popping to get proper logic
                self.price_data[product]['moving_avg'] = np.average(self.price_data[product]['mid_prices'])
                
                # CRITICAL: Calculate deviation using the OLDEST price (index 0)
                # This matches the original profitable strategy logic
                deviation = self.price_data[product]['mid_prices'][0] - self.price_data[product]['moving_avg']
                
                # Remove the oldest price AFTER using it for comparison
                self.price_data[product]['mid_prices'].pop(0)

                # Get threshold based on product
                threshold = self.THRESHOLDS.get(product, 4)  # Default to 4 if product not found
                
                # Generate trading signals (identical logic to the profitable version)
                if deviation < -(threshold * self.MULT_AGGRESSIVE):
                    order_data[product] = -self.POSITION_LARGE
                elif deviation < -(threshold * self.MULT_NORMAL):
                    order_data[product] = -self.POSITION_SMALL
                elif deviation > (threshold * self.MULT_AGGRESSIVE):
                    order_data[product] = self.POSITION_LARGE
                elif deviation > (threshold * self.MULT_NORMAL):
                    order_data[product] = self.POSITION_SMALL

        return order_data

# Global instance
team_algorithm = TradingAlgorithm()

def getOrders(current_data: Dict[str, Dict[str, float]], positions: Dict[str, int]) -> Dict[str, int]:
    team_algorithm.positions = positions
    return team_algorithm.getOrders(current_data, {product: 0 for product in current_data})

def printToExcel() -> None:
    """Export trade history to Excel"""
    with pd.ExcelWriter('orders.xlsx') as writer:
        for product in team_algorithm.price_data:
            data = {
                "Timeframe": range(len(team_algorithm.price_data[product]['mid_prices'])),
                "Price": team_algorithm.price_data[product]['mid_prices'],
                "Moving Average": [team_algorithm.price_data[product]['moving_avg']] * len(team_algorithm.price_data[product]['mid_prices'])
            }
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=product, index=False, engine='openpyxl')
    print("Orders stored in 'orders.xlsx' file.\n")
