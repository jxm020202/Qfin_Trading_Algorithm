import pandas as pd
import main as template
from copy import deepcopy

DATA_LOCATION = "Data/Small"

def run_optimization_backtest(products: list) -> float:
    """
    Streamlined backtester for optimization that only returns PnL.
    No printing or detailed tracking, just the final PnL.
    """
    # Initialize data structures
    price_series = {}
    positions = {}
    cash = {}
    
    # Load data and initialize tracking
    for product in products:
        price_series[product] = pd.read_csv(f"{DATA_LOCATION}/{product}.csv")
        positions[product] = 0
        cash[product] = 0
    
    # Set constants
    position_limit = 100
    fees = 0.002
    n_timestamps = len(price_series[products[0]])
    
    # Main trading loop
    for i in range(n_timestamps):
        # Prepare current market data
        current_data = {}
        for product in products:
            current_data[product] = {
                "Timestamp": i,
                "Bid": price_series[product].iloc[i]["Bids"],
                "Ask": price_series[product].iloc[i]["Asks"]
            }
        
        # Get orders from strategy
        order = template.getOrders(deepcopy(current_data), deepcopy(positions))
        
        # Process orders
        for product in order:
            quant = int(order[product])
            if quant == 0:
                continue
            
            if quant > 0:  # Buying
                if positions[product] + quant > position_limit:
                    quant = 0
                if quant > 0:
                    cash[product] -= current_data[product]["Ask"] * quant * (1 + fees)
            
            elif quant < 0:  # Selling
                if positions[product] + quant < -position_limit:
                    quant = 0
                if quant < 0:
                    cash[product] += current_data[product]["Bid"] * -quant * (1 - fees)
            
            positions[product] += quant
    
    # Close positions at the end
    cash_sum = 0
    for product in products:
        if positions[product] > 0:
            cash[product] += price_series[product].iloc[-1]["Bids"] * positions[product] * (1 - fees)
        elif positions[product] < 0:
            cash[product] -= price_series[product].iloc[-1]["Asks"] * -positions[product] * (1 + fees)
        cash_sum += cash[product]
    
    return cash_sum 