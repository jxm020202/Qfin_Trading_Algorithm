import pandas as pd
# To use the breakout strategy:
#import strategy_breakout as template

# To use the moving average strategy:
#import strategy_moving_average as template

# To use the mean reversion strategy:
#import strategy_mean_reversion as template

# To use the momentum strategy:
#import strategy_momentum as template

# To use the hybrid UEC optimized strategy:
import main as template

from copy import deepcopy
from datetime import datetime

DATA_LOCATION = "Data" # Set this to the location of your data folder

# List of stock names (without .csv)
products = ["UEC", "SOBER"] # This list determiens the stocks your algorithm will be backtested on

# Dictionary to store the price series dataframes for each stock
price_series = {} # In the form {"ABC": Dataframe, ...}

# Dictionary to keep track of the current positions on each stock
positions = {} # In the form {"ABC": position, ...}

# Dictionary to keep track of the current cash on each stock
cash = {}

# Analytics tracking
trade_history = {}  # Track all trades
for product in products:
    trade_history[product] = []

# Populate these dictionaries
for product in products:
    price_series[product] = pd.read_csv(f"{DATA_LOCATION}/{product}.csv") 
    # price_series[product] = pd.read_csv(f"{DATA_LOCATION}/{product}_fut1.csv") 
    positions[product] = 0
    cash[product] = 0

# Set constants
position_limit = 100
fees = 0.002

# Find the number of total timestamps (Should evaluate to 360 * 30)
n_timestamps = len(price_series[products[0]])

# Process the trades your algo would make on each timestamp
for i in range(n_timestamps):
    # Dictionary that is submitted to your getOrders() function with the current timestamp, best bid and best ask
    current_data = {}

    # Loop through each product to populate current_data dictionary
    for product in products:
        current_data[product] = {"Timestamp": i, "Bid": price_series[product].iloc[i]["Bids"], "Ask": price_series[product].iloc[i]["Asks"]}

    # Send this data to your algo's getOrders() function
    order = template.getOrders(deepcopy(current_data), deepcopy(positions))

    # Loop through all the products that your algo submitted
    for product in order:
        # Find the quantity for this product
        quant = int(order[product])

        # If the order quantity is 0, we do not have to process it
        if quant == 0:
            continue

        # Process buys and sells
        if quant > 0: # Team is buying
            # If sent buy quantity exceeds position limit, adjust the quantity to fit within limits
            if positions[product] + quant > position_limit:
                # print("Attemtped to buy past position limit") 
                quant = 0

            if quant > 0:  # Only record if we actually made a trade
                # Record trade
                trade_history[product].append({
                    'timestamp': i,
                    'type': 'BUY',
                    'quantity': quant,
                    'price': current_data[product]["Ask"],
                    'cost': current_data[product]["Ask"] * quant * (1 + fees),
                    'position_after': positions[product] + quant
                })
                # Change cash for this product
                cash[product] -= current_data[product]["Ask"] * quant * (1 + fees)

        elif quant < 0: # Team is selling
            # If sent sell quantity exceeds position limit, adjust the quantity to fit within limits
            if positions[product] + quant < -position_limit:
                # print("Attemtped to sell past position limit") 
                quant = 0

            if quant < 0:  # Only record if we actually made a trade
                # Record trade
                trade_history[product].append({
                    'timestamp': i,
                    'type': 'SELL',
                    'quantity': abs(quant),
                    'price': current_data[product]["Bid"],
                    'revenue': current_data[product]["Bid"] * abs(quant) * (1 - fees),
                    'position_after': positions[product] + quant
                })
                # Change cash for this product
                cash[product] += current_data[product]["Bid"] * -quant * (1 - fees)
        
        # Modify your algo's position for this product
        positions[product] += quant

# Close any open positions at the end of the algorithm
cash_sum = 0
for product in products:
    print(f"\n=== Analytics for {product} ===")
    print(f"Initial Position: 0")
    print(f"Final Position: {positions[product]}")
    print(f"Unclosed PnL: {cash[product]}")

    # If final position is positive, we sell against the last timestamp's best bid
    if positions[product] > 0:
        closing_trade = {
            'timestamp': n_timestamps - 1,
            'type': 'CLOSING_SELL',
            'quantity': positions[product],
            'price': price_series[product].iloc[-1]["Bids"],
            'revenue': price_series[product].iloc[-1]["Bids"] * positions[product] * (1 - fees)
        }
        trade_history[product].append(closing_trade)
        cash[product] += price_series[product].iloc[-1]["Bids"] * positions[product] * (1 - fees)

    # If final position is negative, we buy against the last timestamp's best ask
    elif positions[product] < 0:
        closing_trade = {
            'timestamp': n_timestamps - 1,
            'type': 'CLOSING_BUY',
            'quantity': abs(positions[product]),
            'price': price_series[product].iloc[-1]["Asks"],
            'cost': price_series[product].iloc[-1]["Asks"] * abs(positions[product]) * (1 + fees)
        }
        trade_history[product].append(closing_trade)
        cash[product] -= price_series[product].iloc[-1]["Asks"] * -positions[product] * (1 + fees)

    # Add the cash of this product to the cash sum of all products
    cash_sum += cash[product]

    print(f"Final PnL: {cash[product]}")
    
    # Print trade summary
    print("\nTrade Summary:")
    print(f"Total number of trades: {len(trade_history[product])}")
    buy_trades = [t for t in trade_history[product] if t['type'] in ['BUY', 'CLOSING_BUY']]
    sell_trades = [t for t in trade_history[product] if t['type'] in ['SELL', 'CLOSING_SELL']]
    print(f"Number of buy trades: {len(buy_trades)}")
    print(f"Number of sell trades: {len(sell_trades)}")
    
    # Print detailed trade history
    

# Output final PnL
print(f"\nTotal PnL across all products = {cash_sum}")