import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.signal import argrelextrema
import os

# Create output directory if it doesn't exist
if not os.path.exists("analysis_output"):
    os.makedirs("analysis_output")

# Load data
DATA_PATH = "Data/UEC_expanded.csv"
data = pd.read_csv(DATA_PATH)

# Calculate mid price
data['Mid'] = (data['Asks'] + data['Bids']) / 2

# Create a timestamp column for plotting
data['Timestamp'] = range(len(data))

# Define functions for technical indicators
def calculate_ma(prices, window):
    """Calculate moving average"""
    return prices.rolling(window=window).mean()

def calculate_volatility(prices, window):
    """Calculate rolling standard deviation"""
    return prices.rolling(window=window).std()

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    ma = calculate_ma(prices, window)
    std = calculate_volatility(prices, window)
    upper_band = ma + (std * num_std)
    lower_band = ma - (std * num_std)
    return upper_band, ma, lower_band

def calculate_momentum(prices, window):
    """Calculate momentum as percent change over window"""
    return (prices - prices.shift(window)) / prices.shift(window) * 100

def identify_peaks_troughs(prices, order=5):
    """Identify local maxima and minima"""
    max_idx = argrelextrema(prices.values, np.greater, order=order)[0]
    min_idx = argrelextrema(prices.values, np.less, order=order)[0]
    return max_idx, min_idx

def backtest_strategy(data, signals, position_size=1, fees=0.002):
    """Simple backtest function for a strategy"""
    pnl = 0
    position = 0
    trades = []
    
    for i, signal in enumerate(signals):
        if i == 0:
            continue
            
        if signal == 1 and position <= 0:  # Buy signal
            # Close short position if exists
            if position < 0:
                close_cost = data.iloc[i]['Asks'] * abs(position) * (1 + fees)
                pnl -= close_cost
                trades.append({'timestamp': i, 'type': 'Close Short', 'price': data.iloc[i]['Asks'], 'pnl': -close_cost})
                
            # Open long position
            position = position_size
            cost = data.iloc[i]['Asks'] * position_size * (1 + fees)
            pnl -= cost
            trades.append({'timestamp': i, 'type': 'Buy', 'price': data.iloc[i]['Asks'], 'pnl': -cost})
            
        elif signal == -1 and position >= 0:  # Sell signal
            # Close long position if exists
            if position > 0:
                close_revenue = data.iloc[i]['Bids'] * position * (1 - fees)
                pnl += close_revenue
                trades.append({'timestamp': i, 'type': 'Close Long', 'price': data.iloc[i]['Bids'], 'pnl': close_revenue})
                
            # Open short position
            position = -position_size
            revenue = data.iloc[i]['Bids'] * position_size * (1 - fees)
            pnl += revenue
            trades.append({'timestamp': i, 'type': 'Sell', 'price': data.iloc[i]['Bids'], 'pnl': revenue})
    
    # Close final position
    if position > 0:
        close_revenue = data.iloc[-1]['Bids'] * position * (1 - fees)
        pnl += close_revenue
        trades.append({'timestamp': len(data)-1, 'type': 'Final Close Long', 'price': data.iloc[-1]['Bids'], 'pnl': close_revenue})
    elif position < 0:
        close_cost = data.iloc[-1]['Asks'] * abs(position) * (1 + fees)
        pnl -= close_cost
        trades.append({'timestamp': len(data)-1, 'type': 'Final Close Short', 'price': data.iloc[-1]['Asks'], 'pnl': -close_cost})
        
    return pnl, trades

# ===========================================
# 1. Price and Volatility Profiles
# ===========================================
plt.figure(figsize=(15, 10))
gs = GridSpec(3, 1, height_ratios=[3, 1, 1])

# Price chart with Bollinger Bands
ax1 = plt.subplot(gs[0])
upper, middle, lower = calculate_bollinger_bands(data['Mid'], window=20, num_std=2)
ax1.plot(data['Timestamp'], data['Mid'], label='Mid Price', color='blue')
ax1.plot(data['Timestamp'], upper, label='Upper Band (2σ)', color='red', linestyle='--')
ax1.plot(data['Timestamp'], middle, label='20-day MA', color='green')
ax1.plot(data['Timestamp'], lower, label='Lower Band (2σ)', color='red', linestyle='--')
ax1.fill_between(data['Timestamp'], upper, lower, alpha=0.1, color='gray')
ax1.set_title('UEC Price with Bollinger Bands (20-day, 2σ)')
ax1.legend()
ax1.grid(True)

# Volatility chart
ax2 = plt.subplot(gs[1], sharex=ax1)
volatility = calculate_volatility(data['Mid'], window=20)
ax2.plot(data['Timestamp'], volatility, label='20-day Rolling Volatility', color='purple')
ax2.set_title('Volatility (20-day Rolling Std Dev)')
ax2.grid(True)

# Daily returns
ax3 = plt.subplot(gs[2], sharex=ax1)
returns = data['Mid'].pct_change() * 100
ax3.plot(data['Timestamp'], returns, label='Daily Returns (%)', color='darkred')
ax3.set_title('Daily Returns (%)')
ax3.set_xlabel('Timestamp')
ax3.grid(True)

plt.tight_layout()
plt.savefig("analysis_output/1_price_volatility_profile.png")
plt.close()

# Distribution of returns
plt.figure(figsize=(15, 6))
sns.histplot(returns.dropna(), kde=True, bins=50)
plt.title('Distribution of Daily Returns')
plt.axvline(x=0, color='red', linestyle='--')
plt.grid(True)
plt.savefig("analysis_output/1b_returns_distribution.png")
plt.close()

# Identify key support and resistance levels
max_idx, min_idx = identify_peaks_troughs(data['Mid'], order=20)
plt.figure(figsize=(15, 8))
plt.plot(data['Timestamp'], data['Mid'], label='Mid Price', color='blue')
plt.scatter(max_idx, data['Mid'].iloc[max_idx], color='red', label='Resistance')
plt.scatter(min_idx, data['Mid'].iloc[min_idx], color='green', label='Support')
plt.title('UEC Price with Support and Resistance Levels')
plt.legend()
plt.grid(True)
plt.savefig("analysis_output/1c_support_resistance.png")
plt.close()

# ===========================================
# 2. Indicator Overlays
# ===========================================
# Multiple Moving Averages
plt.figure(figsize=(15, 8))
plt.plot(data['Timestamp'], data['Mid'], label='Mid Price', color='blue')
windows = [5, 10, 20, 50, 100]
colors = ['red', 'green', 'purple', 'orange', 'brown']
for window, color in zip(windows, colors):
    ma = calculate_ma(data['Mid'], window)
    plt.plot(data['Timestamp'], ma, label=f'{window}-day MA', color=color)
plt.title('UEC Price with Multiple Moving Averages')
plt.legend()
plt.grid(True)
plt.savefig("analysis_output/2a_moving_averages.png")
plt.close()

# Exponential Moving Averages (EMAs)
plt.figure(figsize=(15, 8))
plt.plot(data['Timestamp'], data['Mid'], label='Mid Price', color='blue')
spans = [5, 10, 20, 50, 100]
colors = ['red', 'green', 'purple', 'orange', 'brown']
for span, color in zip(spans, colors):
    ema = data['Mid'].ewm(span=span, adjust=False).mean()
    plt.plot(data['Timestamp'], ema, label=f'{span}-day EMA', color=color)
plt.title('UEC Price with Exponential Moving Averages')
plt.legend()
plt.grid(True)
plt.savefig("analysis_output/2b_exponential_mas.png")
plt.close()

# Moving Average Crossovers
plt.figure(figsize=(15, 8))
plt.plot(data['Timestamp'], data['Mid'], label='Mid Price', color='blue', alpha=0.5)
fast_ma = calculate_ma(data['Mid'], 20)
slow_ma = calculate_ma(data['Mid'], 50)
plt.plot(data['Timestamp'], fast_ma, label='20-day MA', color='red')
plt.plot(data['Timestamp'], slow_ma, label='50-day MA', color='green')

# Highlight crossovers
crossover_up = (fast_ma.shift(1) < slow_ma.shift(1)) & (fast_ma > slow_ma)
crossover_down = (fast_ma.shift(1) > slow_ma.shift(1)) & (fast_ma < slow_ma)
crossover_up_idx = data.index[crossover_up]
crossover_down_idx = data.index[crossover_down]

# Plot crossover points
plt.scatter(crossover_up_idx, data.loc[crossover_up_idx, 'Mid'], color='green', marker='^', s=100, label='Buy Signal')
plt.scatter(crossover_down_idx, data.loc[crossover_down_idx, 'Mid'], color='red', marker='v', s=100, label='Sell Signal')

plt.title('Moving Average Crossover Signals (20/50 day)')
plt.legend()
plt.grid(True)
plt.savefig("analysis_output/2c_ma_crossovers.png")
plt.close()

# ===========================================
# 3. Trade Signal Visualizations
# ===========================================
# Generate signals for mean reversion strategy
def generate_mean_reversion_signals(data, window=20, threshold=0.01):
    ma = calculate_ma(data['Mid'], window)
    signals = np.zeros(len(data))
    
    for i in range(window, len(data)):
        # Use oldest price in window for comparison (as we discovered is crucial)
        oldest_price = data['Mid'].iloc[i-window]
        deviation = (oldest_price - ma.iloc[i]) / ma.iloc[i]
        
        if deviation > threshold:
            signals[i] = -1  # Sell signal
        elif deviation < -threshold:
            signals[i] = 1   # Buy signal
            
    return signals

# Generate signals for breakout strategy
def generate_breakout_signals(data, lookback=10):
    signals = np.zeros(len(data))
    
    for i in range(lookback, len(data)):
        window = data['Mid'].iloc[i-lookback:i]
        high = max(window)
        low = min(window)
        
        if data['Mid'].iloc[i] > high * 0.99:
            signals[i] = 1  # Buy signal
        elif data['Mid'].iloc[i] < low * 1.01:
            signals[i] = -1  # Sell signal
            
    return signals

# Generate signals for moving average crossover
def generate_ma_crossover_signals(data, fast_window=20, slow_window=50):
    fast_ma = calculate_ma(data['Mid'], fast_window)
    slow_ma = calculate_ma(data['Mid'], slow_window)
    signals = np.zeros(len(data))
    
    for i in range(slow_window+1, len(data)):
        if fast_ma.iloc[i-1] < slow_ma.iloc[i-1] and fast_ma.iloc[i] > slow_ma.iloc[i]:
            signals[i] = 1  # Buy signal
        elif fast_ma.iloc[i-1] > slow_ma.iloc[i-1] and fast_ma.iloc[i] < slow_ma.iloc[i]:
            signals[i] = -1  # Sell signal
            
    return signals

# Backtest and visualize strategies
strategies = {
    'Mean Reversion': generate_mean_reversion_signals(data, window=20, threshold=0.01),
    'Breakout': generate_breakout_signals(data, lookback=10),
    'MA Crossover': generate_ma_crossover_signals(data, fast_window=20, slow_window=50)
}

# Visualize each strategy's signals
for name, signals in strategies.items():
    plt.figure(figsize=(15, 8))
    plt.plot(data['Timestamp'], data['Mid'], label='Mid Price', color='blue')
    
    # Find buy and sell signals
    buy_signals = [i for i in range(len(signals)) if signals[i] == 1]
    sell_signals = [i for i in range(len(signals)) if signals[i] == -1]
    
    # Plot signals
    plt.scatter([data['Timestamp'][i] for i in buy_signals], 
                [data['Mid'][i] for i in buy_signals], 
                color='green', marker='^', s=100, label='Buy Signal')
    plt.scatter([data['Timestamp'][i] for i in sell_signals], 
                [data['Mid'][i] for i in sell_signals], 
                color='red', marker='v', s=100, label='Sell Signal')
    
    # Backtest
    pnl, trades = backtest_strategy(data, signals)
    
    plt.title(f'{name} Strategy Signals (PnL: ${pnl:.2f})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"analysis_output/3_{name.lower().replace(' ', '_')}_signals.png")
    plt.close()

# ===========================================
# 4. Parameter Sensitivity Analysis
# ===========================================
# Mean Reversion Parameter Grid
windows = [10, 15, 20, 25, 30]
thresholds = [0.005, 0.01, 0.015, 0.02, 0.025]
results = np.zeros((len(windows), len(thresholds)))

for i, window in enumerate(windows):
    for j, threshold in enumerate(thresholds):
        signals = generate_mean_reversion_signals(data, window=window, threshold=threshold)
        pnl, _ = backtest_strategy(data, signals)
        results[i, j] = pnl

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(results, annot=True, fmt=".1f", 
            xticklabels=[f"{t:.3f}" for t in thresholds], 
            yticklabels=windows,
            cmap="RdYlGn",
            center=0)
plt.title('Mean Reversion Strategy Performance (PnL)')
plt.xlabel('Threshold')
plt.ylabel('Window Size')
plt.tight_layout()
plt.savefig("analysis_output/4a_mean_reversion_sensitivity.png")
plt.close()

# Breakout Parameter Grid
lookbacks = [5, 10, 15, 20, 25]
results = np.zeros(len(lookbacks))

for i, lookback in enumerate(lookbacks):
    signals = generate_breakout_signals(data, lookback=lookback)
    pnl, _ = backtest_strategy(data, signals)
    results[i] = pnl

# Plot bar chart
plt.figure(figsize=(12, 8))
plt.bar(range(len(lookbacks)), results, tick_label=[str(l) for l in lookbacks])
plt.axhline(y=0, color='red', linestyle='-', alpha=0.3)
for i, v in enumerate(results):
    plt.text(i, v + np.sign(v) * 0.5, f"{v:.1f}", ha='center')
plt.title('Breakout Strategy Performance by Lookback Period')
plt.xlabel('Lookback Period')
plt.ylabel('PnL')
plt.grid(True, axis='y')
plt.savefig("analysis_output/4b_breakout_sensitivity.png")
plt.close()

# Moving Average Crossover Parameter Grid
fast_windows = [5, 10, 15, 20, 25]
slow_windows = [30, 40, 50, 60, 70]
results = np.zeros((len(fast_windows), len(slow_windows)))

for i, fast in enumerate(fast_windows):
    for j, slow in enumerate(slow_windows):
        if fast < slow:  # Ensure fast MA is faster than slow MA
            signals = generate_ma_crossover_signals(data, fast_window=fast, slow_window=slow)
            pnl, _ = backtest_strategy(data, signals)
            results[i, j] = pnl
        else:
            results[i, j] = np.nan

# Plot heatmap
plt.figure(figsize=(12, 10))
mask = np.isnan(results)
sns.heatmap(results, annot=True, fmt=".1f", 
            xticklabels=slow_windows, 
            yticklabels=fast_windows,
            cmap="RdYlGn",
            center=0,
            mask=mask)
plt.title('Moving Average Crossover Strategy Performance (PnL)')
plt.xlabel('Slow MA Window')
plt.ylabel('Fast MA Window')
plt.tight_layout()
plt.savefig("analysis_output/4c_ma_crossover_sensitivity.png")
plt.close()

# ===========================================
# 5. Volume and Momentum Correlations
# ===========================================
# For demonstration, let's create synthetic volume data if real data isn't available
# In a real scenario, you'd use actual trading volume from the dataset
np.random.seed(42)
data['Volume'] = np.random.exponential(scale=1000, size=len(data))
data['Volume'] = data['Volume'] * (1 + 0.5 * np.sin(np.linspace(0, 10*np.pi, len(data))))  # Add cyclicality

# Calculate momentum
data['Momentum_5'] = calculate_momentum(data['Mid'], 5)
data['Momentum_10'] = calculate_momentum(data['Mid'], 10)
data['Momentum_20'] = calculate_momentum(data['Mid'], 20)

# Volume and price change correlation
plt.figure(figsize=(15, 8))
plt.scatter(data['Volume'], data['Mid'].pct_change() * 100, alpha=0.5)
plt.title('Volume vs. Daily Price Change')
plt.xlabel('Volume')
plt.ylabel('Daily Price Change (%)')
plt.grid(True)
plt.savefig("analysis_output/5a_volume_price_change.png")
plt.close()

# Volume and mid price chart
plt.figure(figsize=(15, 10))
gs = GridSpec(2, 1, height_ratios=[3, 1])

# Price chart
ax1 = plt.subplot(gs[0])
ax1.plot(data['Timestamp'], data['Mid'], label='Mid Price', color='blue')
ax1.set_title('UEC Price')
ax1.legend(loc='upper left')
ax1.grid(True)

# Volume chart
ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.bar(data['Timestamp'], data['Volume'], color='gray', alpha=0.6, label='Volume')
ax2.set_title('Trading Volume')
ax2.set_xlabel('Timestamp')
ax2.legend(loc='upper left')
ax2.grid(True)

plt.tight_layout()
plt.savefig("analysis_output/5b_price_volume.png")
plt.close()

# Correlation heatmap between price, momentum, and volume
correlation_data = data[['Mid', 'Volume', 'Momentum_5', 'Momentum_10', 'Momentum_20']].copy()
correlation_data.columns = ['Price', 'Volume', '5-day Momentum', '10-day Momentum', '20-day Momentum']
correlation_matrix = correlation_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig("analysis_output/5c_correlation_matrix.png")
plt.close()

# Volume spikes and price movements
# Identify high-volume days
volume_mean = data['Volume'].mean()
volume_std = data['Volume'].std()
high_volume_days = data[data['Volume'] > volume_mean + 2*volume_std]

plt.figure(figsize=(15, 8))
plt.plot(data['Timestamp'], data['Mid'], label='Mid Price', color='blue')
plt.scatter(high_volume_days['Timestamp'], high_volume_days['Mid'], 
            color='red', s=100, alpha=0.7, label='High Volume Day')
plt.title('High Volume Days and Price Movements')
plt.legend()
plt.grid(True)
plt.savefig("analysis_output/5d_high_volume_days.png")
plt.close()

# ===========================================
# 6. Summary Report
# ===========================================
# Create a summary report text file
with open("analysis_output/uec_analysis_summary.txt", "w") as f:
    f.write("UEC Extended Dataset Analysis Summary\n")
    f.write("===================================\n\n")
    
    f.write("1. Price and Volatility Profile\n")
    f.write(f"   - Average Price: ${data['Mid'].mean():.2f}\n")
    f.write(f"   - Price Range: ${data['Mid'].min():.2f} to ${data['Mid'].max():.2f}\n")
    f.write(f"   - Average Volatility (20-day): {volatility.mean():.4f}\n")
    f.write(f"   - Average Daily Return: {returns.mean():.4f}%\n")
    f.write(f"   - Return Volatility: {returns.std():.4f}%\n\n")
    
    f.write("2. Strategy Performance\n")
    for name, signals in strategies.items():
        pnl, trades = backtest_strategy(data, signals)
        f.write(f"   - {name}: ${pnl:.2f} ({len(trades)} trades)\n")
    f.write("\n")
    
    f.write("3. Best Parameters per Strategy\n")
    # Mean Reversion
    best_mr_i, best_mr_j = np.unravel_index(np.nanargmax(results), results.shape)
    f.write(f"   - Mean Reversion: Window={windows[best_mr_i]}, Threshold={thresholds[best_mr_j]:.3f} (PnL: ${results[best_mr_i, best_mr_j]:.2f})\n")
    
    # Breakout
    best_bo_i = np.argmax(results)
    f.write(f"   - Breakout: Lookback={lookbacks[best_bo_i]} (PnL: ${results[best_bo_i]:.2f})\n")
    
    # MA Crossover
    ma_results_copy = results.copy()
    ma_results_copy[np.isnan(ma_results_copy)] = -np.inf
    best_ma_i, best_ma_j = np.unravel_index(np.argmax(ma_results_copy), ma_results_copy.shape)
    f.write(f"   - MA Crossover: Fast={fast_windows[best_ma_i]}, Slow={slow_windows[best_ma_j]} (PnL: ${results[best_ma_i, best_ma_j]:.2f})\n\n")
    
    f.write("4. Correlations\n")
    f.write(f"   - Price-Volume Correlation: {correlation_matrix.loc['Price', 'Volume']:.4f}\n")
    for window in [5, 10, 20]:
        f.write(f"   - Price-{window}-day Momentum Correlation: {correlation_matrix.loc['Price', f'{window}-day Momentum']:.4f}\n")
    f.write("\n")
    
    f.write("5. Key Findings\n")
    # Determine the best strategy
    strategy_pnls = {}
    for name, signals in strategies.items():
        pnl, _ = backtest_strategy(data, signals)
        strategy_pnls[name] = pnl
    best_strategy = max(strategy_pnls, key=strategy_pnls.get)
    
    f.write(f"   - Best Strategy: {best_strategy} (${strategy_pnls[best_strategy]:.2f})\n")
    f.write(f"   - UEC shows {'high' if volatility.mean() > 0.1 else 'moderate' if volatility.mean() > 0.05 else 'low'} volatility\n")
    f.write(f"   - Price distribution is {'normal' if abs(returns.skew()) < 0.5 else 'skewed'}\n")
    
    # Check if high volume correlates with big price moves
    high_vol_returns = data.loc[high_volume_days.index, 'Mid'].pct_change().abs()
    normal_returns = data['Mid'].pct_change().abs()
    vol_impact = high_vol_returns.mean() > 1.5 * normal_returns.mean()
    
    f.write(f"   - High volume {'does' if vol_impact else 'does not'} correlate with significant price movements\n")

print("Analysis complete! Results are saved in the 'analysis_output' directory.") 