import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.signal import argrelextrema
import os
from scipy import stats

# Create output directory if it doesn't exist
if not os.path.exists("analysis_output_sober"):
    os.makedirs("analysis_output_sober")

# Load data
DATA_PATH = "Data/SOBER_expanded.csv"
data = pd.read_csv(DATA_PATH)

# Calculate mid price
data['Mid'] = (data['Asks'] + data['Bids']) / 2

# Create a timestamp column for plotting
data['Timestamp'] = range(len(data))

# Define functions for technical indicators
def calculate_ma(prices, window):
    """Calculate moving average"""
    if isinstance(prices, np.ndarray):
        return np.array([np.mean(prices[max(0, i-window+1):i+1]) if i >= window-1 else np.nan 
                         for i in range(len(prices))])
    else:
        return prices.rolling(window=window).mean()

def calculate_volatility(prices, window):
    """Calculate rolling standard deviation"""
    if isinstance(prices, np.ndarray):
        return np.array([np.std(prices[max(0, i-window+1):i+1]) if i >= window-1 else np.nan 
                         for i in range(len(prices))])
    else:
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
    
    asks = data['Asks'].values
    bids = data['Bids'].values
    
    for i in range(1, len(signals)):
        signal = signals[i]
            
        if signal == 1 and position <= 0:  # Buy signal
            # Close short position if exists
            if position < 0:
                close_cost = asks[i] * abs(position) * (1 + fees)
                pnl -= close_cost
                trades.append({'timestamp': i, 'type': 'Close Short', 'price': asks[i], 'pnl': -close_cost})
                
            # Open long position
            position = position_size
            cost = asks[i] * position_size * (1 + fees)
            pnl -= cost
            trades.append({'timestamp': i, 'type': 'Buy', 'price': asks[i], 'pnl': -cost})
            
        elif signal == -1 and position >= 0:  # Sell signal
            # Close long position if exists
            if position > 0:
                close_revenue = bids[i] * position * (1 - fees)
                pnl += close_revenue
                trades.append({'timestamp': i, 'type': 'Close Long', 'price': bids[i], 'pnl': close_revenue})
                
            # Open short position
            position = -position_size
            revenue = bids[i] * position_size * (1 - fees)
            pnl += revenue
            trades.append({'timestamp': i, 'type': 'Sell', 'price': bids[i], 'pnl': revenue})
    
    # Close final position
    if position > 0:
        close_revenue = bids[-1] * position * (1 - fees)
        pnl += close_revenue
        trades.append({'timestamp': len(data)-1, 'type': 'Final Close Long', 'price': bids[-1], 'pnl': close_revenue})
    elif position < 0:
        close_cost = asks[-1] * abs(position) * (1 + fees)
        pnl -= close_cost
        trades.append({'timestamp': len(data)-1, 'type': 'Final Close Short', 'price': asks[-1], 'pnl': -close_cost})
        
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
ax1.set_title('SOBER Price with Bollinger Bands (20-day, 2σ)')
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
plt.savefig("analysis_output_sober/1_price_volatility_profile.png")
plt.close()

# Distribution of returns
plt.figure(figsize=(15, 6))
sns.histplot(returns.dropna(), kde=True, bins=50)
plt.title('SOBER Distribution of Daily Returns')
plt.axvline(x=0, color='red', linestyle='--')
plt.grid(True)

# Add statistics to the plot
skewness = stats.skew(returns.dropna())
kurtosis = stats.kurtosis(returns.dropna())
plt.annotate(f'Skewness: {skewness:.4f}\nKurtosis: {kurtosis:.4f}', 
             xy=(0.05, 0.95), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.savefig("analysis_output_sober/2_returns_distribution.png")
plt.close()

# ===========================================
# 3. Strategy Signals
# ===========================================
# Generate signals for mean reversion strategy
def generate_mean_reversion_signals(data, window=20, threshold=4):
    prices = data['Mid'].values
    ma = calculate_ma(prices, window)
    signals = np.zeros(len(prices))
    
    for i in range(window, len(prices)):
        # Use oldest price in window for comparison
        oldest_price = prices[i-window]
        if not np.isnan(ma[i]):
            deviation = (oldest_price - ma[i])
            
            if deviation > threshold:
                signals[i] = -1  # Sell signal
            elif deviation < -threshold:
                signals[i] = 1   # Buy signal
                
    return signals

# Generate signals for momentum strategy
def generate_momentum_signals(data, window=5, threshold=0.5):
    signals = np.zeros(len(data))
    prices = data['Mid'].values
    
    for i in range(window, len(prices)):
        if i >= window:
            # Calculate momentum directly
            momentum_pct = (prices[i] - prices[i-window]) / prices[i-window] * 100
            
            if momentum_pct > threshold:
                signals[i] = 1  # Buy signal
            elif momentum_pct < -threshold:
                signals[i] = -1  # Sell signal
                
    return signals

# Generate signals for breakout strategy
def generate_breakout_signals(data, lookback=10):
    signals = np.zeros(len(data))
    prices = data['Mid'].values  # Convert to numpy array for faster processing
    
    for i in range(lookback, len(prices)):
        # Use numpy operations instead of pandas for better performance
        window = prices[i-lookback:i]
        high = np.max(window)
        low = np.min(window)
        
        if prices[i] > high * 0.99:
            signals[i] = 1  # Buy signal
        elif prices[i] < low * 1.01:
            signals[i] = -1  # Sell signal
            
    return signals

# Backtest and visualize strategies
strategies = {
    'Mean Reversion': generate_mean_reversion_signals(data, window=20, threshold=4),
    'Momentum': generate_momentum_signals(data, window=5, threshold=0.5),
    'Breakout': generate_breakout_signals(data, lookback=10)
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
    
    plt.title(f'SOBER {name} Strategy Signals (PnL: ${pnl:.2f}, Trades: {len(trades)})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"analysis_output_sober/3_{name.lower().replace(' ', '_')}_signals.png")
    plt.close()

# ===========================================
# 4. Parameter Sensitivity Analysis
# ===========================================
# Mean Reversion Parameter Grid - Note the higher threshold values for SOBER
windows = [10, 15, 20, 25, 30]
thresholds = [2, 3, 4, 5, 6]  # SOBER typically needs higher thresholds
results_mr = np.zeros((len(windows), len(thresholds)))

for i, window in enumerate(windows):
    for j, threshold in enumerate(thresholds):
        signals = generate_mean_reversion_signals(data, window=window, threshold=threshold)
        pnl, _ = backtest_strategy(data, signals)
        results_mr[i, j] = pnl

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(results_mr, annot=True, fmt=".1f", 
            xticklabels=[str(t) for t in thresholds], 
            yticklabels=windows,
            cmap="RdYlGn",
            center=0)
plt.title('SOBER Mean Reversion Strategy Performance (PnL)')
plt.xlabel('Threshold')
plt.ylabel('Window Size')
plt.tight_layout()
plt.savefig("analysis_output_sober/4a_mean_reversion_sensitivity.png")
plt.close()

# Momentum Parameter Grid
windows = [3, 5, 7, 10, 15]
thresholds = [0.3, 0.5, 0.7, 1.0, 1.5]
results_mom = np.zeros((len(windows), len(thresholds)))

for i, window in enumerate(windows):
    for j, threshold in enumerate(thresholds):
        signals = generate_momentum_signals(data, window=window, threshold=threshold)
        pnl, _ = backtest_strategy(data, signals)
        results_mom[i, j] = pnl

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(results_mom, annot=True, fmt=".1f", 
            xticklabels=[str(t) for t in thresholds], 
            yticklabels=windows,
            cmap="RdYlGn",
            center=0)
plt.title('SOBER Momentum Strategy Performance (PnL)')
plt.xlabel('Threshold')
plt.ylabel('Window Size')
plt.tight_layout()
plt.savefig("analysis_output_sober/4b_momentum_sensitivity.png")
plt.close()

# Breakout Parameter Grid
lookbacks = [5, 10, 15, 20, 25]
results_bo = np.zeros(len(lookbacks))

for i, lookback in enumerate(lookbacks):
    signals = generate_breakout_signals(data, lookback=lookback)
    pnl, _ = backtest_strategy(data, signals)
    results_bo[i] = pnl

# Plot bar chart
plt.figure(figsize=(12, 8))
plt.bar(range(len(lookbacks)), results_bo, tick_label=[str(l) for l in lookbacks])
plt.axhline(y=0, color='red', linestyle='-', alpha=0.3)
for i, v in enumerate(results_bo):
    plt.text(i, v + np.sign(v) * 0.5, f"{v:.1f}", ha='center')
plt.title('SOBER Breakout Strategy Performance by Lookback Period')
plt.xlabel('Lookback Period')
plt.ylabel('PnL')
plt.grid(True, axis='y')
plt.savefig("analysis_output_sober/4c_breakout_sensitivity.png")
plt.close()

# ===========================================
# 5. Correlation Analysis
# ===========================================
# Calculate additional technical indicators
data['SMA5'] = calculate_ma(data['Mid'], 5)
data['SMA20'] = calculate_ma(data['Mid'], 20)
data['Momentum_1'] = calculate_momentum(data['Mid'], 1)
data['Momentum_5'] = calculate_momentum(data['Mid'], 5)
data['Momentum_10'] = calculate_momentum(data['Mid'], 10)
data['Volatility_10'] = calculate_volatility(data['Mid'], 10)

# Create a synthetic volume feature for demonstration
np.random.seed(42)
data['Volume'] = np.random.exponential(scale=1000, size=len(data))
data['Volume'] = data['Volume'] * (1 + 0.5 * np.sin(np.linspace(0, 10*np.pi, len(data))))

# Correlation matrix with these features
correlation_data = data[['Mid', 'SMA5', 'SMA20', 'Momentum_1', 'Momentum_5', 'Momentum_10', 'Volatility_10', 'Volume']].copy()
correlation_data.columns = ['Price', 'SMA5', 'SMA20', '1-day Mom', '5-day Mom', '10-day Mom', 'Volatility', 'Volume']
correlation_matrix = correlation_data.corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, mask=mask)
plt.title('SOBER Correlation Matrix')
plt.tight_layout()
plt.savefig("analysis_output_sober/5_correlation_heatmap.png")
plt.close()

# ===========================================
# 6. Comparison with UEC - SOBER vs UEC Analysis
# ===========================================
# Only if UEC data is available
try:
    uec_data = pd.read_csv("Data/UEC_expanded.csv")
    uec_data['Mid'] = (uec_data['Asks'] + uec_data['Bids']) / 2
    
    # Normalize both price series to start at the same value
    sober_norm = data['Mid'] / data['Mid'].iloc[0]
    uec_norm = uec_data['Mid'] / uec_data['Mid'].iloc[0]
    
    # Plot normalized prices
    plt.figure(figsize=(15, 8))
    plt.plot(sober_norm.index, sober_norm, label='SOBER (normalized)', color='blue')
    plt.plot(uec_norm.index[:len(sober_norm)], uec_norm[:len(sober_norm)], label='UEC (normalized)', color='red')
    plt.title('SOBER vs UEC - Normalized Price Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig("analysis_output_sober/6_sober_vs_uec.png")
    plt.close()
    
    # Compare volatility
    sober_vol = calculate_volatility(data['Mid'], 20)
    uec_vol = calculate_volatility(uec_data['Mid'][:len(data)], 20)
    
    plt.figure(figsize=(15, 8))
    plt.plot(sober_vol.index, sober_vol, label='SOBER Volatility', color='blue')
    plt.plot(uec_vol.index[:len(sober_vol)], uec_vol[:len(sober_vol)], label='UEC Volatility', color='red')
    plt.title('SOBER vs UEC - Volatility Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig("analysis_output_sober/6b_volatility_comparison.png")
    plt.close()
except Exception as e:
    print(f"Couldn't load UEC data for comparison: {e}")

# ===========================================
# 7. Summary Report
# ===========================================
# Create a summary report text file
with open("analysis_output_sober/sober_analysis_summary.txt", "w") as f:
    f.write("SOBER Extended Dataset Analysis Summary\n")
    f.write("======================================\n\n")
    
    f.write("1. Price and Volatility Profile\n")
    f.write(f"   - Average Price: ${data['Mid'].mean():.2f}\n")
    f.write(f"   - Price Range: ${data['Mid'].min():.2f} to ${data['Mid'].max():.2f}\n")
    f.write(f"   - Average Volatility (20-day): {volatility.mean():.4f}\n")
    f.write(f"   - Average Daily Return: {returns.mean():.4f}%\n")
    f.write(f"   - Return Volatility: {returns.std():.4f}%\n")
    f.write(f"   - Skewness: {skewness:.4f} ({'positively' if skewness > 0 else 'negatively'} skewed)\n")
    f.write(f"   - Kurtosis: {kurtosis:.4f} ({'leptokurtic' if kurtosis > 0 else 'platykurtic'})\n\n")
    
    f.write("2. Strategy Performance\n")
    for name, signals in strategies.items():
        pnl, trades = backtest_strategy(data, signals)
        f.write(f"   - {name}: ${pnl:.2f} ({len(trades)} trades)\n")
    f.write("\n")
    
    f.write("3. Best Parameters per Strategy\n")
    # Mean Reversion
    best_mr_i, best_mr_j = np.unravel_index(np.nanargmax(results_mr), results_mr.shape)
    f.write(f"   - Mean Reversion: Window={windows[best_mr_i]}, Threshold={thresholds[best_mr_j]} (PnL: ${results_mr[best_mr_i, best_mr_j]:.2f})\n")
    
    # Momentum
    best_mom_i, best_mom_j = np.unravel_index(np.nanargmax(results_mom), results_mom.shape)
    f.write(f"   - Momentum: Window={windows[best_mom_i]}, Threshold={thresholds[best_mom_j]} (PnL: ${results_mom[best_mom_i, best_mom_j]:.2f})\n")
    
    # Breakout
    best_bo_i = np.argmax(results_bo)
    f.write(f"   - Breakout: Lookback={lookbacks[best_bo_i]} (PnL: ${results_bo[best_bo_i]:.2f})\n\n")
    
    f.write("4. Key Findings\n")
    # Determine the best strategy
    strategy_pnls = {}
    for name, signals in strategies.items():
        pnl, _ = backtest_strategy(data, signals)
        strategy_pnls[name] = pnl
    best_strategy = max(strategy_pnls, key=strategy_pnls.get)
    
    f.write(f"   - Best Strategy: {best_strategy} (${strategy_pnls[best_strategy]:.2f})\n")
    f.write(f"   - SOBER shows {'high' if volatility.mean() > 0.1 else 'moderate' if volatility.mean() > 0.05 else 'low'} volatility\n")
    f.write(f"   - Mean Reversion strategy works {'well' if max(strategy_pnls.values()) > 0 else 'poorly'} for SOBER\n")
    
    # Correlations summary
    strongest_correlation = correlation_matrix['Price'].drop('Price').abs().idxmax()
    strongest_value = correlation_matrix.loc['Price', strongest_correlation]
    f.write(f"   - Price has strongest correlation with {strongest_correlation} ({strongest_value:.4f})\n")

print("SOBER analysis complete! Results are saved in the 'analysis_output_sober' directory.") 