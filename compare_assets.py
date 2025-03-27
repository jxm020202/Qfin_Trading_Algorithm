import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.signal import argrelextrema
import os
from scipy import stats
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# Create output directory if it doesn't exist
output_dir = "asset_comparison"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Asset settings
assets = {
    "UEC": {
        "data_path": "Data/UEC_expanded.csv",
        "color": "blue",
        "thresholds": {
            "mean_reversion": [0.1, 0.2, 0.3, 0.5, 1.0],
            "momentum": [0.3, 0.5, 0.7, 1.0, 1.5],
        },
        "windows": {
            "mean_reversion": [10, 15, 20, 25, 30],
            "momentum": [3, 5, 7, 10, 15],
        }
    },
    "SOBER": {
        "data_path": "Data/SOBER_expanded.csv",
        "color": "green",
        "thresholds": {
            "mean_reversion": [2, 3, 4, 5, 6],
            "momentum": [0.3, 0.5, 0.7, 1.0, 1.5],
        },
        "windows": {
            "mean_reversion": [10, 15, 20, 25, 30],
            "momentum": [3, 5, 7, 10, 15],
        }
    }
}

# Load data for each asset
for asset_name, asset_info in assets.items():
    try:
        df = pd.read_csv(asset_info["data_path"])
        df['Mid'] = (df['Asks'] + df['Bids']) / 2
        df['Timestamp'] = range(len(df))
        df['Returns'] = df['Mid'].pct_change() * 100
        
        # Downsample data for plotting (to improve performance)
        if len(df) > 5000:
            sample_rate = max(1, len(df) // 5000)
            asset_info["plot_data"] = df.iloc[::sample_rate].copy()
            print(f"Downsampled {asset_name} data for plotting: {len(asset_info['plot_data'])} points")
        else:
            asset_info["plot_data"] = df.copy()
            
        assets[asset_name]["data"] = df
        print(f"Loaded {asset_name} data: {len(df)} rows")
    except Exception as e:
        print(f"Error loading {asset_name} data: {e}")
        assets[asset_name]["data"] = None
        assets[asset_name]["plot_data"] = None

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

def generate_mean_reversion_signals(data, window=20, threshold=4):
    """Generate signals for mean reversion strategy"""
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

def generate_momentum_signals(data, window=5, threshold=0.5):
    """Generate signals for momentum strategy"""
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

def generate_breakout_signals(data, lookback=10):
    """Generate signals for breakout strategy"""
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

def backtest_strategy(data, signals, position_size=1, fees=0.002):
    """Simple backtest function for a strategy"""
    pnl = 0
    position = 0
    trades = []
    equity_curve = np.zeros(len(signals))
    
    asks = data['Asks'].values
    bids = data['Bids'].values
    
    for i in range(1, len(signals)):
        # Carry forward previous equity value
        equity_curve[i] = equity_curve[i-1]
        
        signal = signals[i]
            
        if signal == 1 and position <= 0:  # Buy signal
            # Close short position if exists
            if position < 0:
                close_cost = asks[i] * abs(position) * (1 + fees)
                pnl -= close_cost
                equity_curve[i] -= close_cost
                trades.append({'timestamp': i, 'type': 'Close Short', 'price': asks[i], 'pnl': -close_cost})
                
            # Open long position
            position = position_size
            cost = asks[i] * position_size * (1 + fees)
            pnl -= cost
            equity_curve[i] -= cost
            trades.append({'timestamp': i, 'type': 'Buy', 'price': asks[i], 'pnl': -cost})
            
        elif signal == -1 and position >= 0:  # Sell signal
            # Close long position if exists
            if position > 0:
                close_revenue = bids[i] * position * (1 - fees)
                pnl += close_revenue
                equity_curve[i] += close_revenue
                trades.append({'timestamp': i, 'type': 'Close Long', 'price': bids[i], 'pnl': close_revenue})
                
            # Open short position
            position = -position_size
            revenue = bids[i] * position_size * (1 - fees)
            pnl += revenue
            equity_curve[i] += revenue
            trades.append({'timestamp': i, 'type': 'Sell', 'price': bids[i], 'pnl': revenue})
        
        # Mark-to-market open positions for equity curve
        if position > 0:
            equity_curve[i] = equity_curve[i-1] + position * (bids[i] - bids[i-1])
        elif position < 0:
            equity_curve[i] = equity_curve[i-1] - position * (asks[i] - asks[i-1])
    
    # Close final position
    if position > 0:
        close_revenue = bids[-1] * position * (1 - fees)
        pnl += close_revenue
        trades.append({'timestamp': len(data)-1, 'type': 'Final Close Long', 'price': bids[-1], 'pnl': close_revenue})
    elif position < 0:
        close_cost = asks[-1] * abs(position) * (1 + fees)
        pnl -= close_cost
        trades.append({'timestamp': len(data)-1, 'type': 'Final Close Short', 'price': asks[-1], 'pnl': -close_cost})
        
    return pnl, trades, equity_curve

# Generate all visualizations for each asset
for asset_name, asset_info in assets.items():
    if asset_info["data"] is None:
        print(f"Skipping {asset_name} analysis due to missing data")
        continue
    
    data = asset_info["data"]
    plot_data = asset_info["plot_data"]
    color = asset_info["color"]
    
    print(f"Analyzing {asset_name}...")
    
    # 1. Price & Volatility Profile
    plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 1, height_ratios=[3, 1, 1])
    
    # Price chart with Bollinger Bands
    ax1 = plt.subplot(gs[0])
    upper, middle, lower = calculate_bollinger_bands(data['Mid'], window=20, num_std=2)
    
    # Use downsampled data for plotting
    plot_idx = plot_data.index
    ax1.plot(plot_data['Timestamp'], plot_data['Mid'], label='Mid Price', color=color)
    ax1.plot(plot_data['Timestamp'], upper.iloc[plot_idx], label='Upper Band (2σ)', color='red', linestyle='--')
    ax1.plot(plot_data['Timestamp'], middle.iloc[plot_idx], label='20-day MA', color='purple')
    ax1.plot(plot_data['Timestamp'], lower.iloc[plot_idx], label='Lower Band (2σ)', color='red', linestyle='--')
    ax1.fill_between(plot_data['Timestamp'], upper.iloc[plot_idx], lower.iloc[plot_idx], alpha=0.1, color='gray')
    ax1.set_title(f'{asset_name} Price with Bollinger Bands (20-day, 2σ)')
    ax1.legend()
    ax1.grid(True)
    
    # Volatility chart
    ax2 = plt.subplot(gs[1], sharex=ax1)
    volatility = calculate_volatility(data['Mid'], window=20)
    ax2.plot(plot_data['Timestamp'], volatility.iloc[plot_idx], label='20-day Rolling Volatility', color='darkred')
    ax2.set_title('Volatility (20-day Rolling Std Dev)')
    ax2.grid(True)
    
    # Daily returns
    ax3 = plt.subplot(gs[2], sharex=ax1)
    returns = data['Returns']
    ax3.plot(plot_data['Timestamp'], returns.iloc[plot_idx], label='Daily Returns (%)', color='green')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_title('Daily Returns (%)')
    ax3.set_xlabel('Timestamp')
    ax3.grid(True)
    
    plt.subplots_adjust(hspace=0.4)  # Use subplots_adjust instead of tight_layout
    plt.savefig(f"{output_dir}/{asset_name}_price_volatility_profile.png")
    plt.close()
    
    # 2. Returns Distribution
    plt.figure(figsize=(15, 8))
    sns.histplot(returns.dropna(), kde=True, bins=50, color=color)
    plt.axvline(x=0, color='red', linestyle='--')
    
    # Add statistics to the plot
    skewness = stats.skew(returns.dropna())
    kurtosis = stats.kurtosis(returns.dropna())
    mean_return = returns.mean()
    std_return = returns.std()
    
    plt.title(f'{asset_name} Distribution of Daily Returns')
    plt.annotate(f'Mean: {mean_return:.4f}%\nStd Dev: {std_return:.4f}%\nSkewness: {skewness:.4f}\nKurtosis: {kurtosis:.4f}', 
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    plt.grid(True)
    plt.savefig(f"{output_dir}/{asset_name}_returns_distribution.png")
    plt.close()
    
    # 3. Strategy Signals Chart
    strategies = {
        'Mean Reversion': generate_mean_reversion_signals(
            data, 
            window=asset_info["windows"]["mean_reversion"][2], 
            threshold=asset_info["thresholds"]["mean_reversion"][2]
        ),
        'Momentum': generate_momentum_signals(
            data, 
            window=asset_info["windows"]["momentum"][1], 
            threshold=asset_info["thresholds"]["momentum"][1]
        ),
        'Breakout': generate_breakout_signals(data, lookback=10)
    }
    
    # Visualize each strategy's signals
    for name, signals in strategies.items():
        plt.figure(figsize=(15, 8))
        plt.plot(plot_data['Timestamp'], plot_data['Mid'], label='Mid Price', color=color)
        
        # Find buy and sell signals (use downsampled data for plotting)
        buy_idx = [i for i in plot_idx if i < len(signals) and signals[i] == 1]
        sell_idx = [i for i in plot_idx if i < len(signals) and signals[i] == -1]
        
        # Plot signals
        if buy_idx:
            plt.scatter([data['Timestamp'].iloc[i] for i in buy_idx], 
                        [data['Mid'].iloc[i] for i in buy_idx], 
                        color='green', marker='^', s=100, label='Buy Signal')
        if sell_idx:
            plt.scatter([data['Timestamp'].iloc[i] for i in sell_idx], 
                        [data['Mid'].iloc[i] for i in sell_idx], 
                        color='red', marker='v', s=100, label='Sell Signal')
        
        # Backtest
        pnl, trades, equity = backtest_strategy(data, signals)
        
        plt.title(f'{asset_name} {name} Strategy Signals (PnL: ${pnl:.2f}, Trades: {len(trades)})')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/{asset_name}_{name.lower().replace(' ', '_')}_signals.png")
        plt.close()
        
        # 6. Equity Curve
        plt.figure(figsize=(15, 8))
        plt.plot(plot_data['Timestamp'], equity[plot_idx], label='Strategy Equity', color=color)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title(f'{asset_name} {name} Strategy Equity Curve')
        plt.xlabel('Timestamp')
        plt.ylabel('Cumulative P&L ($)')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{output_dir}/{asset_name}_{name.lower().replace(' ', '_')}_equity.png")
        plt.close()
    
    # 4. Parameter Sensitivity Analysis for Mean Reversion
    windows = asset_info["windows"]["mean_reversion"]
    thresholds = asset_info["thresholds"]["mean_reversion"]
    results = np.zeros((len(windows), len(thresholds)))
    
    for i, window in enumerate(windows):
        for j, threshold in enumerate(thresholds):
            signals = generate_mean_reversion_signals(data, window=window, threshold=threshold)
            pnl, _, _ = backtest_strategy(data, signals)
            results[i, j] = pnl
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(results, annot=True, fmt=".1f", 
                xticklabels=[str(t) for t in thresholds], 
                yticklabels=windows,
                cmap="RdYlGn",
                center=0)
    plt.title(f'{asset_name} Mean Reversion Strategy Performance (PnL)')
    plt.xlabel('Threshold')
    plt.ylabel('Window Size')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{asset_name}_mean_reversion_sensitivity.png")
    plt.close()
    
    # Parameter Sensitivity for Momentum
    windows = asset_info["windows"]["momentum"]
    thresholds = asset_info["thresholds"]["momentum"]
    results = np.zeros((len(windows), len(thresholds)))
    
    for i, window in enumerate(windows):
        for j, threshold in enumerate(thresholds):
            signals = generate_momentum_signals(data, window=window, threshold=threshold)
            pnl, _, _ = backtest_strategy(data, signals)
            results[i, j] = pnl
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(results, annot=True, fmt=".1f", 
                xticklabels=[str(t) for t in thresholds], 
                yticklabels=windows,
                cmap="RdYlGn",
                center=0)
    plt.title(f'{asset_name} Momentum Strategy Performance (PnL)')
    plt.xlabel('Threshold')
    plt.ylabel('Window Size')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{asset_name}_momentum_sensitivity.png")
    plt.close()
    
    # 5. Correlation Matrix
    # Calculate additional indicators
    data['SMA5'] = calculate_ma(data['Mid'], 5)
    data['SMA20'] = calculate_ma(data['Mid'], 20)
    data['Momentum_1'] = calculate_momentum(data['Mid'], 1)
    data['Momentum_5'] = calculate_momentum(data['Mid'], 5)
    data['Momentum_10'] = calculate_momentum(data['Mid'], 10)
    data['Volatility_10'] = calculate_volatility(data['Mid'], 10)
    
    # Create synthetic volume for demonstration (if real volume isn't available)
    if 'Volume' not in data.columns:
        np.random.seed(42)
        data['Volume'] = np.random.exponential(scale=1000, size=len(data))
        data['Volume'] = data['Volume'] * (1 + 0.5 * np.sin(np.linspace(0, 10*np.pi, len(data))))
    
    # Create correlation matrix
    correlation_data = data[['Mid', 'SMA5', 'SMA20', 'Momentum_1', 'Momentum_5', 'Momentum_10', 'Volatility_10', 'Volume']].copy()
    correlation_data.columns = ['Price', 'SMA5', 'SMA20', '1-day Mom', '5-day Mom', '10-day Mom', 'Volatility', 'Volume']
    correlation_matrix = correlation_data.corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, mask=mask)
    plt.title(f'{asset_name} Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{asset_name}_correlation_matrix.png")
    plt.close()

# Create a comparison dashboard
print("Creating comparison dashboard...")

# Price comparison (normalized)
plt.figure(figsize=(15, 8))
for asset_name, asset_info in assets.items():
    if asset_info["data"] is not None:
        plot_data = asset_info["plot_data"]
        norm_price = plot_data['Mid'] / plot_data['Mid'].iloc[0] * 100  # Normalize to start at 100
        plt.plot(plot_data['Timestamp'], norm_price, label=f'{asset_name}', color=asset_info["color"])

plt.title('Normalized Price Comparison (Start = 100)')
plt.xlabel('Timestamp')
plt.ylabel('Normalized Price')
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/normalized_price_comparison.png")
plt.close()

# Volatility comparison
plt.figure(figsize=(15, 8))
for asset_name, asset_info in assets.items():
    if asset_info["data"] is not None:
        data = asset_info["data"]
        plot_data = asset_info["plot_data"]
        volatility = calculate_volatility(data['Mid'], window=20)
        plt.plot(plot_data['Timestamp'], volatility.iloc[plot_data.index], 
                 label=f'{asset_name} Volatility', color=asset_info["color"])

plt.title('20-day Volatility Comparison')
plt.xlabel('Timestamp')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/volatility_comparison.png")
plt.close()

# Strategy performance comparison
strategies = ['Mean Reversion', 'Momentum', 'Breakout']
performance = {asset_name: [] for asset_name in assets.keys()}

for asset_name, asset_info in assets.items():
    if asset_info["data"] is not None:
        data = asset_info["data"]
        
        # Generate signals for each strategy using default parameters
        mr_signals = generate_mean_reversion_signals(
            data, 
            window=asset_info["windows"]["mean_reversion"][2], 
            threshold=asset_info["thresholds"]["mean_reversion"][2]
        )
        mom_signals = generate_momentum_signals(
            data, 
            window=asset_info["windows"]["momentum"][1], 
            threshold=asset_info["thresholds"]["momentum"][1]
        )
        bo_signals = generate_breakout_signals(data, lookback=10)
        
        # Backtest each strategy
        mr_pnl, mr_trades, _ = backtest_strategy(data, mr_signals)
        mom_pnl, mom_trades, _ = backtest_strategy(data, mom_signals)
        bo_pnl, bo_trades, _ = backtest_strategy(data, bo_signals)
        
        # Store results
        performance[asset_name] = [mr_pnl, mom_pnl, bo_pnl]

# Create bar chart for strategy performance comparison
plt.figure(figsize=(12, 8))
bar_width = 0.25
x = np.arange(len(strategies))

for i, (asset_name, asset_info) in enumerate(assets.items()):
    if asset_info["data"] is not None:
        plt.bar(x + i*bar_width, performance[asset_name], width=bar_width, 
                label=asset_name, color=asset_info["color"])

plt.xlabel('Strategy')
plt.ylabel('Profit & Loss ($)')
plt.title('Strategy Performance Comparison')
plt.xticks(x + bar_width/2, strategies)
plt.legend()
plt.grid(True, axis='y')

# Add value labels above/below bars
for i, (asset_name, asset_info) in enumerate(assets.items()):
    if asset_info["data"] is not None:
        for j, v in enumerate(performance[asset_name]):
            plt.text(x[j] + i*bar_width, v + np.sign(v) * 5, 
                    f"${v:.0f}", ha='center', va='bottom' if v > 0 else 'top')

plt.savefig(f"{output_dir}/strategy_performance_comparison.png")
plt.close()

print(f"Analysis complete! Results saved in the '{output_dir}' directory.")

# Create summary report
with open(f"{output_dir}/analysis_summary.txt", "w") as f:
    f.write("Asset Analysis Summary\n")
    f.write("=====================\n\n")
    
    for asset_name, asset_info in assets.items():
        if asset_info["data"] is None:
            f.write(f"{asset_name}: Data not available\n")
            continue
            
        data = asset_info["data"]
        
        f.write(f"\n{asset_name} Summary:\n")
        f.write("------------------------\n")
        f.write(f"Price Range: ${data['Mid'].min():.2f} to ${data['Mid'].max():.2f}\n")
        f.write(f"Average Price: ${data['Mid'].mean():.2f}\n")
        f.write(f"Volatility (20-day avg): {volatility.mean():.4f}\n")
        f.write(f"Average Daily Return: {data['Returns'].mean():.4f}%\n")
        f.write(f"Return Volatility: {data['Returns'].std():.4f}%\n")
        
        f.write("\nStrategy Performance:\n")
        for i, strategy in enumerate(strategies):
            f.write(f"  - {strategy}: ${performance[asset_name][i]:.2f}\n")
        
        # Determine best strategy
        best_idx = np.argmax(performance[asset_name])
        f.write(f"\nBest Strategy: {strategies[best_idx]} (${performance[asset_name][best_idx]:.2f})\n")
        
        # Key insights based on correlations
        correlation_data = data[['Mid', 'SMA5', 'SMA20', 'Momentum_1', 'Momentum_5', 'Momentum_10', 'Volatility_10', 'Volume']].copy()
        correlation_data.columns = ['Price', 'SMA5', 'SMA20', '1-day Mom', '5-day Mom', '10-day Mom', 'Volatility', 'Volume']
        correlation_matrix = correlation_data.corr()
        
        strongest_correlation = correlation_matrix['Price'].drop('Price').abs().idxmax()
        strongest_value = correlation_matrix.loc['Price', strongest_correlation]
        f.write(f"Strongest correlation with price: {strongest_correlation} ({strongest_value:.4f})\n\n")
    
    # Overall comparison
    f.write("\nComparison Insights:\n")
    f.write("------------------------\n")
    
    # Find overall best strategy
    best_strategy_overall = None
    best_performance_overall = float('-inf')
    
    for asset_name, perfs in performance.items():
        for i, perf in enumerate(perfs):
            if perf > best_performance_overall:
                best_performance_overall = perf
                best_strategy_overall = (asset_name, strategies[i])
    
    if best_strategy_overall:
        f.write(f"Best overall strategy: {best_strategy_overall[1]} on {best_strategy_overall[0]} (${best_performance_overall:.2f})\n")
    
    # Compare volatility
    volatilities = {}
    for asset_name, asset_info in assets.items():
        if asset_info["data"] is not None:
            volatilities[asset_name] = calculate_volatility(asset_info["data"]['Mid'], window=20).mean()
    
    if len(volatilities) >= 2:
        more_volatile = max(volatilities.items(), key=lambda x: x[1])[0]
        less_volatile = min(volatilities.items(), key=lambda x: x[1])[0]
        f.write(f"{more_volatile} shows {volatilities[more_volatile]/volatilities[less_volatile]:.1f}x higher volatility than {less_volatile}\n")
    
    f.write("\nRecommendations for Universal Strategy:\n")
    
    if len(performance) >= 2:
        # Check if the same strategy works well for both assets
        uec_best = np.argmax(performance.get('UEC', [0, 0, 0]))
        sober_best = np.argmax(performance.get('SOBER', [0, 0, 0]))
        
        if uec_best == sober_best:
            f.write(f"• {strategies[uec_best]} strategy works best for both assets.\n")
            f.write("• Consider using this strategy with asset-specific parameters.\n")
        else:
            f.write("• Different strategies work best for each asset.\n")
            f.write(f"• For UEC: {strategies[uec_best]} performs best.\n")
            f.write(f"• For SOBER: {strategies[sober_best]} performs best.\n")
            f.write("• Consider a hybrid approach that detects regime and adapts strategy.\n")
        
        f.write("• Key adjustments needed per asset:\n")
        for asset_name in performance.keys():
            f.write(f"  - {asset_name}: Use {'higher' if asset_name == 'SOBER' else 'lower'} thresholds for signal generation\n")
    
    f.write("\nNote: These analyses use limited historical data and simplified assumptions.")
    f.write("Real-world performance may vary significantly.") 