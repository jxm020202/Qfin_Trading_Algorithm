# Algorithmic Trading Backtesting System

This system implements an automated backtesting framework for algorithmic trading strategies. It tests multiple trading strategies on historical market data and provides performance analysis and visualization.

## Features

- Multiple trading strategies:
  - Moving Average Crossover
  - Mean Reversion
  - Breakout
  - Momentum
- Realistic trading simulation with:
  - Bid/Ask spread
  - Transaction fees
  - Position limits
  - Cash management
- Performance analysis:
  - Equity curves
  - Total returns
  - Number of trades
  - Visual performance comparison

## Requirements

- Python 3.7 or higher
- Required packages (install using `pip install -r requirements.txt`):
  - pandas
  - numpy
  - matplotlib

## Data Format

The system expects CSV files with the following columns:
- Timestamp
- Bid
- Ask

## Usage

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your data files in the appropriate directories:
   - UEC.csv in `Data 1/UEC Data/`
   - SOBER.csv in `Data 1/SOBER Data/`

3. Run the backtester:
   ```bash
   python backtester.py
   ```

4. View the results:
   - Performance comparison plot: `performance_comparison.png`
   - Individual strategy plots:
     - `performance_moving_average.png`
     - `performance_mean_reversion.png`
     - `performance_breakout.png`
     - `performance_momentum.png`
   - Console output with performance metrics

## Strategy Details

### Moving Average Crossover
- Short-term MA: 20 periods
- Long-term MA: 50 periods
- Buffer: 0.1% to avoid whipsaws

### Mean Reversion
- Moving average window: 20 periods
- Deviation threshold: 1%

### Breakout
- Lookback period: 10 periods
- Trades on price breaking above recent high or below recent low

### Momentum
- Lookback period: 5 periods
- Threshold: 0.5% momentum

## Configuration

You can modify the following parameters in `backtester.py`:
- Initial cash: `initial_cash` (default: 100,000)
- Position limit: `position_limit` (default: 100)
- Transaction fee: `fee` (default: 0.2%)

## Output

The system generates:
1. Performance plots for each strategy
2. A comparison plot of all strategies
3. Console output with detailed performance metrics including:
   - Initial portfolio value
   - Final portfolio value
   - Total return
   - Number of trades 