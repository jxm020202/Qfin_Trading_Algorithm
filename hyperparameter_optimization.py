from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from itertools import product
import main as template
from copy import deepcopy
import time
from optimization_backtester import run_optimization_backtest

# Define parameter ranges for optimization
PARAMETER_RANGES = {
    "UEC": {  # Changed to match Small dataset filename
        "threshold": {
            "start": 0.3,  # Much smaller starting value
            "end": 0.9,     # Much larger ending value
            "step": 0.1    # Smaller step size for finer granularity
        },
        "window_size": {
            "start": 40,     # Smaller window sizes
            "end": 50,      # Larger window sizes
            "step": 5       # Step size of 5
        },
        "position_small": {
            "start": 1,
            "end": 5,      # Increased range
            "step": 2
        },
        "position_large": {
            "start": 10,
            "end": 18,      # Increased range
            "step": 4
        },
        "min_trade_interval": {
            "start": 15,     # Smaller intervals
            "end": 15,      # Larger intervals
            "step": 1
        }
    },
    "SOBER": {  # Changed to match Small dataset filename
        "threshold": {
            "start": 3.0,
            "end": 4.5,
            "step": 0.5
        },
        "window_size": {
            "start": 20,
            "end": 50,
            "step": 15
        },
        "position_small": {
            "start": 1,
            "end": 5,
            "step": 1
        },
        "position_large": {
            "start": 5,
            "end": 50,
            "step": 5
        },
        "min_trade_interval": {
            "start": 5,
            "end": 30,
            "step": 5
        }
    }
}

def generate_parameter_combinations(product_name: str) -> List[Dict]:
    """Generate all possible combinations of parameters for a given product."""
    ranges = PARAMETER_RANGES[product_name]
    param_names = list(ranges.keys())
    
    # Generate ranges for each parameter
    param_values = []
    for param in param_names:
        start = ranges[param]["start"]
        end = ranges[param]["end"]
        step = ranges[param]["step"]
        param_values.append(np.arange(start, end + step, step))
    
    # Generate all combinations
    combinations = []
    for values in product(*param_values):
        combination = dict(zip(param_names, values))
        combinations.append(combination)
    
    return combinations

def backtest_with_parameters(parameters: Dict, product_name: str) -> float:
    """Run backtest with given parameters and return PnL."""
    # Modify template parameters
    template.team_algorithm.THRESHOLDS[product_name] = parameters["threshold"]
    template.team_algorithm.WINDOW_SIZE[product_name] = parameters["window_size"]
    template.team_algorithm.POSITION_SMALL[product_name] = parameters["position_small"]
    template.team_algorithm.POSITION_LARGE[product_name] = parameters["position_large"]
    template.team_algorithm.MIN_TRADE_INTERVAL[product_name] = parameters["min_trade_interval"]
    
    # Reset algorithm state
    template.team_algorithm.positions = {}
    template.team_algorithm.price_data = {}
    template.team_algorithm.last_trade = {}
    
    # Run optimization backtest
    pnl = run_optimization_backtest([product_name])
    
    return pnl

def optimize_parameters():
    """Run optimization for all products and return top 100 configurations."""
    results = {}
    
    for product_name in PARAMETER_RANGES.keys():
        print(f"\nOptimizing parameters for {product_name}...")
        combinations = generate_parameter_combinations(product_name)
        product_results = []
        
        total_combinations = len(combinations)
        print(f"Testing {total_combinations} combinations...")
        
        for i, params in enumerate(combinations):
            if i % 100 == 0:  # Progress update every 10 combinations
                print(f"Progress: {i}/{total_combinations} ({(i/total_combinations)*100:.1f}%)")
            
            pnl = backtest_with_parameters(params, product_name)
            product_results.append({
                "parameters": params,
                "pnl": pnl
            })
        
        # Sort by PnL and get top 100
        product_results.sort(key=lambda x: x["pnl"], reverse=True)
        results[product_name] = product_results[:100]
        
        # Print results immediately after each product
        print(f"\n=== Top 100 Results for {product_name} ===")
        print("Rank | PnL | Parameters")
        print("-" * 80)
        for i, result in enumerate(product_results[:100], 1):
            params = result["parameters"]
            pnl = result["pnl"]
            print(f"{i:4d} | {pnl:8.2f} | Threshold: {params['threshold']:.2f}, "
                  f"Window: {params['window_size']}, "
                  f"Small: {params['position_small']}, "
                  f"Large: {params['position_large']}, "
                  f"Interval: {params['min_trade_interval']}")
    
    return results

def print_results(results: Dict):
    """Print the top 100 results for each product."""
    # This function is now empty since we print results during optimization
    pass

if __name__ == "__main__":
    start_time = time.time()
    results = optimize_parameters()
    end_time = time.time()
    
    print(f"\nOptimization completed in {end_time - start_time:.2f} seconds") 