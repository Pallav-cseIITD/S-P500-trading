import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import logging 

# Parameters
START_DATE = "2014-01-01"
END_DATE = "2024-12-31"
LOOKBACK_DAYS = 252
TOP_PERCENT = 0.2
MIN_PRICE = 5.0

def load_tickers(path="sp500_tickers.csv"):
    df = pd.read_csv(path)
    return df["Symbol"].str.strip().str.replace(".", "-").tolist()

def download_prices(tickers, start, end):
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Adj Close"]
    else:
        prices = data[["Adj Close"]]
        prices.columns = tickers

    return prices

def calculate_returns(prices):
    shifted = prices.shift(1)
    daily_returns = np.log(prices / shifted)
    daily_returns = daily_returns.dropna(how="all")
    return daily_returns

def max_drawdown(returns):
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return -drawdown.min()

def performance_stats(returns):
    returns = returns.dropna()
    daily_return = returns.mean()
    daily_vol = returns.std()
    
    annual_return = (1 + daily_return) ** 252 - 1
    annual_vol = daily_vol * np.sqrt(252)

    if daily_vol > 0:
        sharpe = (daily_return / daily_vol) * np.sqrt(252)
    else:
        sharpe = 0.0

    mdd = max_drawdown(returns)
    
    return {
        "Daily Return": daily_return,
        "Annual Return": annual_return,
        "Annual Volatility": annual_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": mdd
    }

def run_strategy(prices):

    # remove the cheap stocks
    prices = prices.where(prices >= MIN_PRICE)

    returns = calculate_returns(prices)
    
    trailing_return = prices / prices.shift(LOOKBACK_DAYS) - 1
    
    strategy_returns = []

    for date in returns.index:
        scores = trailing_return.loc[date].dropna()
        daily_returns = returns.loc[date].dropna()
        
        valid_stocks = scores.index.intersection(daily_returns.index)
        
        if len(valid_stocks) == 0:
            strategy_returns.append(0.0)
            continue
        
        scores = scores[valid_stocks]
        
        # Select top performers
        cutoff = scores.quantile(1 - TOP_PERCENT)
        winners = scores[scores >= cutoff].index
        
        if len(winners) == 0:
            strategy_returns.append(0.0)
            continue
        
        daily_ret = daily_returns[winners].mean()
        if pd.isna(daily_ret):
            daily_ret = 0.0
        strategy_returns.append(daily_ret)

    return pd.Series(strategy_returns, index=returns.index, name="strategy")

def regression_analysis(strategy_returns, market_returns):
    # Run regression: strategy = alpha + beta * market
    df = pd.concat([strategy_returns, market_returns], axis=1, join="inner").dropna()
    df.columns = ["strategy", "market"]

    X = sm.add_constant(df["market"])
    y = df["strategy"]

    model = sm.OLS(y, X).fit()

    alpha = model.params["const"]
    beta = model.params["market"]
    r2 = model.rsquared
    t_alpha = model.tvalues["const"]
    t_beta = model.tvalues["market"]

    results = {
        "Alpha (daily)": alpha,
        "Beta": beta,
        "R-squared": r2,
        "T-stat Alpha": t_alpha,
        "T-stat Beta": t_beta,
    }
    return results

def main():

    print("Loading S&P 500 tickers...")
    tickers = load_tickers()
    
    prices = download_prices(tickers, START_DATE, END_DATE)
    
    print("Downloading SPY for benchmark...")
    spy = download_prices(["SPY"], START_DATE, END_DATE).squeeze()
    spy_returns = calculate_returns(spy.to_frame())["SPY"]
    
    print("\nRunning growth strategy...")
    strategy_returns = run_strategy(prices)
    
    print("\n" + "="*50)
    print("PERFORMANCE STATISTICS")
    print("="*50)
    
    stats = performance_stats(strategy_returns)
    for name, value in stats.items():
        print(f"{name:20s}: {value:>10.4f}")
    
    print("\n" + "="*50)
    print("REGRESSION vs SPY")
    print("="*50)
    
    reg = regression_analysis(strategy_returns, spy_returns)
    for name, value in reg.items():
        print(f"{name:20s}: {value:>10.4f}")
    
    print("\n" + "="*50)
    
    # Save results
    equity_curve = (1 + strategy_returns).cumprod()
    results = pd.DataFrame({
        "returns": strategy_returns,
        "equity": equity_curve
    })
    results.to_csv("growth_strategy_results.csv")
    print("\nResults saved to 'growth_strategy_results.csv'")

if __name__ == "__main__":
    main()