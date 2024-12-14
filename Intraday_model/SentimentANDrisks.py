# Full Python Code for Determining Stock Market Sentiment

def determine_sentiment(data):
    """
    Determines Indian stock market sentiment based on specified conditions.

    Args:
    data (dict): A dictionary containing key metrics for analysis.

    Returns:
    str: 'Positive', 'Negative', or 'Neutral' based on the input data.
    """
    # Extract data points from the input dictionary
    nifty_change = data.get('nifty_change', 0)
    india_vix = data.get('india_vix', 0)
    advances = data.get('advances', 0)
    declines = data.get('declines', 0)
    inflation_rate = data.get('inflation_rate', 0)
    interest_rate = data.get('interest_rate', 0)
    usd_inr = data.get('usd_inr', 0)
    earnings_growth = data.get('earnings_growth', 0)
    fii_net_investment = data.get('fii_net_investment', 0)
    crude_oil_price = data.get('crude_oil_price', 0)
    geopolitical_risk_index = data.get('geopolitical_risk_index', 0)
    budget_deficit = data.get('budget_deficit', 0)

    # Negative sentiment conditions
    if (
        nifty_change < -0.5 or
        india_vix > 20 or
        advances < declines or
        inflation_rate > 6 or
        interest_rate > 6.5 or
        usd_inr > 85 or
        earnings_growth < 0 or
        fii_net_investment < 0 or
        crude_oil_price > 90 or
        geopolitical_risk_index > 50 or  # Example threshold
        budget_deficit > 5
    ):
        return "Negative"

    # Positive sentiment conditions
    elif (
        nifty_change > 0.5 or
        india_vix < 15 or
        advances > declines or
        inflation_rate < 4 or
        interest_rate < 5.5 or
        usd_inr < 80 or
        earnings_growth > 5 or
        fii_net_investment > 0 or
        crude_oil_price < 70 or
        geopolitical_risk_index < 30 or  # Example threshold
        budget_deficit < 3.5
    ):
        return "Positive"

    # Neutral sentiment if neither positive nor negative conditions are met
    else:
        return "Neutral"


# Example Usage
if __name__ == "_main_":
    # Input data example (replace these with real data)
    market_data = {
        "nifty_change": 0.7,  # NIFTY50 percentage change
        "india_vix": 12,      # Volatility Index
        "advances": 1200,     # Advancing stocks
        "declines": 800,      # Declining stocks
        "inflation_rate": 3.8, # Inflation rate in percentage
        "interest_rate": 5.0,  # Interest rate in percentage
        "usd_inr": 78,        # USD to INR exchange rate
        "earnings_growth": 6, # Quarterly earnings growth in percentage
        "fii_net_investment": 1000, # FII net investment in crores
        "crude_oil_price": 65, # Crude oil price in USD
        "geopolitical_risk_index": 25, # Geopolitical risk index
        "budget_deficit": 3.2, # Budget deficit as a percentage of GDP
    }

    # Determine and print sentiment
    sentiment = determine_sentiment(market_data)
    print(f"The market sentiment is: {sentiment}")