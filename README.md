# gabagool

Polymarket 15min crypto arbitrage bot.

The bot works – the biggest issue is the response time for both orders to be executed. In demo mode, everything works great, but the delays in this market are problematic.

## Reinforcement learning entry/hedge model

`ml/rl_entry_hedge.py` provides a lightweight Q-learning baseline for deciding when to enter and hedge using:

- Polymarket top-of-book bid/ask
- Chainlink price
- Binance price
- Binance CVD

The trainer expects a CSV with these columns:

```
timestamp,polymarket_bid,polymarket_ask,chainlink_price,binance_price,binance_cvd
```

Example usage:

```
python ml/rl_entry_hedge.py data/market_snapshots.csv --episodes 50
```

Notebook demo (executed with sample data):

```
ml/rl_entry_hedge_demo.ipynb
```
