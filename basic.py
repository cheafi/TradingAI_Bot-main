from src.strategies.scalping import ScalpingStrategy
import pandas as pd

# Example quick test
df = pd.DataFrame({
    'close': [100, 101, 102, 99, 98, 100, 103, 105]
})
bot = ScalpingStrategy(capital=10000)
signals = bot.generate_signals(df)
print(signals)
