import yfinance as yf

ticker = yf.Ticker("^BSESN")
df = ticker.history(period="5y")
print(df)
