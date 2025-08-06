from lumibot.brokers import Alpaca 
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime

API_KEY = ""
API_KEY_SECRET = ""
BASE_URL = "" 

ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_KEY_SECRET": API_KEY_SECRET,
    "PAPER": True
}

class MLTrader(Strategy) :
    def initialize(self, symbol:str="SPY", cash_at_risk:float = .5):
        self.symbol = symbol
        self.sleeptime = "24H"
        self.last_trade = None 
        self.cash_at_risk = cash_at_risk
        

    def position_sizing(self) :
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round((cash * self.cash_at_risk) / last_price,0)
        return quantity 

        
    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        if cash > last_price :
            if self.last_trade is None :
                order = self.create_order(
                    self.symbol,
                    10,
                    "buy",
                    type="market",
                )
                self.submit_order(order)
                self.last_trade = "buy"

start_date = datetime(2023,12,15)
end_date = datetime(2023,12,31)
broker = Alpaca(ALPACA_CREDS)
strategy = MLtrader(name='mlstrat', broker = broker, parameters={"symbol": "SPY", "cash_at_risk": 0.5})

strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters = {} 
)