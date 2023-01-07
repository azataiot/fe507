# fe507 / test.py
# Created by azat at 5.01.2023
import logging

from matplotlib import pyplot as plt

logging.basicConfig(level=logging.ERROR)  # default logging for other libraries
logging.getLogger('fe507').level = logging.DEBUG
from fe507 import Data, Collection, DataSource, settings, CollectionGroup

settings.data_dir = "./data"

exchange_rates = Data(DataSource.EXCHANGE_RATES)
sp500 = Data(DataSource.SP500)
b100 = Data(DataSource.BIST100)
ball = Data(DataSource.BISTALL)
gold = Data(DataSource.GOLD)
btc = Data(DataSource.BTCETH)

year_range = {
    "from_year": 2015,
    "to_year": 2022
}

csp = Collection(sp500.data, name="S&P500").get_range(**year_range).get(on='Index')
cb1 = Collection(b100.data, name="BIST100").get_range(**year_range).get(on='Index')
cba = Collection(ball.data, name="BIST ALL").get_range(**year_range).get(on='Index')
cg = Collection(gold.data, name='Gold').get_range(**year_range).get(on='Price ($/t oz)')
cbtc = Collection(btc.data, name="Bitcoin").get_range(**year_range).get(on='Bitcoin')

g = CollectionGroup([csp, cb1, cba, cg, cbtc])
# print(g[0])
corr = g.correlation
corr.style.background_gradient(cmap='coolwarm')
plt.matshow(corr)
plt.show()
