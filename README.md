# FE507

FE507 is a simple yet very powerful, 'batteries included' intuitive package for data analysing.

## How to use?

1. import the `settings` model to configure the `data_dir` where all of your data is located. (Notice: FE507 expects all
   your data to be in `csv` format.)
   ```python
   from fe507 import settings
   settings.data_dir = "./data/"  # you csv files is stored in the directory named `data` in your current directory
   ```
2. Import base classes from the package
   ```python
   from fe507 import Data, DataSource, RateOfReturnType, TimeFrameType
  ```
3. Enjoy.
# fe507

  ```

## Basic Structures

![image-20230109214822500](https://raw.githubusercontent.com/azataiot/images/master/2023/01/upgit_20230109_1673290215.png)

![image-20230109214829790](https://raw.githubusercontent.com/azataiot/images/master/2023/01/upgit_20230109_1673290221.png)

![image-20230109214836543](https://raw.githubusercontent.com/azataiot/images/master/2023/01/upgit_20230109_1673290230.png)

## Examples:

```python
import plotly.express as px
import matplotlib.pyplot as plt

sp500 = Data(DataSource.SP500)
bist100 = Data(DataSource.BIST100)
bistall = Data(DataSource.BISTALL)
gold = Data(DataSource.GOLD)
btceth = Data(DataSource.BTCETH)
exchange_rates = Data(DataSource.EXCHANGE_RATES)

year_range = YearRange(from_year=2015, to_year=2022)

sp = Collection(sp500.data, name="S&P500", currency=USD).get_range(year_range.from_year,
                                                                   year_range.to_year).get(on="Index")
b1 = CurrencyAwareCollection(bist100, exchange_rates, name="BIST100", currency=TRY).get_range(year_range.from_year,
                                                                                              year_range.to_year).get(
    on="IndexUSD")
ba = CurrencyAwareCollection(bistall, exchange_rates, name="BISTALL", currency=TRY).get_range(year_range.from_year,
                                                                                              year_range.to_year).get(
    on="IndexUSD")
gd = Collection(gold.data, name="Gold", currency=USD).get_range(year_range.from_year,
                                                                year_range.to_year).get(on='Price ($/t oz)')
btc = Collection(btceth.data, name="Bitcoin", currency=USD).get_range(year_range.from_year,
                                                                      year_range.to_year).get(on='Bitcoin')

g = CollectionGroup([sp, b1, ba, gd, btc])

ror_sp_w = sp.frequency(WEEK).ror()
ror_b1_w = b1.frequency(WEEK).ror()
ror_ba_w = ba.frequency(WEEK).ror()
ror_gd_w = gd.frequency(WEEK).ror()
ror_btc_w = btc.frequency(WEEK).ror()

g_ror_d = CollectionGroup([ror_sp_w, ror_b1_w, ror_ba_w, ror_gd_w, ror_btc_w])
```

