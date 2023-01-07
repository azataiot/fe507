# src / io.py
# Created by azat at 1.01.2023


from datetime import date, datetime
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from fe507 import settings
from fe507.utils import log


# link to the data folder.
# data_dir = Path("../data")


class DataSource(Enum):
    SP500 = 'SP500.xlsx'
    BIST100 = 'BIST100.xlsx'
    BISTALL = 'BISTALL.xlsx'
    GOLD = 'Gold.xlsx'
    BTCETH = 'BTCETH.xlsx'
    EXCHANGE_RATES = 'EXCHANGE.xlsx'
    INFLATION = 'TLINF.xlsx'
    INTEREST_RATE = 'TLDEPO.xlsx'


class TimeFrameType(Enum):
    DAY = 'D'
    WEEK = 'W'
    MONTH = 'M'


class RateOfReturnType(Enum):
    SIMPLE = 0
    LOGARITHMIC = 1
    GEOMETRIC = 2


def _calc_iqr(x):
    return np.subtract(*np.percentile(x, [75, 25]))


class Data:
    """Single DataFrame Object with all possible statistical, probabilistic and mathematical
    analyzing functions with one call
    :param source: One of the DataSource Enum. Possible values are:
           SP500, BIST100, BISTALL, GOLD, BTCETH, EXCHANGE_RATES, INFLATION, INTEREST_RATE
    """
    data: pd.DataFrame = None
    data_dir: Path | str

    def __new__(cls, *args, **kwargs):
        cls.data_dir = settings.data_dir
        log.debug(f'Set up "data_dir" as: {cls.data_dir}')
        return super().__new__(cls)

    def __init__(self, source: DataSource):
        self._filtered_df = None
        dir_path = str(self.data_dir) + '/' + source.value
        self.source = Path(dir_path)
        log.debug(f'working on file: {self.source.resolve()}')
        # read the csv file
        self._dataframe = pd.read_excel(self.source)
        log.debug(f"read dataframe from excel as: \n{self._dataframe.columns.tolist()}")
        # remove duplicated rows
        self._dataframe = self._dataframe.drop_duplicates()
        # setup some source file specific files, such as each csv has different filed name for "date"
        self._init_source_specific_fields(source)
        # adding data and quarter fields
        self._add_date_filed(self._date_column)
        # changing the data types to numerical data types and setting the date as index
        self.data = self._dataframe.astype(
            dict([(v, 'float') for v in self.numerical_columns])
        )
        self.shape = self._dataframe.shape
        self.info = self._dataframe.info

    def _init_source_specific_fields(self, source: DataSource):
        log.debug(f"handling source specific fields for {source}")
        match source:
            case DataSource.SP500:
                self.numerical_columns = ['Index', 'Market cap (m$)']
                self._date_column = 'Date'
                self.name = 'sp500'
                # self._normalize_strings()
            case DataSource.BIST100:
                self.numerical_columns = ['Index', 'Market cap (m TL)']
                self._date_column = 'Date'
                self.name = 'bist100'
            case DataSource.BISTALL:
                self.numerical_columns = ['Index', 'Market cap (m TL)']
                self._date_column = 'Code'
                self.name = 'bistall'
            case DataSource.GOLD:
                self.numerical_columns = ['Price ($/t oz)']
                self._date_column = 'Date'
                self.name = 'gold'
            case DataSource.BTCETH:
                self.numerical_columns = ['Bitcoin', 'Ethereum']
                self._date_column = 'Date'
                self.name = 'btceth'
            case DataSource.EXCHANGE_RATES:
                self.numerical_columns = ['TL/USD', 'TL/Euro']
                self._date_column = 'Date'
                self.name = 'exchange_rate'
            case DataSource.INTEREST_RATE:
                self.numerical_columns = []
                self._date_column = None
                self.name = 'interest_rate'
            case DataSource.INFLATION:
                self.numerical_columns = []
                self._date_column = None
                self.name = 'inflation'

    def _set_index(self):
        if self._date_column is None:
            pass
        self.data.set_index('date')

    def _has_column(self, column: str) -> bool:
        ret = column in self._dataframe.columns
        return ret

    def _add_date_filed(self, reference_field: str | None = 'Date'):
        if reference_field is None:
            log.debug(f"reference field is {reference_field}, skipping (None)")
            pass
        elif not self._has_column(reference_field):
            raise ValueError(f"{reference_field} dose not exists in {self._dataframe.info()}")
        else:
            self._dataframe['date'] = pd.to_datetime(self._dataframe[f"{reference_field}"], dayfirst=True)
            self._dataframe['quarter'] = self._dataframe['date'].dt.to_period('Q')

    def numerical_data(self, exclude_date: bool = False):
        if exclude_date:
            return self.data[self.numerical_columns]
        return self.data[['date'] + self.numerical_columns]

    def _normalize_strings(self):
        for each in self.numerical_columns:
            self._dataframe[each] = self._dataframe[each].apply(lambda x: x.strip('').strip(' ').replace(',', '.'))

    def first(self):
        return self.data.iloc[0]

    def last(self):
        return self.data.iloc[-1]

    def get(self, on_date: date | str):
        _tmp = self.numerical_data()
        self._filtered_df = _tmp.loc[
            _tmp['date'] == on_date
            ]
        return self._filtered_df

    def get_column(self, column_name: str, timeframe: TimeFrameType | None = None):
        if column_name in self.numerical_columns:
            _df = self.numerical_data().set_index('date')[column_name]
            if timeframe is not None:
                match timeframe:
                    case TimeFrameType.DAY:
                        pass
                    case TimeFrameType.WEEK:
                        _df = _df.resample('W-MON').ffill()
                    case TimeFrameType.MONTH:
                        _df = _df.resample('MS').ffill()
            return _df
        else:
            raise ValueError(f"{column_name} dose not exists in {self.numerical_data().info}")

    def get_range(self, from_year: int | str, to_year: int | str,
                  from_month: int | str = '01', to_month: int | str = '12',
                  from_day: int | str = '01', to_day: int | str = '31'
                  ):
        _tmp = self.numerical_data()
        _tmp_date_from = datetime(year=int(from_year), month=int(from_month), day=int(from_day))
        _tmp_date_to = datetime(year=int(to_year), month=int(to_month), day=int(to_day))
        self._filtered_df = _tmp.loc[
            (_tmp['date'] >= _tmp_date_from) & (_tmp['date'] <= _tmp_date_to)
            ]
        return self._filtered_df

    def max(self):
        return self.numerical_data().max()

    def min(self):
        return self.numerical_data().min()

    def mean(self):
        _tmp = self.numerical_data()
        return _tmp.mean(numeric_only=True)

    def median(self):
        _tmp = self.numerical_data()
        return _tmp.median(numeric_only=True)

    def geometric_mean(self):
        _tmp = self.numerical_data(exclude_date=True)
        ret = _tmp[_tmp.columns.tolist()].apply(lambda x: stats.gmean(x))
        return ret

    def variance(self):
        _tmp = self.numerical_data()
        return _tmp.var(numeric_only=True)

    def standard_deviation(self):
        _tmp = self.numerical_data()
        return _tmp.std(numeric_only=True)

    def inter_quartile_range(self):
        ret = self.data[self.numerical_columns].apply(_calc_iqr)
        return ret

    def skewness(self):
        _tmp = self.numerical_data()
        return _tmp.skew(numeric_only=True)

    def kurtosis(self):
        _tmp = self.numerical_data()
        return _tmp.kurtosis(numeric_only=True)

    def autocorrelation(self):
        _tmp = self.numerical_data(exclude_date=True)
        return _tmp[_tmp.columns.tolist()].apply(lambda x: x.autocorr())

    def rate_of_return(self, mode: RateOfReturnType | None = RateOfReturnType.SIMPLE,
                       timeframe: TimeFrameType | None = TimeFrameType.DAY):
        _tmp = self.numerical_data()
        # _tmp['date'] = pd.to_datetime(_tmp['date'])
        _tmp = _tmp.set_index('date')
        # prepare the data samples with given frequency
        sample = _tmp
        match timeframe:
            case TimeFrameType.DAY:
                pass
            case TimeFrameType.WEEK:
                sample = _tmp.resample('W-MON').ffill()
            case TimeFrameType.MONTH:
                sample = _tmp.resample('MS').ffill()
        match mode:
            case RateOfReturnType.SIMPLE:
                # return the simple rate of return for the given sample
                return sample.pct_change()
            case RateOfReturnType.LOGARITHMIC:
                for each in sample.columns.tolist():
                    sample[each] = np.log(
                        sample[each] / sample[each].shift(1)
                    )
                return sample
            case RateOfReturnType.GEOMETRIC:
                # first calculate geometric mean for each row
                raise NotImplementedError(
                    "Calculating rate of return for this type is not supported in this version yet.")

# sp500 = Data(DataSource.SP500)
# bist100 = Data(DataSource.BIST100)
# bistall = Data(DataSource.BISTALL)
# gold = Data(DataSource.GOLD)
# btceth = Data(DataSource.BTCETH)
# exchange_rates = Data(DataSource.EXCHANGE_RATES)
# interest_rates = Data(DataSource.INTEREST_RATE)
# inflation = Data(DataSource.INFLATION)

# print('mean\n', sp500.mean())
# print('median\n', sp500.median())
# print('variance\n', sp500.variance())
# print('standard_deviation\n', sp500.standard_deviation())
# print('max\n', sp500.max())
# print('min\n', sp500.min())
# print('iqr\n', sp500.inter_quartile_range())
# print('skewness\n', sp500.skewness())
# print('kurtosis\n', sp500.kurtosis())
# print('autocorrelation\n', sp500.autocorrelation())
# print('geometric mean', sp500.geometric_mean())
# print('rate of return d', sp500.rate_of_return(mode=RateOfReturnType.LOGARITHMIC, timeframe=TimeFrameType.DAY))
# common_plot_options = {
#     'figsize': (16, 12),
# }
# spindex = sp500.get_column('Index', timeframe=TimeFrameType.MONTH)
# spindex.plot(x="index", y="Index", title='S&P 500 Weekly', **common_plot_options)
# plt.show()

# sp500 = Data(DataSource.SP500)
