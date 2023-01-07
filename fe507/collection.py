# fe507 / collection.py
# Created by azat at 5.01.2023
from datetime import date, datetime
from enum import auto

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy import stats
from strenum import StrEnum

from fe507 import Data
from fe507.utils import log


class RateOfReturnType(StrEnum):
    """
    Rate of return type.
    """
    SIMPLE = auto()
    LOGARITHMIC = auto()


SIMPLE = RateOfReturnType.SIMPLE
LOGARITHMIC = RateOfReturnType.LOGARITHMIC


class DataResampleFrequency(StrEnum):
    """
    Data resampling frequency.
    """
    DAY = auto()
    WEEK = auto()
    MONTH = auto()


DAY = DataResampleFrequency.DAY
WEEK = DataResampleFrequency.WEEK
MONTH = DataResampleFrequency.MONTH


class CurrencyType(StrEnum):
    """
    Currency type.
    """
    USD = auto()
    EUR = auto()
    TRY = auto()


USD = CurrencyType.USD
EUR = CurrencyType.EUR
TRY = CurrencyType.TRY


def _calc_iqr(x):
    return np.subtract(*np.percentile(x, [75, 25]))


class Collection:
    _numerics_data_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    def __init__(self, item: DataFrame | Series, name: str | None = None,
                 currency: CurrencyType | None = CurrencyType.USD):
        # get the dataframe
        self._dataframe: DataFrame = item
        # remove duplicated
        self._dataframe = self._dataframe.drop_duplicates()
        self._dataframe = self._dataframe.dropna()
        # remove unnamed columns (if it is a DataFrame)
        if isinstance(self._dataframe, DataFrame):
            self._dataframe = self._dataframe.loc[:, ~self._dataframe.columns.str.contains("Unnamed")]
            self.columns = self._dataframe.columns.tolist()
            if 'date' in self.columns:
                self._dataframe = self._dataframe.set_index(self._dataframe['date'])
        self._numeric_data_frame = self._dataframe.select_dtypes(include=self._numerics_data_types)
        self.data_type = type(item)
        self.shape = item.shape
        self.name = name
        self.currency = currency

    def __repr__(self):
        return f'Collection({self.name})'

    def __call__(self, *args, **kwargs) -> DataFrame | Series:
        return self._dataframe

    @property
    def data(self):
        """Return the dataframe as a new collection object"""
        return Collection(self._dataframe)

    @property
    def count(self):
        return self._dataframe.shape[0]

    @property
    def raw(self):
        return self.data()

    @property
    def first(self) -> Series:
        return self._dataframe.iloc[0]

    @property
    def last(self) -> Series:
        return self._dataframe.iloc[-1]

    def get(self, by: date | str | None = None, on: str | None = None):
        ret = None
        if by is None:
            if on is None:
                ret: DataFrame | Series = self._dataframe.to_frame()
            else:
                # on is not none, select the column
                ret = self._dataframe[on].to_frame()
        elif by is not None:
            if on is None:
                try:
                    ret = self._dataframe.loc[(self._dataframe['date'] == by)]
                except KeyError:
                    ret = self._dataframe.loc[(self._dataframe.index == by)].to_frame()
            else:
                ret = self._dataframe.loc[(self._dataframe[on] == by)]
                log.debug(f'query results {ret}, count:{ret.shape[0]}')
        return Collection(ret, name=self.name, currency=self.currency)

    def get_range(self, from_year: int | str, to_year: int | str,
                  from_month: int | str = '01', to_month: int | str = '12',
                  from_day: int | str = '01', to_day: int | str = '31'
                  ):
        _tmp = self._dataframe
        _tmp_date_from = datetime(year=int(from_year), month=int(from_month), day=int(from_day))
        _tmp_date_to = datetime(year=int(to_year), month=int(to_month), day=int(to_day))
        ret = _tmp.loc[
            (_tmp['date'] >= _tmp_date_from) & (_tmp['date'] <= _tmp_date_to)
            ]
        return Collection(ret, name=self.name, currency=self.currency)

    def frequency(self, freq: DataResampleFrequency | None = DAY):
        _tmp = self._dataframe
        # prepare the data samples with given frequency
        sample = _tmp
        match freq:
            case DataResampleFrequency.DAY:
                pass
            case DataResampleFrequency.WEEK:
                sample = _tmp.resample('W-MON').ffill()
            case DataResampleFrequency.MONTH:
                sample = _tmp.resample('MS').ffill()
        return Collection(sample, name=self.name, currency=self.currency)

    @property
    def max(self) -> Series:
        return self._dataframe.max()

    @property
    def min(self) -> Series:
        return self._dataframe.min()

    @property
    def mean(self) -> Series:
        return self._dataframe.mean(numeric_only=True)

    @property
    def median(self) -> Series:
        return self._dataframe.median(numeric_only=True)

    @property
    def geometric_mean(self) -> Series:
        _tmp = self._numeric_data_frame
        ret = _tmp[_tmp.columns.tolist()].apply(lambda x: stats.gmean(x))
        return ret

    @property
    def variance(self):
        return self._dataframe.var(numeric_only=True)

    @property
    def standard_deviation(self):
        return self._dataframe.std(numeric_only=True)

    @property
    def inter_quartile_range(self):
        _tmp = self._numeric_data_frame
        ret = _tmp[_tmp.columns.tolist()].apply(_calc_iqr)
        return ret

    @property
    def skewness(self):
        return self._dataframe.skew(numeric_only=True)

    @property
    def kurtosis(self):
        return self._dataframe.kurtosis(numeric_only=True)

    @property
    def autocorrelation(self):
        _tmp = self._numeric_data_frame
        ret = _tmp[_tmp.columns.tolist()].apply(lambda x: x.autocorr())
        return ret

    def ror(self, method: RateOfReturnType | None = LOGARITHMIC):
        ret = None
        _tmp = self._numeric_data_frame
        match method:
            case RateOfReturnType.SIMPLE:
                ret = _tmp.pct_change()
            case RateOfReturnType.LOGARITHMIC:
                for each in _tmp.columns.tolist():
                    _tmp[each] = np.log(
                        _tmp[each] / _tmp[each].shift(1)
                    )
                ret = _tmp
        return Collection(ret, name=self.name, currency=self.currency)


class CurrencyAwareCollection(Collection):
    def __init__(self, data: Data, exchange_rate: Data | None = None, currency: CurrencyType | None = CurrencyType.USD):
        data.data = data.data.merge(exchange_rate.data, how='left')
        if currency == CurrencyType.TRY:
            _tmp = data.data.select_dtypes(include=self._numerics_data_types)
            for each in _tmp.columns.tolist():
                data.data[each + 'USD'] = _tmp[each] / _tmp['TL/USD']
        super().__init__(data.data, currency=currency)


class CollectionGroup:
    _dataframe: DataFrame = None

    def __init__(self, _: [Collection | CurrencyAwareCollection]):
        self._collections = _
        self._dfs = [each.raw for each in self._collections]
        self._keys = [each.name for each in self._collections]
        self._merge()

    def __repr__(self):
        return f'{self.__class__.__name__}({self._keys})'

    def __call__(self, *args, **kwargs):
        return self._dataframe

    def _merge(self):
        # self._dataframe: DataFrame = self._collections[0].raw
        self._dataframe: DataFrame = pd.concat(self._dfs, axis=1)
        self._dataframe.columns = self._keys

    def __iter__(self):
        return iter(self._collections)

    def __len__(self):
        return len(self._collections)

    def __getitem__(self, key):
        return self._collections[key]

    @property
    def count(self):
        return len(self._collections)

    @property
    def data(self):
        return self._dataframe

    @property
    def covariance(self):
        return self._dataframe.cov().round(5)

    @property
    def correlation(self):
        return self._dataframe.corr()

    @property
    def max(self) -> Series:
        return self._dataframe.max()

    @property
    def min(self) -> Series:
        return self._dataframe.min()

    @property
    def mean(self) -> Series:
        return self._dataframe.mean(numeric_only=True)

    @property
    def median(self) -> Series:
        return self._dataframe.median(numeric_only=True)

    @property
    def geometric_mean(self) -> Series:
        _tmp = self._dataframe
        ret = _tmp[_tmp.columns.tolist()].apply(lambda x: stats.gmean(x))
        return ret

    @property
    def variance(self):
        return self._dataframe.var(numeric_only=True)

    @property
    def standard_deviation(self):
        return self._dataframe.std(numeric_only=True)

    @property
    def inter_quartile_range(self):
        _tmp = self._dataframe
        ret = _tmp[_tmp.columns.tolist()].apply(_calc_iqr)
        return ret

    @property
    def skewness(self):
        return self._dataframe.skew(numeric_only=True)

    @property
    def kurtosis(self):
        return self._dataframe.kurtosis(numeric_only=True)

    @property
    def autocorrelation(self):
        _tmp = self._dataframe
        ret = _tmp[_tmp.columns.tolist()].apply(lambda x: x.autocorr())
        return ret
