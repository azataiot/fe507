# fe507 / df.py
# Created by azat at 6.01.2023
from datetime import datetime, date
from enum import auto
from strenum import StrEnum
import numpy as np
from pandas import DataFrame, Series

from fe507 import log

_numerics_data_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']


def get(df: DataFrame, by: date | str | None = None, on: str | None = None):
    ret = None
    if by is None:
        if on is None:
            ret: DataFrame | Series = df.to_frame()
        else:
            # on is not none, select the column
            ret = df[on].to_frame()
    elif by is not None:
        if on is None:
            try:
                ret = df.loc[(df['date'] == by)]
            except KeyError:
                ret = df.loc[(df.index == by)].to_frame()
        else:
            ret = df.loc[(df[on] == by)]
            log.debug(f'query results {ret}, count:{ret.shape[0]}')
    return ret


def get_range(df: DataFrame, from_year: int | str, to_year: int | str,
              from_month: int | str = '01', to_month: int | str = '12',
              from_day: int | str = '01', to_day: int | str = '31'
              ):
    _tmp = df
    _tmp_date_from = datetime(year=int(from_year), month=int(from_month), day=int(from_day))
    _tmp_date_to = datetime(year=int(to_year), month=int(to_month), day=int(to_day))
    _filtered_df = _tmp.loc[
        (_tmp['date'] >= _tmp_date_from) & (_tmp['date'] <= _tmp_date_to)
        ]
    return _filtered_df


def clean(df: DataFrame, numerical_only: bool = False):
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    columns = df.columns.tolist()
    if 'date' in columns:
        df = df.set_index(df['date'])
    if not numerical_only:
        return df
    df.select_dtypes(include=_numerics_data_types)
    return df


class RateOfReturnMethod(StrEnum):
    """
    Rate of return type.
    """
    SIMPLE = auto()
    LOGARITHMIC = auto()


def rate_of_return(df: DataFrame, method: RateOfReturnMethod | None = RateOfReturnMethod.LOGARITHMIC):
    match method:
        case RateOfReturnMethod.SIMPLE:
            return df.pct_change()
        case RateOfReturnMethod.LOGARITHMIC:
            for each in df.columns.to_list():
                df[each] = np.log(
                    df[each] / df[each].shift(1)
                )
            return df
