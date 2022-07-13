"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Technical Analysis                                                                         -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: @Rub27182n | @if722399 | @hectoronate                                                       -- #
# -- license: TGNU General Public License v3.0                                                           -- #
# -- repository: https://github.com/Rub27182n/myst_proyecto_eq6.git                                      -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import ta
import data as dt
import numpy as np
import pandas as pd
import pandas_ta as pta
import visualizations as vz
from pandas_ta import Imports
from numpy import nan as npNaN
from pandas_ta.utils import get_drift, get_offset, verify_series

def strategy_test(data):
    """Trading strategy test
    This function is built to test and show the concept of the trading strategy in the following order:
        1- Calculate Stochastic Relative Strenght Index K and D parameters.
        2- Calculate 3 levels of Exponencial Moving Average
        3- Calculate the sefault Average True Range
        4- Define SRSI K/D cross 
        5- Define Take Profit and Stop Loss with ATR
        6- Calculate buy signals when SRSI and EMA conditions are met
        7- Calculate sell signals 
        8- filter 2 times to exclude consecutive buys
        9- define points when a buy or sell is made
        10- calls a vz function to plot the result

    Args:
        pd.DataFrame with the following structure:
            Open (pd.Series): Series of 'open's
            High (pd.Series): Series of 'high's
            Low (pd.Series): Series of 'low's
            Close (pd.Series): Series of 'close's
            Volume (pd.Series): Series of 'volume's

    Returns:
        vz.strategy_test_viz(test, emas):
            plotly graph that shows the following trading strategy:
                OHLC values in candlestick graph
                buy and entry points (green-red color)
                SRSI k/d (blue-red color)
                3 EMA levels (green-red-blue color)
"""

    # Technical Indicators
    # Stochastic Relative Strenght Index
    data['K'] = stochrsi_k(pd.Series(data.Close), 14, 3, 3)
    data['D'] = stochrsi_d(pd.Series(data.Close), 14, 3, 3)

    # Exponential Weighted Moving Average
    emas = [8, 14, 40]
    for i in emas:
        data['EMA_'+str(i)] = ema(data.Close, i)

    # Average True Range
    data['ATR'] = atr(data.High, data.Low, data.Close)

    data.dropna(inplace=True)

    data['KD_Cross'] = (data['K'] > data['D']) & (data['K'] > data['D']).diff()
    data.dropna(inplace=True)

    data['TP'] = data.Close+(data.ATR*1.05)

    data['SL'] = data.Close*.99

    data['Buy_signal'] = np.where((data.KD_Cross) &
                                    (data.Close > data['EMA_'+str(emas[0])]) &
                                    (data['EMA_'+str(emas[0])] > data['EMA_'+str(emas[1])]) &
                                    (data['EMA_'+str(emas[1])] > data['EMA_'+str(emas[2])]), 1, 0)

    selldates = []
    outcome = []
    for i in range(len(data)):
        if data.Buy_signal.iloc[i]:
            k = 1
            SL = data.SL.iloc[i]
            TP = data.TP.iloc[i]
            in_position = True
            while in_position:
                if i + k ==len(data):
                    break
                looping_high = data.High.iloc[i+k]
                looping_low = data.Low.iloc[i+k]
                if looping_high >= TP:
                    selldates.append(data.iloc[i+k].name)
                    outcome.append('TP')
                    in_position = False
                elif looping_low <= SL:
                    selldates.append(data.iloc[i+k].name)
                    outcome.append('SL')
                    in_position = False
                k += 1

    data.loc[selldates, 'Sell_signal'] = 1
    data.loc[selldates, 'Outcome'] = outcome

    data.Sell_signal = data.Sell_signal.fillna(0).astype(int)

    # filter 1
    mask = data[(data.Buy_signal == 1) | (data.Sell_signal == 1)]

    # filter 2
    mask2 = mask[(mask.Buy_signal.diff() == 1) | (mask.Sell_signal.diff() == 1)]

    data[['Buy_signal', 'Sell_signal']] = 0
    data['Outcome'] = np.NaN

    data.loc[mask2.index.values, 'Buy_signal'] = mask2['Buy_signal']
    data.loc[mask2.index.values, 'Sell_signal'] = mask2['Sell_signal']
    data.loc[mask2.index.values, 'Outcome'] = mask2['Outcome']

    first_buy = data[data.Buy_signal == 1].first_valid_index()

    data = data.loc[str(first_buy):].copy()

    test = data['2018-01-01 00:00:00': '2018-01-05 00:00:00']

    def pointpos(x, type):
        if type == 'Buy':
            if x['Buy_signal']== 1:
                return x['Close']
            else:
                return np.nan
        elif type == 'Sell':
            if x['Sell_signal']== 1:
                return x['Close']
            else:
                return np.nan

    test.loc[:,'buys'] = test.apply(lambda row: pointpos(row, 'Buy'), axis=1)
    test.loc[:,'sells'] = test.apply(lambda row: pointpos(row, 'Sell'), axis=1)

    

    return vz.strategy_test_viz(test, emas)


# ------------------------- Stochastic RSI ----------------------------------------------------------------------------
def stochrsi_d(close: pd.Series, 
               window: int = 14, 
               smooth1: int = 3, 
               smooth2: int = 3) -> pd.Series:
    
    """Stochastic Relative Strenght Index D (SRSId)
    The SRSI takes advantage of both momentum indicators in order to create a more 
    sensitive indicator that is attuned to a specific security's historical performance
    rather than a generalized analysis of price change.
    
    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period
        smooth1(int): moving average of Stochastic RSI
        smooth2(int): moving average of %K

    Returns:
            pandas.Series: New feature generated.

    References:
        [1] https://www.investopedia.com/terms/s/stochrsi.asp
    """

    return ta.momentum.StochRSIIndicator(
        close=close, 
        window=window, 
        smooth1=smooth1, 
        smooth2=smooth2).stochrsi_d()

def stochrsi_k(close: pd.Series,window: int = 14,smooth1: int = 3,smooth2: int = 3,fillna: bool = False,) -> pd.Series:
    
    """Stochastic Relative Strenght Index K (SRSId)
    The SRSI takes advantage of both momentum indicators in order to create a more 
    sensitive indicator that is attuned to a specific security's historical performance
    rather than a generalized analysis of price change.

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period
        smooth1(int): moving average of Stochastic RSI
        smooth2(int): moving average of %K

    Returns:
            pandas.Series: New feature generated.

    References:
        [1] https://www.investopedia.com/terms/s/stochrsi.asp
    """

    return ta.momentum.StochRSIIndicator(
        close=close, 
        window=window, 
        smooth1=smooth1, 
        smooth2=smooth2).stochrsi_k()

# --------------------------------ATR--------------------------------------------------------------
def atr(high, 
        low, 
        close, 
        length=None, 
        mamode=None, 
        talib=None, 
        drift=None, 
        offset=None, **kwargs):

    """Average True Range (ATR)
    Averge True Range is used to measure volatility, especially volatility caused by
    gaps or limit moves.

    Args:
        high (pd.Series): Series of 'high's
        low (pd.Series): Series of 'low's
        close (pd.Series): Series of 'close's
        length (int): It's period. Default: 14
        mamode (str): See ```help(ta.ma)```. Default: 'rma'
        talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
            version. Default: True
        drift (int): The difference period. Default: 1
        offset (int): How many periods to offset the result. Default: 0

    Returns:
        pd.Series: New feature generated.

    References:
        https://www.tradingview.com/wiki/Average_True_Range_(ATR)
"""
    # Validate arguments
    length = int(length) if length and length > 0 else 14
    mamode = mamode.lower() if mamode and isinstance(mamode, str) else "rma"
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if high is None or low is None or close is None: return

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import ATR
        atr = ATR(high, low, close, length)
    else:
        tr = pta.true_range(high=high, low=low, close=close, drift=drift)
        atr = pta.overlap.ma(mamode, tr, length=length)

    percentage = kwargs.pop("percent", False)
    if percentage:
        atr *= 100 / close

    # Offset
    if offset != 0:
        atr = atr.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        atr.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        atr.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    atr.name = f"ATR{mamode[0]}_{length}{'p' if percentage else ''}"
    atr.category = "volatility"

    return atr


#----------------------------- EMA ---------------------------------------------------------------------------------------------

def ema(close, 
        length=None, 
        talib=None, 
        offset=None, **kwargs):

    """Exponential Moving Average (EMA)
    The Exponential Moving Average is more responsive moving average compared to the
    Simple Moving Average (SMA).  The weights are determined by alpha which is
    proportional to it's length.  There are several different methods of calculating
    EMA.  One method uses just the standard definition of EMA and another uses the
    SMA to generate the initial value for the rest of the calculation.
    
    Args:
        close (pd.Series): Series of 'close's
        length (int): It's period. Default: 10
        talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
            version. Default: True
        offset (int): How many periods to offset the result. Default: 0

    Returns:
        pd.Series

    References:
        [1] https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_averages
        [2] https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
"""

    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    adjust = kwargs.pop("adjust", False)
    sma = kwargs.pop("sma", True)
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if close is None: return

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import EMA
        ema = EMA(close, length)
    else:
        if sma:
            close = close.copy()
            sma_nth = close[0:length].mean()
            close[:length - 1] = npNaN
            close.iloc[length - 1] = sma_nth
        ema = close.ewm(span=length, adjust=adjust).mean()

    # Offset
    if offset != 0:
        ema = ema.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        ema.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        ema.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    ema.name = f"EMA_{length}"
    ema.category = "overlap"

    return ema


#---------------------------------VWAP--------------------------------------------------------
def vwap(high, low, close, volume, anchor=None, offset=None, **kwargs):
    """Indicator: Volume Weighted Average Price (VWAP)"""

    anchor = anchor.upper() if anchor and isinstance(anchor, str) and len(anchor) >= 1 else "D"
    offset = pta.utils.get_offset(offset)
    typical_price = pta.hlc3(high=high, low=low, close=close)

    # Calculate Result
    wp = typical_price * volume
    vwap  = wp.groupby(wp.index.to_period(anchor)).cumsum()
    vwap /= volume.groupby(volume.index.to_period(anchor)).cumsum()

    # Offset
    if offset != 0:
        vwap = vwap.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        vwap.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        vwap.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    vwap.name = f"VWAP_{anchor}"
    vwap.category = "overlap"

    return vwap


#-------------------------------PIVOTS--------------------------------------------------------
def pivots(_open, high, low, close, anchor=None, method=None):

    anchor = anchor.upper() if anchor and isinstance(anchor, str) and len(anchor) >= 1 else "D"
    method_list = ["traditional", "fibonacci", "woodie", "classic", "camarilla"]
    method = method if method in method_list else "traditional"
    date = close.index

    freq = pd.infer_freq(date)
    df = pd.DataFrame(
        index=date,
        data={"open": _open.values, "high": high.values, "low": low.values, "close": close.values},
    )

    if freq is not anchor:
        a = pd.DataFrame()
        a["open"] = df["open"].resample(anchor).first()
        a["high"] = df["high"].resample(anchor).max()
        a["low"] = df["low"].resample(anchor).min()
        a["close"] = df["close"].resample(anchor).last()
    else:
        a = df

    # Calculate the Pivot Points
    if method == "traditional":
        a["p"] = (a.high.values + a.low.values + a.close.values) / 3

        a["bc"] = (a.high.values + a.low.values ) / 2
        a["tc"] = (2 * a.p.values) - a.bc.values
        a["rng"] = abs(a.tc.values-a.bc.values)/a.p.values*100

        a["s1"] = (2 * a.p.values) - a.high.values
        a["s2"] = a.p.values - (a.high.values - a.low.values)
        a["s3"] = a.p.values - (a.high.values - a.low.values) * 2
        a["r1"] = (2 * a.p.values) - a.low.values
        a["r2"] = a.p.values + (a.high.values - a.low.values)
        a["r3"] = a.p.values + (a.high.values - a.low.values) * 2

    elif method == "fibonacci":
        a["p"] = (a.high.values + a.low.values + a.close.values) / 3
        a["pivot_range"] = a.high.values - a.low.values
        a["s1"] = a.p.values - 0.382 * a.pivot_range.values
        a["s2"] = a.p.values - 0.618 * a.pivot_range.values
        a["s3"] = a.p.values - 1 * a.pivot_range.values
        a["r1"] = a.p.values + 0.618 * a.pivot_range.values
        a["r2"] = a.p.values + 0.382 * a.pivot_range.values
        a["r3"] = a.p.values + 1 * a.pivot_range.values
        a.drop(["pivot_range"], axis=1, inplace=True)

    elif method == "woodie":
        a["pivot_range"] = a.high.values - a.low.values
        a["p"] = (a.high.values + a.low.values + a.open.values * 2) / 4
        a["s1"] = a.p.values * 2 - a.high.values
        a["s2"] = a.p.values - 1 * a.pivot_range.values
        a["s3"] = a.high.values + 2 * (a.p.values - a.low.values)
        a["s4"] = a.s3 - a.p.values
        a["r1"] = a.p.values * 2 - a.low.values
        a["r2"] = a.p.values + 1 * a.pivot_range.values
        a["r3"] = a.low.values - 2 * (a.high.values - a.p.values)
        a["r4"] = a.r3 + a.p.values
        a.drop(["pivot_range"], axis=1, inplace=True)

    elif method == "classic":
        a["p"] = (a.high.values + a.low.values + a.close.values) / 3
        a["pivot_range"] = a.high.values - a.low.values
        a["s1"] = a.p.values * 2 - a.high.values
        a["s2"] = a.p.values - 1 * a.pivot_range.values
        a["s3"] = a.p.values - 2 * a.pivot_range.values
        a["s4"] = a.p.values - 3 * a.pivot_range.values
        a["r1"] = a.p.values * 2 - a.low.values
        a["r2"] = a.p.values + 1 * a.pivot_range.values
        a["r3"] = a.p.values + 2 * a.pivot_range.values
        a["r4"] = a.p.values + 3 * a.pivot_range.values
        a.drop(["pivot_range"], axis=1, inplace=True)
    else:
        raise ValueError("Invalid method")

    if freq is not anchor:
        pivots_df = a.reindex(df.index, method="ffill")
    else:
        pivots_df = a

    pivots_df.drop(columns=["open", "high", "low", "close"], inplace=True)

    return pivots_df