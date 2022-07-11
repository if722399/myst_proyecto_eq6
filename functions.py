"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                               -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import data as dt
import pandas as pd
import numpy as np
import pandas_ta as pta

# -------------- Stochastic RSI -------------- #

def Stoch_RSI(df,n,rsi_name):

    ''' Indicator: Relative Strength Index (RSI) '''

    Stoch_RSI_t = []
    for i in range(len(df[rsi_name])-n):
        temp = df[rsi_name][i:i+n]
        L_RSI = np.min(temp)
        M_RSI = np.max(temp)
        RSI = temp.values[-1]

        Stoch_RSI_t.append(( RSI - L_RSI )   /  ( M_RSI - L_RSI))


    Stoch_RSI =  np.zeros(len(df))
    Stoch_RSI[n:] = np.array(Stoch_RSI_t)*100

    return Stoch_RSI

#------------------------------VWAP--------------------------------------
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


#-------------------------------PIVOTS-----------------------------------
def pivots(_open, high, low, close, anchor=None, method=None):

    anchor = anchor.upper() if anchor and isinstance(anchor, str) and len(anchor) >= 1 else "D"
    method_list = ["traditional", "fibonacci", "woodie", "classic", "demark", "camarilla"]
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
        a["r1"] = a.p.values + 0.382 * a.pivot_range.values
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

    elif method == "demark":
        conds = (
            a.close.values == a.open.values,
            a.close.values > a.open.values,
        )
        vals = (
            a.high.values + a.low.values + a.close.values * 2,
            a.high.values * 2 + a.low.values + a.close.values,
        )
        p = np.select(conds, vals, default=(a.high.values + a.low.values * 2 + a.close.values))
        a["p"] = p / 4
        a["s1"] = p / 2 - a.high.values
        a["r1"] = p / 2 - a.low.values
    elif method == "camarilla":
        a["p"] = (a.high.values + a.low.values + a.close.values) / 3
        a["pivot_range"] = a.high.values - a.low.values
        a["s1"] = a.close.values - a.pivot_range.values * 1.1 / 12
        a["s2"] = a.close.values - a.pivot_range.values * 1.1 / 6
        a["s3"] = a.close.values - a.pivot_range.values * 1.1 / 4
        a["s4"] = a.close.values - a.pivot_range.values * 1.1 / 2
        a["r1"] = a.close.values + a.pivot_range.values * 1.1 / 12
        a["r2"] = a.close.values + a.pivot_range.values * 1.1 / 6
        a["r3"] = a.close.values + a.pivot_range.values * 1.1 / 4
        a["r4"] = a.close.values + a.pivot_range.values * 1.1 / 2
        a.drop(["pivot_range"], axis=1, inplace=True)
    else:
        raise ValueError("Invalid method")

    if freq is not anchor:
        pivots_df = a.reindex(df.index, method="ffill")
    else:
        pivots_df = a

    pivots_df.drop(columns=["open", "high", "low", "close"], inplace=True)

    return pivots_df


#----------------------------- EMA200 & Signal ---------------------------------

def EMA200_Signal(df)-> dict:

    # -------Calculate EMA200 Metric: -------
    df['EMA200'] = pta.ema(df.Close, length = 200)



    # -------Generate the signal: -------
    emasignal = [0]*len(df)
    backcandles = 8

    for j in range(backcandles-1,len(df)):
        upt = 1
        dnt = 1
        for i in range(j-backcandles,j+1):
            if df.High[j]>= df.EMA200[j]:
                dnt = 0
            if df.Low[j]<= df.EMA200[j]:
                upt = 0
        if upt == 1 and dnt == 1:
            emasignal[j] = 3
        elif upt == 1:
            emasignal[j] = 2
        elif dnt == 1:
            emasignal[j] = 1

    return {'EMA200':df['EMA200'],'EMA200_Signal':emasignal}


    #----------------------------- Total Signal ------------------------------#

def TotSignal(df)-> 'List':
    
    ''' The input df must have columns with the RSI and the EMA200 signals'''

    TotSignal = [0] * len(df)
    for i in range(len(df)):
        TotSignal[i] = 0
        if df.EMA_Signal[i] == 1 and df.RSI[i] >= 70:
            TotSignal[i] = 1
        if df.EMA_Signal[i] == 2 and df.RSI[i] <= 30:
            TotSignal[i] = 2

    return TotSignal

#----------------------------- Point Positions ------------------------------#

def pointpos(x):

    ''' This function was created for visualizations purposes'''

    if x['TotSignal'] == 1:
        return x['High'] + 50
    elif x['TotSignal'] == 2:
        return x['Low'] - 50
    else: 
        return np.nan