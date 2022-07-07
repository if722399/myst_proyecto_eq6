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

# ob_ts = list(dt.ob_data.keys())
# l_ts = [pd.to_datetime(i_ts) for i_ts in ob_ts] # Modificar todos los valores de str a unidades de tiempo

# mid_prices = [(dt.ob_data[ob_ts[i]]['ask'][0] + dt.ob_data[ob_ts[i]]['bid'][0])/2 for i in range(len(ob_ts))]

#n_df = dt.ptm.drop(columns = ['Unnamed: 0', 'timestamp'])







# Leer datos
aave = dt.aave

# ----------- Promedio del volumen de los ultimos 14 precios: ----------- #

v = np.zeros(len(aave))
v[:len(aave)-14] = [np.mean(aave.iloc[0:,4][i:i+13]) for i in range(len(aave)-14)]
avg_volume = pd.DataFrame({'avg_volume':v})

# ---------- RSI: -----------# 

diferencias = np.diff(aave.iloc[0:,4][::-1])


RSI = []
for i in range(len(diferencias)-14):
    subidas = []
    bajadas = []
    temp = diferencias[i:i+14]
    for j in temp:
        if j>0:
            subidas.append(j)
        elif j<0:
            bajadas.append(j)
        else:
            None
        m_subidas = np.mean(subidas)
        m_bajadas = np.abs(np.mean(bajadas))

        RS = (m_subidas/m_bajadas)
        RSI.append(100 - (100/(1+RS)))

# 1) ¿Qué hacer cuando no hay subidas/bajadas? 



# -------------- Stochastic RSI -------------- #

# Stoch RSI =  ( RSI - Lowest RSI )   /  ( Max RSI - Lowest RSI)

# RSI = Current RSI reading
# Lower RSI = Minimum RSI reading since the last 14 oscillations
# Max RSI = Maximum RSI reading since the last 14 oscillations





# -------------- Pivot Points -------------- #

# Obtenemos metricas del día 1 de Junio

def Pivot_points(df) -> dict:

    '''
    df: DataFrame que contiene las columnas: [Open,High,Low,Close]
    
    '''

    H = df.iloc[0:,[1,2,3,4]].max().max() # Máximo del día anterior
    L = df.iloc[0:,[1,2,3,4]].min().min() # Mínimo del día anterior
    C = df['Close'][df['Close time'] == np.max(df['Close time'])].values[0] # Último cierre

    P = (H + L + C)/3 # Pivot point base
    S1 = (P*2) - H # Primer soporte
    R1 = (P*2) - L # Primer resistencia 
    S2 = P - (R1-S1) # Segundo soporte
    R2 = P + (R1-S1) # Segunda resistencia 
    S3 = P - (R2-S2) # Tercer soporte
    R3 = P + (R2-S2) # Tercer resistencia 

    return {'Pivot point base':P, 'Primer soporte':S1, 'Primer resistencia':R1, 'Segundo soporte':S2,
            'Segunda resistencia':R2, 'Tercer soporte':S3, 'Tercer resistencia':R3}



PP = Pivot_points(aave)



aave = dt.aave # Data del día 2 de Junio
(aave['Close'][aave['Close time'] == np.max(aave['Close time'])])


aave.iloc[0:,[1,2,3,4]]






# VWAP (se calcula con los libros de ordenes)