import numpy as np
from scipy.stats import linregress
import pandas as pd
import torch


import GlobalData

df_sn_d = GlobalData.df_sn_d
df_c_div_d = GlobalData.df_c_div_d

def DaytoMon(df, window=20, threshold=10):
    # 先将原始数据回看二十天取均值，在resample按月划分
    # resample会将index自动归位每月最后一天，不影响计算数据

    daily_mean = df.rolling(window=window, min_periods=threshold).mean()
    monthly_factor_raw = daily_mean.resample('M').last()

    return monthly_factor_raw


# 自实现3倍中位数绝对偏差去极值

def MAD(factor):
    med = np.median(factor)
    mad = np.median(abs(factor - med))

    # 求出3倍中位数的上下限制
    up = med + (3 * 1.4826 * mad)
    down = med - (3 * 1.4826 * mad)

    # 利用3倍中位数的值去极值
    factor = np.where(factor > up, up, factor)
    factor = np.where(factor < down, down, factor)

    return factor

def Neutralization(df, df_cmv, window=20, threshold=10):
    '''
    Factor neutralization
    '''
    neutralized_factor = df.copy()
    for i in range(neutralized_factor.shape[0]):
        y = MAD(df.iloc[i, :].values)
        x = np.log(df_cmv.iloc[i, :].values)
        non_nan_indices = ~np.isnan(x) & ~np.isnan(y)
        slope, intercept, r_value, p_value, std_err = linregress(y[non_nan_indices], x[non_nan_indices])
        neutralized_factor.iloc[i, :] = y - (intercept + slope * x)
    return neutralized_factor

def CirculationMarketValue(df_sn, df_c_div):
    '''
    流通市值 = 交易日流通股本*交易日收盘价
    '''
    return df_sn * df_c_div

def FactorConstruction(df_d):
    df_m = DaytoMon(df_d)
    df_cmv_d = CirculationMarketValue(df_sn_d, df_c_div_d)
    df_cmv_m = df_cmv_d.resample('M').last()
    return Neutralization(df_m, df_cmv_m)

def MACD(df_c, lag_short, lag_long, lag_dea):
    """
    短期指数移动平均线和长期指数移动平均线的聚合、分离情况
    """
    short_ema = df_c.ewm(span=lag_short).mean()
    long_ema = df_c.ewm(span=lag_long).mean()  # 计算移动加权平均线
    diff = short_ema - long_ema
    dea = diff.ewm(span=lag_dea).mean()
    macd = 2 * (diff - dea)
    return macd

def AMA(df_c, lag_short, lag_long):
    short_ema = df_c.ewm(span=lag_short).mean()
    long_ema = df_c.ewm(span=lag_long).mean()
    return (short_ema / long_ema).ewm(span=lag_short).mean()

def VHF(df_c, lag):
    high = df_c.rolling(window=lag, min_periods=1).max()
    low = df_c.rolling(window=lag, min_periods=1).min()
    diff_price = df_c.diff(periods=1)
    total_asb_change = diff_price.abs().rolling(window=lag, min_periods=1).sum()
    return (high - low) / total_asb_change

def ForceIndex(df_c, df_t, lag):
    ret = df_c / df_c.shift(1) - 1
    forceindex = (ret * df_t).rolling(window=lag, min_periods=1).mean()
    return forceindex

def VR(df_c, df_v, lag):
    ret = df_c / df_c.shift(1) - 1
    volume_up = df_v.copy()
    volume_down = df_v.copy()
    volume_up[ret < 0] = 0
    volume_down[ret > 0] = 0
    vr = volume_up.rolling(window=lag, min_periods=1).mean() / volume_down.rolling(window=lag, min_periods=1).mean()
    return vr

def OBV(df_c, df_v, lag):
    d = (df_c > df_c.shift(1)).astype(int)
    d[d == 0] = -1
    obv = (d * df_v).rolling(window=lag, min_periods=1).mean()
    return obv


def TRIX(df_c, lag_short, lag_long):
    short = df_c.rolling(window=lag_short, min_periods=1).mean()
    long = df_c.rolling(window=lag_long, min_periods=1).mean()
    trix = short / long - 1
    return trix

# Oscillation

def RVI(df_c, df_o,df_h, df_l, lag):
    """
    Relative Vigor Index:High RVI values are interpreted as overbought conditions in the market,
    indicating that the market may be overheating.
    Undividended data.
    """
    MovAverage = (df_c - df_o).rolling(window = lag, min_periods = 1).mean()
    RangeAverage = (df_h - df_l).rolling(window = lag, min_periods = 1).mean()
    rvi = MovAverage/RangeAverage
    return rvi

def BIAS(df_c, lag):
    MA = df_c.rolling(window = lag, min_periods = 1).mean()
    bias = df_c/MA - 1
    return bias

def KDJ(df_l, df_h, df_c, lag):
    low = df_l.rolling(window = lag, min_periods = 1).min()
    high = df_h.rolling(window = lag, min_periods = 1).max()
    rsv = (df_c - low) / (high - low) * 100
    KDJ_K = rsv.ewm(adjust = False, alpha = 1/3).mean()
    KDJ_D = KDJ_K. ewm(adjust = False, alpha = 1/3).mean()
    KDJ_J = 3*KDJ_K - 2*KDJ_D
    return KDJ_J

def RSI(df_c, lag):
    P_delta = df_c - df_c.shift(1)
    P_delta_pos = P_delta.copy()
    P_delta_pos[P_delta<=0] = 0
    P_delta_neg = P_delta.copy()
    P_delta_neg [P_delta>=0] = 0
    P_delta_up = P_delta_pos.rolling(window = lag, min_periods = 1).mean()
    P_delta_down = P_delta_neg.rolling(window = lag, min_periods = 1).mean()
    RS = P_delta_up/P_delta_down
    rsi = 100 - 100/ (1 + RS.abs())
    return rsi

def CMO(df_c, lag):
    P_delta = df_c - df_c.shift(1)
    P_delta_up = P_delta.copy()
    P_delta_down = P_delta.copy()
    P_delta_up[P_delta < 0] = 0
    P_delta_down[P_delta > 0] = 0
    P_delta_up_sum = P_delta_up.rolling(window = lag, min_periods = 1).sum ()
    P_delta_down_sum = P_delta_down.rolling(window = lag, min_periods = 1).sum ()
    cmo = (P_delta_up_sum + P_delta_down_sum)/(P_delta_up_sum - P_delta_down_sum)
    return cmo


df_c_div_d = GlobalData.df_c_div_d
df_o_div_d = GlobalData.df_o_div_d
df_v_d = GlobalData.df_v_d
df_tr_d = GlobalData.df_tr_d
df_l_div_d = GlobalData.df_l_div_d
df_h_div_d = GlobalData.df_h_div_d
df_return_rate_m = GlobalData.df_return_rate_m

#Construct your raw daily factor and adjust the parameters

df_AMA_m = FactorConstruction(AMA(df_c_div_d, lag_short=12, lag_long=26))
df_MACD_m = FactorConstruction(MACD(df_c_div_d, lag_short=12, lag_long=26, lag_dea=9))
df_VHF_m = FactorConstruction(VHF(df_c_div_d, lag=12))
df_FI_m = FactorConstruction(ForceIndex(df_c_div_d, df_tr_d, lag=12))
df_VR_m = FactorConstruction(VR(df_c_div_d, df_v_d, lag=12))
df_OBV_m = FactorConstruction(OBV(df_c_div_d, df_v_d, lag=12))
df_TRIX_m = FactorConstruction(TRIX(df_c_div_d, lag_short=12, lag_long=26))

df_RVI_m = FactorConstruction(RVI(df_c_div_d, df_o_div_d, df_h_div_d, df_l_div_d, lag=12))
df_BIAS_m = FactorConstruction(BIAS(df_c_div_d, lag=12))
df_KDJ_m = FactorConstruction(KDJ(df_l_div_d, df_h_div_d, df_c_div_d, lag=12))
df_RSI_m = FactorConstruction(RSI(df_c_div_d, lag=12))
df_CMO_m = FactorConstruction(CMO(df_c_div_d, lag=12))

#准备训练数据
factors = [df_AMA_m, df_MACD_m, df_VHF_m, df_FI_m, df_VR_m, df_OBV_m, df_TRIX_m, df_RVI_m, df_BIAS_m, df_KDJ_m, df_RSI_m, df_CMO_m]
n_stock = df_AMA_m.shape[1]
n_factor = len(factors)
time_range = df_AMA_m.shape[0]



data_temp = np.zeros((n_stock, n_factor+1))
for i in range(n_factor):
    data_temp[:, i] = factors[i].iloc[0]
data_temp[:, -1] = df_return_rate_m.iloc[1]

df = pd.DataFrame(data_temp) # 构建同一期所有因子值
#df = df.dropna()
features = torch.tensor(df.iloc[:, 0:n_factor].values)
rr = torch.tensor(df.iloc[:, -1].values).view(-1,1)

for k in range(1, time_range-1):
    data_temp = np.zeros((n_stock, n_factor + 1))
    for i in range(n_factor):
        data_temp[:, i] = factors[i].iloc[k]
    data_temp[:, -1] = df_return_rate_m.iloc[k+1]
    df = pd.DataFrame(data_temp)  # 构建同一期所有因子值
    #df = df.dropna()
    if k == 1:
        features = torch.stack((features, torch.tensor(df.iloc[:, 0:n_factor].values)), dim=2)
        rr = torch.stack((rr, torch.tensor(df.iloc[:, -1].values).view(-1, 1)), dim=2)
    else:
        features = torch.cat((features, torch.tensor(df.iloc[:, 0:n_factor].values).unsqueeze(2)), dim=2)
        rr = torch.cat((rr, torch.tensor(df.iloc[:, -1].values).view(-1, 1).unsqueeze(2)), dim=2)

torch.save(features, 'features.pt')
torch.save(rr, 'return_rate.pt')
print(features.shape)
print(rr.shape)

