import numpy as np
import pandas as pd
from DataLoad import read_mat

# df: dataframe of any certain factor
# _d: daily data; _m: monthly data; _div: dividend
# _c: stock_close_price   row:date; column:stock_code
# _v: stock_volume   成交量
# _tr: stock_turnover_rate
# _sn: share number 流通股本

start = '2014-01-01'
end = '2023-7-31'
# daily price data
df_c_d = read_mat('./Data/AllStock_DailyClose.mat', start, end)
df_o_d = read_mat('./Data/AllStock_DailyOpen.mat', start, end)
df_h_d = read_mat('./Data/AllStock_DailyHigh.mat', start, end)
df_l_d = read_mat('./Data/AllStock_DailyLow.mat', start, end)

# daily dividend price data
df_c_div_d = read_mat('./Data/AllStock_DailyClose_dividend.mat', start, end)
df_o_div_d = read_mat('./Data/AllStock_DailyOpen_dividend.mat', start, end)
df_h_div_d = read_mat('./Data/AllStock_DailyHigh_dividend.mat', start, end)
df_l_div_d = read_mat('./Data/AllStock_DailyLow_dividend.mat', start, end)

# daily trading data
df_sn_d = read_mat('./Data/AllStock_DailyAShareNum.mat', start, end)
df_ld_d = read_mat('./Data/AllStock_DailyListedDate.mat', start, end)
df_st_d = read_mat('./Data/AllStock_DailyST.mat', start, end)
df_sta_d = read_mat('./Data/AllStock_DailyStatus.mat', start, end)
df_v_d = read_mat('./Data/AllStock_DailyVolume.mat', start, end)
df_tr_d = read_mat('./Data/AllStock_DailyTR.mat', start, end)

# all close data(for the check of sub_new stock)
df_c_all = read_mat('./Data/AllStock_DailyClose.mat', '2004-01-02', end)

# monthly close price, open price
df_c_div_m = df_c_div_d.resample('M').last()
df_o_div_m = df_o_div_d.resample('M').first()
df_return_rate_m = df_c_div_m / df_o_div_m - 1

# Factor preprocess and constraints preparation
# Factor = Factor * ST * Halt * Sub_new * Limit_up

# ST
# monthly, nan for st and 1 for non-st, eliminate the stock when it is st last month
df_st_m = df_st_d.replace(np.nan, 0).resample('M').mean()
df_st_m = df_st_m.applymap(lambda x: np.nan if x > 0.5 else 1)

# Halt stock(status)
# monthly，nan for suspended stock and 1 for normal, eliminate the stock when it is suspended last month
df_sta_m = df_sta_d.replace(np.nan, 0).resample('M').mean()
df_sta_m = df_sta_m.applymap(lambda x: 1 if x > 0.5 else np.nan)

# Sub_new
# subnew 需要借助全部日期信息，最后取截断
df_subnew_d = df_c_all.where(df_c_all.isna(), 1)
df_subnew_d = df_subnew_d.apply(lambda x: x.cumsum())
df_subnew_m = df_subnew_d.resample('M').last()
df_subnew_m = df_subnew_m.applymap(lambda x: 1 if x >= 60 else np.nan)

df_subnew_m = df_subnew_m[(df_subnew_m.index >= start) & (df_subnew_m.index <= end)]

# what about limit up and down
daily_return = (df_o_div_d / df_c_div_d) - 1  # there's nan values

# buy for limit up, nan: you can not purchase the stock, 1:you can purchase the stock
df_lu_m = daily_return.replace(np.nan, 0).resample('M').last()
df_lu_m = df_lu_m.applymap(lambda x: np.nan if x > 0.1 else 1)

# sell for limit down, used in the backtest part not the preparation(by keeping the group)

df_ld_m = daily_return.replace(np.nan, 0).resample('M').last()
df_ld_m = df_ld_m.applymap(lambda x: np.nan if x < -0.1 else 1)
