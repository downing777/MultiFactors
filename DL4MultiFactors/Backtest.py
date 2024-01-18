import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import GlobalData

# monthly close price, open price
df_st_m = GlobalData.df_st_m
df_subnew_m = GlobalData.df_subnew_m
df_sta_m = GlobalData.df_sta_m
df_lu_m = GlobalData.df_lu_m
df_ld_m = GlobalData.df_ld_m
df_o_div = GlobalData.df_o_div_m
df_c_div = GlobalData.df_c_div_m
df_return_rate_m = GlobalData.df_return_rate_m


def Factor_backtest(df, hedge_method='10-1'):
    '''
    df: factors by the end of month
    df_c_div: dividend close value by the end of month
    df_o_div: dividend open value by the start of month
    '''

    df = df * df_st_m * df_sta_m * df_subnew_m * df_lu_m
    # net / rate dataframe initialization
    columns = ['group1', 'group2', 'group3', 'group4', 'group5', 'group6', 'group7', 'group8', 'group9', 'group10',
               'hedge']
    group_rate = pd.DataFrame(index=df.index, columns=columns)
    group_net = pd.DataFrame(index=df.index, columns=columns)
    group_rate.iloc[0, :] = 0

    # Initialize the group transfer indicator
    group_transfer = pd.Series(0, index=df.columns)

    for i in range(df.shape[0] - 1):
        factor = df.iloc[i, :].to_frame()
        open_price = df_o_div.iloc[(i + 1), :].to_frame()
        close_price = df_c_div.iloc[(i + 1), :].to_frame()
        return_rate = df_return_rate_m.iloc[(i + 1), :].to_frame()
        ld_indicator = np.isnan(df_ld_m.iloc[(i + 1), :].to_frame())
        df_temp = pd.concat([factor, open_price, close_price, return_rate, ld_indicator], axis=1)
        column_names = ['factor', 'open_price', 'close_price', 'return_rate', 'limit_down_indicator']
        df_temp.columns = column_names

        # 排序，十分组
        df_temp_sorted = df_temp.sort_values(by='factor')
        df_temp_sorted = df_temp_sorted.dropna(axis=0)  # a little bit weird,axis 表示按照轴的方向进行操作， axis = 0 按照行的方向
        df_temp_sorted.drop(df_temp_sorted[df_temp_sorted['return_rate'] > 20].index,
                            inplace=True)  # 剔除极端异常值（return rate = 900）
        df_temp_sorted['Group'] = pd.qcut(df_temp_sorted['factor'], q=10,
                                          labels=False) + 1  # qcut 分位数划分，保证每个范围内的观测差不多，cut是按照factor 数值范围十等分

        for stock in df_temp_sorted.index:
            if (stock in group_transfer.index) and (group_transfer.loc[stock] != 0):
                df_temp_sorted.loc[stock, 'Group'] = group_transfer.loc[stock]

        group_mean = df_temp_sorted.groupby('Group')['return_rate'].mean().values

        group_transfer = df_temp_sorted['Group'] * df_temp_sorted['limit_down_indicator']

        '''
        Debug
        #print(i)
        #print(df_temp_sorted)
        #print(group_mean.shape)
        #print(group_rate.iloc[(i+1), :].shape)
        '''

        # gruop_rate 赋值
        # hedging (group10 - group1) or (group1 - group 10)
        group_rate.iloc[(i + 1), : -1] = group_mean

        if hedge_method == '10-1':
            group_rate.iloc[(i + 1), -1] = group_mean[-1] - group_mean[0]
        else:
            group_rate.iloc[(i + 1), -1] = group_mean[0] - group_mean[-1]

    # group net construction
    for i in range(group_net.shape[0]):
        for j in range(group_net.shape[1]):
            group_net.iloc[i, j] = (group_rate.iloc[0: i, j] + 1).product()
        # backsee test

    fig, ax1 = plt.subplots(figsize=(15, 8))

    for column in columns[:-1]:
        ax1.plot(group_net.index, group_net[column], label=column)

    ax2 = ax1.twinx()
    if hedge_method == '10-1':
        ax2.plot(group_net.index, group_net['hedge'], linestyle='--', color='gray', label='hedge 10-1')
    else:
        ax2.plot(group_net.index, group_net['hedge'], linestyle='--', color='gray', label='hedge 1-10')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines + lines2, labels + labels2)

    # 显示图形
    plt.show()
    return group_rate, group_net


import time


# 月度IC与RankIC均值
def get_IC(df_factor_m):
    IC = []
    RankIC = []
    for i in range(df_factor_m.shape[0] - 1):
        factor = df_factor_m.iloc[i, :]
        return_rate = df_return_rate_m.iloc[(i + 1), :]
        IC.append(factor.corr(return_rate, method='pearson'))
        RankIC.append(factor.corr(return_rate, method='spearman'))
    IC = pd.DataFrame(IC)
    RankIC = pd.DataFrame(RankIC)
    return IC, RankIC


# 年化IC&IR
def get_Cor(df_factor_m):
    IC, RankIC = get_IC(df_factor_m)
    cor = pd.DataFrame()
    cor['IC'] = IC.mean()
    cor['ICIR'] = (IC.mean() / IC.std()) * (12 ** 0.5)
    cor['RankIC'] = RankIC.mean()
    cor['RankICIR'] = (RankIC.mean() / RankIC.std()) * (12 ** 0.5)
    cor.index = ['factor']
    return cor.T

# 最大回撤
def get_maxDrawDown(return_list):
    """
    求最大回撤率
    #param return_list:Series格式月度收益率
    #return：0~1

    """
    return_list=list((return_list+1).cumprod())
    i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  # 结束位置
    if i == 0:
        return 0
    else:
        j = np.argmax(return_list[:i])  # 开始位置
        return((return_list[j] - return_list[i]) / (return_list[j]))


# Evaluation
# 胜率
def get_winrate(Rev_seq):
    ret_winrate=Rev_seq[Rev_seq>=0].count()/Rev_seq.count()
    return ret_winrate

# 数据输出:年化收益率、信息比率、最大回撤、胜率
def evaluate_PortfolioRet(Rev_seq,t=12,tests='Hedge'):
    """
    数据输出:多空对冲年化收益率、信息比率、最大回撤、胜率
    #Rev_seq:DataFrame 收益数据
    #return：回测指标表
    """
    group_num = 10

    if type(Rev_seq)==type(pd.DataFrame()):
        if tests=='Hedge':
            num= group_num
        else:
            num=0
        Rev_seq=Rev_seq.iloc[:,(num)]
    else:
        Rev_seq=Rev_seq
    Rev_seq=Rev_seq.replace(np.nan,0)
    ret_mean=((np.prod(Rev_seq.values+1))**(1/len(Rev_seq.values)))**t-1
    ret_std=Rev_seq.std()*t**0.5
    ret_winrate=get_winrate(Rev_seq)
    ret_maxloss=get_maxDrawDown(Rev_seq)
    ret_sharp=ret_mean/ret_std
    return pd.DataFrame([ret_mean,ret_std,ret_sharp,ret_winrate,ret_maxloss],index=['年化收益率:','波动率:','信息比率:','胜率:','最大回撤'], columns= ['factor'])
