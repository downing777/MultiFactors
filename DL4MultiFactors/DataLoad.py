import scipy.io as scio
import pandas as pd
import numpy as np
def read_mat(path,start,end):
    """
    读取mat文件
    """
    col=list(scio.loadmat('./Data/AllStockCode.mat').values())[3]
    index=list(scio.loadmat('./Data/TradingDate_Daily.mat').values())[3]
    col = [i[0] for i in col[0]]
    index = [i for i in index]
    data = list(scio.loadmat(path).values())[3]
    data = pd.DataFrame(data,index=index,columns=col)
    data = data.reset_index()
    data['level_0'] = data['level_0'].astype('str')  # reset之后原index保留为 level0
    data['level_0'] = pd.to_datetime(data['level_0'], format='%Y%m%d')
    data=data.rename(columns={'level_0' : 'Date'})
    data=data.replace(0,np.nan)  # Now the DataFrame will have NaN instead of 0 values
    data=data[(data['Date'] >= start) & (data['Date'] <= end)]
    data=data.set_index('Date', drop='True')
    return data
