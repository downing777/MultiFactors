# 技术类因子合成 机器学习&因子选股

## 12个单技术指标

  - MACD  AMA VHF ForceIndex VR OBV PSY TRIX RVI KDJ BIAS RSI CMO 参数见'回测数据.xlsx', time range  '2014-01-01'
    -- '2023-07-31'
  - project : ML4 Mutifactors, 运行主程序'main.py'得到训练数据； ‘features.pt’ ‘rr.pt’
    - Features: stock numbers(5523) * features(12) * time （114 months）tensor
    - rr: return rate; stock numbers(5523) * rr (1) * time （114 months）
 	- ...

## 回测框架

'factors_backtest.ipynb'  ''**Load DATA**' -> '**Test Collection**' 部分

- monthly factor 格式见‘AMA.csv’, 注意index需要恢复到datetime 格式

  ```python
  df_AMA_m = pd.read_csv('AMA.csv')
  df_AMA_m['Date'] = pd.to_datetime(df_AMA_m['Date'])
  df_AMA_m.set_index('Date', inplace=True)
  ```

- Test(factor, freq = 'M') 回测月频因子,(freq = 'd'则通过 **FactorConstruction()** 首先进行频率day - mon转化)
- 输出十分组净值曲线，评价指标 IC	ICIR	RankIC	RankICIR	年化收益率	波动率	信息比率	胜率	最大回撤
- 中间产物，未输出，可调整作后续用途：
  - 十分组收益表： index: '2014-01-01' -- '2023-07-31' Datetime; Columns : ['group1'- 'group10', 'hedge']
  - 十分组净值表

## 机器学习多因子合成

‘Factors_training.ipynb’, features： 12 normalized features during certain past period， target：return rate

 of the next slot(month), 回测结果：'回测数据.xlsx'.

#### 等权多因子

Factors .\* sign(IC) .mean()

#### Linear Regression

- 滚动训练，每一期更新回归系数,训练集为截止当期所有数据

- 年化收益率最高，0.142045

#### SVR

'factor_svr', trained on google Colab

- 滚动训练，每12期更新模型参数，incremental learning，新增12个月数据训练而非当前所有数据
- 年化 **0.134655**， 信息比率**1.276015**

#### SVR + RFE

Recursive Feature Elimination, 'factor_svr_rfe‘

- min_ factors_num = 5
- 回测结果暴跌，原因不明

- 一个地方有隐患，这里对测试集（预测下一期）标准化使用的是测试集的均值和方差，是维持mean 0 std1 还是维持和训练集一样的map？
- ...

#### RF

- 特征少，无法区分
- default 参数，回测指数暴跌
- 。。。
- 
