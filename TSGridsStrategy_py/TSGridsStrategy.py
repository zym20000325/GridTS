import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from pmdarima.arima import auto_arima

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.stattools import durbin_watson


############################### 全局变量 ###############################

T1=180      # 训练集时间段
freq = 10   # 预测时间段
grids = [-12,-9,-6,-3,0,3,6]  # 两个策略结合的初始网格

# ARIMA
volatility = []
entry_num = 240
# grids_arima = [-12,-9,-6,-3,0,3,6]  # ARIMA的初始网格
grids2arima = [-12,-9,-6,-3,0,3,6]
grids1arima = [-9,-6,-3,0,3,6,9]


# VAR
canshift = 0 # 初始设为0
mape_history = []
previous_forecast = [] 
turn = 0
# grids_var = [-12,-9,-6,-3,0,3,6]  # VAR的初始网格
grids2var = [-12,-9,-6,-3,0,3,6]
grids1var = [-9,-6,-3,0,3,6,9]


############################### ARIMA ###############################


def get_volatility(data):
    global volatility
    
    for i in range(240,len(data)):
        if len(volatility)==T1:
            volatility.pop(0)
        volatility.append(np.std(data.iloc[i-240:i,2]))



def do_time_series(data,T1):
    if len(volatility)<T1: # 波动率的数据不够，时间太靠前，直接pass
        return -1
    else:
        return 1



class gridSummary():
    # 当前时间，根据时间序列（如果有的话，没有就是0）应该作出的网格调整
    def __init__(self,data,fc_len=0):
        self.data = data
        self.train = None
        self.fc_len = fc_len
        self.t_len = 180
        self.forecast = None
        self.shift = 0 #跟之前的网格相比，应该偏移的量
        self.density = 3 #网格间隙
        if self.fc_len!=0:
            self.split()
            self.arima()
            self.change()

    def split(self):#data只保留spread,index_volatility
        data2 = pd.DataFrame()
        data_train = pd.DataFrame()
    
        
        data2['spread'] = self.data.loc[len(self.data)-T1:,'spread'].reset_index(drop=True)
        data2['volatility'] = volatility
        
        data_train['spread'] = self.data.loc[len(self.data)-T1+freq:,'spread'].reset_index(drop=True)
        data_train['volatility'] = volatility[freq:]
        
        self.data = data2
        self.train = data_train
    
    
    
    def arima(self):
        model = auto_arima(y=self.train['spread'], x=self.train['volatility'])
        self.forecast = model.predict(x=self.data.loc[-freq:,'volatility'],n_periods = freq)


    def change(self):
        '''
            根据forecast结果，调整shift & density
            
            shift: forecast与one.data.iloc[-1,0]比较大小
            density: forecast预测出来的波动情况
            :return:
            shift, density改变
            '''
            
            # shift
        shift_zoom = 0.1 # 预测差异放大倍数
        dif = (np.mean(self.forecast)-self.data.iloc[-1,0])*shift_zoom
        if dif%0.2>0.1:
            self.shift = round(dif//0.2*0.2+0.2,2)
        else:
            self.shift = round(dif//0.2*0.2,2)      
                        
        # density
        density_zoom = 10 # 预测波动程度放大倍数
        std = np.std(self.forecast)
        self.density = std # 这里肯定要改


def ARIMA1(data): # 上半年
    global grids1arima
    
    get_volatility(data)
    
    exe = do_time_series(data,T1)
    fc_len = freq if exe==1 else 0
    
    one = gridSummary(data, fc_len=fc_len)
    grids1arima = [round(x + one.shift, 2) for x in grids1arima]
    
    return grids1arima


def ARIMA2(data): #下半年
    global grids2arima
    
    get_volatility(data)
    
    
    exe = do_time_series(data,T1)
    fc_len = freq if exe==1 else 0
    
    one = gridSummary(data, fc_len=fc_len)
    grids2arima = [round(x + one.shift, 2) for x in grids2arima]
    
    return grids2arima


def timeseriesGridStrategyARIMA(data):
    global grids2arima
    global grids1arima
    
    contract = data.columns[2]  # 'IF2009'
    
    if contract[-1]=='9' or contract[-1]=='6':
        grids2arima = ARIMA2(data)
        return grids2arima
    else:
        grids1arima = ARIMA1(data)
        return grids1arima


############################### VAR ###############################

def timeseriesGridsStrategyVAR(data):
    
    global T1
    global freq
    global grids2var
    global grids1var
#     global base_line
#     global service_rate
#     global max_hold
#     global min_hold 
#     global add 
#     global market 
    global canshift
    global mape_history
    global previous_forecast
    global turn
    
    new_data = data[-180:]
    
    time_idx = do_time_series2(turn) # 姑且先这么设定
    fc_len = 0 if time_idx == -1 else freq

    contract = data.columns[2]  # 'IF2009'
    
    # if contract[-1]=='9' or contract[-1]=='6':
    #     grids2arima = ARIMA2(data)
    #     return grids2arima
    # else:
    #     grids1arima = ARIMA1(data)
    #     return grids1arima


    if(time_idx==0):

        # 先计算上一次预测的mape
        # print(new_data)
        previous_real = new_data['spread']
        # print(len(previous_real))
        # print(previous_real[-30:])
        previous_real = list(previous_real)
        previous_real = previous_real[-10:]
        # print(previous_real[-20:-10])
        if(len(previous_forecast)!=0):
            mape_last_turn = forecast_accuracy(previous_forecast, previous_real)
            # print(previous_forecast)
            # print(previous_real)
            # print(mape_last_turn)
            mape_history.append(mape_last_turn)
            # print(mape_history)

        canshift = mapeStrategy(canshift,mape_history)

        two = gridSummary2(new_data, fc_len=fc_len)

        if(canshift==1):   # mapeStrategy认为预测结果可信时才调整网格 否则不动
            print("did the shift")
            if contract[-1]=='9' or contract[-1]=='6':
                grids2var = [x+two.shift for x in grids2var]
            else:
                grids1var = [x+two.shift for x in grids1var]

        previous_forecast=two.forecast  # 更新“上一次预测值”
    
    
    print("turn")
    print(turn+1)
    print("canshift")
    print(canshift)
    # print("grids_var")
    # print(grids_var)
    
    turn = turn+1

    if contract[-1]=='9' or contract[-1]=='6':
        return grids2var
    else:
        return grids1var
    
    # return grids_var

# 是否使用预测移动网格

def do_time_series2(turn):

    # return -1  # 初始
    return 0   # var strategy

# 计算上一次预测结果的mape值

def forecast_accuracy(forecast, actual):

    mape = 0
    for i in range(10):

        mape = mape+(abs(forecast[i]-actual[i]))/abs(actual[i])
    
    mape = mape/10
    
    # mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    return mape

def mapeStrategy(canshift,mape_history):

    if(len(mape_history)<=10):    # 前10次不做 不稳定
        return 0
    
    # print("mape_history")
    # print(len(mape_history))
    # print(mape_history[-1])
    # print(mape_history[-2])
    # print(mape_history[-3])
    # print(mape_history[-4])
    # print(mape_history[-5])

    if(canshift==1):

        if(mape_history[-1]>=0.1):
            return 0
        else:
            return 1
    
    if(canshift==0):

        if(mape_history[-1]<0.1 and mape_history[-2]<0.1 and mape_history[-3]<0.1 and mape_history[-4]<0.1 and mape_history[-5]<0.1):
            # print("this step")
            return 1
        else:
            return 0


def select_p(df): # 选择VAR模型的p值
    
    model = VAR(df)
    AIC = []
    for i in range(1,20):
        result = model.fit(i)
        AIC.append(result.aic)
        
    p = 1
    flag=False

    if(len(AIC)>1):
    
        for i in range(len(AIC)-1):

            if(AIC[i+1]>AIC[i]):
                
                flag=True
                p=i+1
                break
    
    if(flag==False): 
        p=20
    
    return p  

def invert_transformation(df_train, df_forecast, second_diff=False):   # 将一阶或二阶差分还原
    
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

class gridSummary2():

    def __init__(self,data,fc_len=0):

        self.data = data
        self.train = None
        self.fc_len = fc_len
        self.t_len = 180
        self.forecast = None
        self.shift = 0 #跟之前的网格相比，应该偏移的量
        self.density = 3 #网格间隙

        if self.fc_len!=0:
            self.split()
            self.var()
            self.change()

    def split(self): # 分出训练集

        data2 = self.data[:]
        data2.index = data2["datetime"]
        data2 = data2.drop(columns=["datetime"])
        data_train = data2[0:T1]
        self.train = data_train[["spread","volume"]]  # 注意这里选择训练集只能放入datetime和构建VAR的因子  
        self.data = data2
    
    def var(self):  # 建立VAR模型并预测

        # 1st difference
        diff_train = self.train.diff().dropna()
        # 2st difference
        diffdiff_train = diff_train.diff().dropna()
        # 确定lag order
        # print("diffdiff_train")
        # print(diffdiff_train)
        diffdiff_train1 = diffdiff_train.reset_index(drop=True)
        p = select_p(diffdiff_train1)  # warning1
        # print("warning1")
        # print(diffdiff_train)
        # print(diffdiff_train1)
        # 建立VAR模型
        model = VAR(diffdiff_train1)   # warning2
        model_fitted = model.fit(p)   
        # 预测
        forecast_input = diffdiff_train.values[-p:]
        fc = model_fitted.forecast(y=forecast_input, steps=freq)
        # print("fc")
        # print(fc)
        # print("self.data.columns")
        # print(self.data.columns)
        df_forecast = pd.DataFrame(fc, index=self.data.index[-freq:], columns=self.train.columns + '_2d')
        
        # 将差分还原
        df_results = invert_transformation(self.train, df_forecast, second_diff=True)  
        self.forecast = list(df_results['spread_forecast'].values)


    def change(self):
 

        # shift
        # print('forecast:',self.forecast)  # test
        # print(self.data)
        # print('prev:',self.data.iloc[-1,0])
        # print('prev:',self.train['spread'][-1])  # test
        shift_zoom = 0.1 # 预测差异放大倍数
        dif = (np.mean(self.forecast)-self.train['spread'][-1])*shift_zoom
        # print(dif)   # test
        if dif%0.2>0.1:
            self.shift = dif//0.2*0.2+0.2
        else:
            self.shift = dif//0.2*0.2

        # print('shift:',self.shift)

        # density
        density_zoom = 10 # 预测波动程度放大倍数
        std = np.std(self.forecast)
        self.density = std # 这里肯定要改




############################### 整个策略调用的函数 ###############################

def TSGridsStrategy(data):
    
    global grids
    # global grids_arima
    # global grids_var
    
    grids_arima0 = timeseriesGridStrategyARIMA(data)
    grids_var0 = timeseriesGridsStrategyVAR(data)
    print("grid_arima")
    print(grids_arima0)
    print("grid_var")
    print(grids_var0)

    grids.clear()
    for i in range(len(grids_arima0)):
        grids.append((grids_arima0[i]+grids_var0[i])/2)
    
    print("grids")
    print(grids)
    print("\n")
    return grids