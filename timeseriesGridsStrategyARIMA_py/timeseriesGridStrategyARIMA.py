import pandas as pd
import numpy as np
import sys
from pmdarima.arima import auto_arima


###############################################
# Do not need this on web
# sys.path.insert(0,'.')
# path = 'data/web/example_ok.csv'
# data_full = pd.read_csv(path)
# data: 每次传入长度250的数据

# 12-09以6.12上午9：30传入为起始点
# 有avg3数据后，是从6.19上午9:30传入为起始点，因此30min idx 从26开始

###############################################


T1 = 180
freq = 10
grids2 = [-12,-9,-6,-3,0,3,6]
grids1 = [-9,-6,-3,0,3,6,9]


volatility = []
entry_num = 240 # 新的合约开始时，初始化为240


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
    global grids1
    
    get_volatility(data)
    
    exe = do_time_series(data,T1)
    fc_len = freq if exe==1 else 0
    
    one = gridSummary(data, fc_len=fc_len)
    grids1 = [round(x + one.shift, 2) for x in grids1]
    
    return grids1


def ARIMA2(data): #下半年
    global grids2
    
    get_volatility(data)
    
    
    exe = do_time_series(data,T1)
    fc_len = freq if exe==1 else 0
    
    one = gridSummary(data, fc_len=fc_len)
    grids2 = [round(x + one.shift, 2) for x in grids2]
    
    return grids2


def timeseriesGridStrategyARIMA(data):
    global grids2
    global grids1
    
    
    contract = data.columns[2]  # 'IF2009'
    
    if contract[-1]=='9' or contract[-1]=='6':
        grids2 = ARIMA2(data)
        return grids2
    else:
        grids1 = ARIMA1(data)
        return grids1

'''
    def timeseriesGridStrategyARIMA2(data):
    global grids2
    global girds1
    
    
    for read1 in range(0, len(data_full) - 250, 10):
    read2 = read1 + 250
    data = data_full.iloc[read1:read2, :].reset_index(drop=True)
    
    
    contract = data.columns[2]  # 'IF2009'
    
    if contract[-1] == '9' or contract[-1] == '6':
    grids2 = ARIMA2(data)
    print(grids2)
    else:
    grids1 = ARIMA1(data)
    print(grids1)
    
    timeseriesGridStrategyARIMA2(data_full)
    '''


