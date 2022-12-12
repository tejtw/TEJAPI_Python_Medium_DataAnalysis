#!/usr/bin/env python
# coding: utf-8

# In[16]:


#載入資料


# In[52]:


#下載資料
import tejapi
import pandas as pd
tejapi.ApiConfig.api_base="http://10.10.10.66"
tejapi.ApiConfig.api_key = "05ZW2W5YJP44b1qwosBAkBPc0llKXF"
tejapi.ApiConfig.ignoretz = True
coid = "0050"
mdate = {'gte':'2011-01-01', 'lte':'2022-11-15'}
data = tejapi.get('TWN/APRCD1',
                          coid = coid,
                          mdate = {'gte':'2011-01-01', 'lte':'2022-11-15'},
                          paginate=True)


#開高低收、成交量
data = data[["coid","mdate","open_adj","high_adj","low_adj","close_adj","amount"]]
data = data.rename(columns={"coid":"coid","mdate":"mdate","open_adj":"open",
                   "high_adj":"high","low_adj":"low","close_adj":"close","amount":"vol"})


# In[53]:


#技術指標
#動能 : KD、RSI、MACD、MOM
from talib import abstract
data["rsi"] = abstract.RSI(data,timeperiod=14)
data[["macd","macdsig","macdhist"]] = abstract.MACD(data)
data[["kdf","kds"]] = abstract.STOCH(data)
data["mom"] = abstract.MOM(data,timeperiod=15)
data.set_index(data["mdate"],inplace = True)


# In[54]:


#總經指標
#台股本益比
data1 = tejapi.get('GLOBAL/ANMAR',
                          mdate = mdate,
                          coid = "SA15",
                          paginate=True)
data1.set_index(data1["mdate"],inplace = True)
data1 = data1.resample('D').ffill()
data = pd.merge(data,data1["val"],how='left', left_index=True, right_index=True)
data.rename({"val":"pe"}, axis=1, inplace=True)
#芝加哥VIX指數
data2 = tejapi.get('GLOBAL/GIDX',
                   coid = "SB82",
                          mdate = mdate,
                          paginate=True)
data2.set_index(data2["mdate"],inplace = True)
data = pd.merge(data,data2["val"],how='left', left_index=True, right_index=True)
data.rename({"val":"vix"}, axis=1, inplace=True)
#景氣對策訊號
data3 = tejapi.get('GLOBAL/ANMAR',
                   coid = "EA1101",
                          mdate = mdate,
                          paginate=True)
data3.set_index(data3["mdate"],inplace = True)
data3 = data3.resample('D').ffill()
data = pd.merge(data,data3["val"],how='left', left_index=True, right_index=True)
data.rename({"val":"light"}, axis=1, inplace=True)
#領先指標
data4 = tejapi.get('GLOBAL/ANMAR',
                   coid = "EB0101",
                          mdate = mdate,
                          paginate=True)
data4.set_index(data4["mdate"],inplace = True)
data4 = data4.resample('D').ffill()
data = pd.merge(data,data4["val"],how='left', left_index=True, right_index=True)
data.rename({"val":"advance"}, axis=1, inplace=True)


# In[55]:


#刪除空值
data.set_index(data["mdate"],inplace=True)
data = data.fillna(method="pad",axis=0)
data = data.dropna(axis=0)
del data["coid"]
del data["mdate"]


# In[48]:


data


# In[56]:


#定義趨勢(5,20,5,20)
data["short_mom"] = data["rsi"].rolling(window=10,min_periods=1,center=False).mean()
data["long_mom"] = data["rsi"].rolling(window=20,min_periods=1,center=False).mean()
data["short_mov"] = data["close"].rolling(window=10,min_periods=1,center=False).mean()
data["long_mov"] = data["close"].rolling(window=20,min_periods=1,center=False).mean()


# In[57]:


#標記Label
import numpy as np
data['label'] = np.where(data.short_mov > data.long_mov, 1, 0)
data = data.drop(columns=["short_mov"])
data = data.drop(columns=["long_mov"])
data = data.drop(columns=["short_mom"])
data = data.drop(columns=["long_mom"])


# In[58]:


#觀察資料分佈情形
import matplotlib.pyplot as plt
fig = plt.figure
plot = data.groupby(["label"]).size().plot(kind="barh",color="grey")


# In[ ]:


#資料前處理


# In[59]:


#標準化
X = data.drop('label', axis = 1)
from sklearn.preprocessing import StandardScaler
X[X.columns] = StandardScaler().fit_transform(X[X.columns])
y = pd.DataFrame({"label":data.label})


# In[60]:


# 切割成學習樣本以及測試樣本
import numpy as np
split = int(len(data)*0.7)
train_X = X.iloc[:split,:].copy()
test_X = X.iloc[split:].copy()
train_y = y.iloc[:split,:].copy()
test_y = y.iloc[split:].copy()

X_train, y_train, X_test, y_test = np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)


# In[61]:


#將資料改成三維
X_train = np.reshape(X_train, (X_train.shape[0],1,16))
y_train = np.reshape(y_train, (y_train.shape[0],1,1))
X_test = np.reshape(X_test, (X_test.shape[0],1,16))
y_test = np.reshape(y_test, (X_test.shape[0],1,1))


# In[62]:


test_X


# In[ ]:


#模型設計


# In[62]:


#載入套件
#import packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import BatchNormalization


# In[63]:


#加入模型
regressor = Sequential()
regressor.add(LSTM(units = 32, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(BatchNormalization())
regressor.add(Dropout(0.35))
regressor.add(LSTM(units = 32, return_sequences = True))
regressor.add(Dropout(0.35))
regressor.add(LSTM(units = 32, return_sequences = True))
regressor.add(Dropout(0.35))
regressor.add(LSTM(units = 32))
regressor.add(Dropout(0.35))
regressor.add(Dense(units = 1,activation="sigmoid"))
regressor.compile(optimizer = 'adam', loss="binary_crossentropy",metrics=["accuracy"])
regressor.summary()


# In[64]:


#模型結果
train_history = regressor.fit(X_train,y_train,
                          batch_size=200,
                          epochs=100,verbose=2,
                          validation_split=0.2)


# In[65]:


#模型評估
#畫出loss, val_loss
import matplotlib.pyplot as plt
loss = train_history.history["loss"]
var_loss = train_history.history["val_loss"]
plt.plot(loss,label="loss")
plt.plot(var_loss,label="val_loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("model loss")
plt.legend(["train","valid"],loc = "upper left")


# In[26]:


#變數重要性
from tqdm.notebook import tqdm
results = []
print(' Computing LSTM feature importance...')
# COMPUTE BASELINE (NO SHUFFLE)
oof_preds = regressor.predict(X_test, verbose=0).squeeze() 
baseline_mae = np.mean(np.abs(oof_preds-y_test))

results.append({'feature':'BASELINE','mae':baseline_mae})           

for k in tqdm(range(len(list(test_X.columns)))):
                
  # SHUFFLE FEATURE K
  save_col = X_test[:,:,k].copy()
  np.random.shuffle(X_test[:,:,k])
                        
  # COMPUTE OOF MAE WITH FEATURE K SHUFFLED
  oof_preds = regressor.predict(X_test, verbose=0).squeeze() 
  mae = np.mean(np.abs( oof_preds-y_test ))
  results.append({'feature':test_X.columns[k],'mae':mae})
  X_test[:,:,k] = save_col


# In[27]:


#變數重要性視覺化
import matplotlib.pyplot as plt

print()
df = pd.DataFrame(results)
df = df.sort_values('mae')
plt.figure(figsize=(10,20))
plt.barh(np.arange(len(list(test_X.columns))+1),df.mae)
plt.yticks(np.arange(len(list(test_X.columns))+1),df.feature.values)
plt.title('LSTM Feature Importance',size=16)
plt.ylim((-1,len(list(test_X.columns))+1))
plt.plot([baseline_mae,baseline_mae],[-1,len(list(test_X.columns))+1], '--', color='orange',
  label=f'Baseline OOF\nMAE={baseline_mae:.3f}')
plt.xlabel(f'Fold {1} OOF MAE with feature permuted',size=14)
plt.ylabel('Feature',size=14)
plt.legend()
plt.show()


# In[ ]:


#模型結果(測試集)


# In[66]:


#評估模型準確率
regressor.evaluate(X_test, y_test,verbose=1)


# In[67]:


#查看測試資料預測結果
predict_x = regressor.predict(X_test) 
df_predict = pd.DataFrame(predict_x,columns = ["Buy"])
df_predict["Action"] = np.where(df_predict["Buy"] > 0.5, 1, 0)
result = pd.DataFrame({"Close":data.iloc[split:]["close"]})
result["Real"] = test_y["label"]
result["Predict"] = list(df_predict["Action"])
result["mdate"] = result.index
result['mdate'] = pd.to_datetime(result['mdate'],format='%Y/%m/%d')
result.set_index(result["mdate"],inplace=True)
result


# In[ ]:


#策略視覺化


# In[68]:


#視覺化Predict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

df = result.copy()
df = df.resample('D').ffill()

t = mdates.drange(df.index[0], df.index[-1], dt.timedelta(hours = 24))
y = np.array(df.Close[:-1])

fig, ax = plt.subplots()
ax.plot_date(t, y, 'b-', color = 'black')
for i in range(len(df)):
    if df.Predict[i] == 1:
        ax.axvspan(
            mdates.datestr2num(df.index[i].strftime('%Y-%m-%d')) - 0.5,
            mdates.datestr2num(df.index[i].strftime('%Y-%m-%d')) + 0.5,
            facecolor = 'red', edgecolor = 'none', alpha = 0.5
            )
    else:
        ax.axvspan(
            mdates.datestr2num(df.index[i].strftime('%Y-%m-%d')) - 0.5,
            mdates.datestr2num(df.index[i].strftime('%Y-%m-%d')) + 0.5,
            facecolor = 'green', edgecolor = 'none', alpha = 0.5
            )
fig.autofmt_xdate()
fig.set_size_inches(20,10.5)


# In[69]:


#回測
test_data = data.iloc[split:].copy()
backtest = pd.DataFrame(index=result.index)
backtest["r_signal"] = list(test_data["label"])
backtest["p_signal"] = list(result["Predict"])
backtest["m_return"] = list(test_data["close"].pct_change())

backtest["r_signal"] = backtest["r_signal"].replace(0,-1)
backtest["p_signal"] = backtest["p_signal"].replace(0,-1)
backtest["a_return"] = backtest["m_return"]*backtest["r_signal"].shift(1)
backtest["s_return"] = backtest["m_return"]*backtest["p_signal"].shift(1)
backtest[["m_return","s_return","a_return"]].cumsum().hist()
backtest[["m_return","s_return","a_return"]].cumsum().plot()


# In[70]:


backtest[["m_return","s_return","a_return"]].cumsum()[-1:]

