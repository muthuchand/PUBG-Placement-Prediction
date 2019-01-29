#!/usr/bin/env python
# coding: utf-8

# **Import Libraries**

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")
import gc
import os
print(os.listdir("../input"))


# **Pre - processing function  - Normalizing using Standard Scaler**

from sklearn.preprocessing import StandardScaler
def normalize_data(data,is_test):
    orig_cols = list(data)    
    if is_test == 0:
        to_remove = ['Id','groupId','matchId','matchType','winPlacePerc']
    else:
        to_remove = ['Id','groupId','matchId','matchType']
    cols = [col for col in orig_cols if col not in to_remove]
    data[cols] = StandardScaler().fit_transform(data[cols])
    return data


# **Memory Reduction function**
# Ref :  https://www.kaggle.com/gemartin

def reduce_mem_usage(datafr):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    df = datafr.copy()
    start_mem = df.memory_usage().sum() 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
#     df = df.drop('matchType',axis = 1)
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            # print(col_type)
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


# **Adding and removing explicit features**

def addl_features(data):
    #Addition of features 
    print('Addl features..')
    gc.collect()
    data['headshotRate'] = data['kills']/(data['headshotKills']+1)
    data['killstreakRate'] = data['killStreaks']/(data['kills']+1)

    data['totalDist'] = data['rideDistance'] + data['swimDistance'] + data['walkDistance']
    data['weaponsDist']  =data['weaponsAcquired'] / (data['totalDist'] + 1)
    data['healsandboosts'] = data['heals'] + data['boosts']

    data['boostsperWalkdist'] = data['boosts'] / (data['walkDistance']+ 1)
    data['healsperWalkdist'] = data['heals'] / (data['walkDistance']+ 1)
    data['boostshealsperWalkdist'] = data['healsandboosts'] / (data['walkDistance']+ 1)
    
    data['killsandassists'] = data['kills'] + data['assists']
    #Removing certain colums 
    data.drop(['swimDistance','rideDistance','walkDistance'],axis = 1)
    data.drop(['roadKills','headshotKills','vehicleDestroys'],axis = 1)
    data.drop(['matchDuration','numGroups'],axis = 1)
    
    return data


# **Adding addition groupId and matchId based rank and other Features**

def feat_engg(data,is_test):
    print('Adding Additional group and match features')
    gc.collect()
#     data = reduce_mem_usage(data)
    #Useful features to be used 
    columns = list(data)
    if is_test == 0:
        feat_remove = {'Id','groupId','matchId','winPlacePerc','matchType'}
    else:
        feat_remove = {'Id','groupId','matchId','matchType'}   
    feat = [col for col in columns if col not in feat_remove]
   
    #Get group features    
    print('Adding group features')
    #Adding group mean 
    grp = data.groupby(['matchId','groupId'])[feat].agg('mean')
#     grp.columns = ["_group_".join(x) for x in grp.columns.ravel()]
#     grp.rename(columns={'matchId_group_': 'matchId','groupId_group_':'groupId'}, inplace=True)
    
    grp_rank = grp.groupby('matchId')[feat].rank(pct=True).reset_index()
    if is_test == 0:
        data_dup = grp.reset_index()[['matchId','groupId']]
    else: 
        data_dup = data[['matchId','groupId']]

    data_dup = pd.merge(data_dup,grp.reset_index(), how='left', on=['matchId', 'groupId'],suffixes=["", ""])
    data_dup = pd.merge(data_dup,grp_rank, how='left', on=['matchId', 'groupId'], suffixes=["_mean", "_mean_rank"])

    
    #Adding group standard deviation
    methods = ['max','min','var']
    for m in methods :
        grp = data.groupby(['matchId','groupId'])[feat_float].agg(m)
#         grp.columns = ["_group_".join(x) for x in grp.columns.ravel()]
#         grp.rename(columns={'matchId_group_': 'matchId','groupId_group_':'groupId'}, inplace=True)
        grp_rank = grp.groupby('matchId')[feat_float].rank(pct=True).reset_index()
        data_dup = pd.merge(data_dup,grp.reset_index(), how='left', on=['matchId', 'groupId'], suffixes=["", ""])
        data_dup = pd.merge(data_dup,grp_rank, how='left', on=['matchId', 'groupId'], suffixes=["_"+m, "_"+m+""])

    
#     #Get Match Features 
#     print(data_dup.shape)
    print('Adding match features')
    match = data.groupby('matchId')[feat].agg(['mean','min','max']).reset_index()
    match.columns = ["_".join(x) for x in match.columns.ravel()]
    match.rename(columns={'matchId_': 'matchId'}, inplace=True)
#     print('Match:',match.shape,list(match))
    data_dup= pd.merge(data_dup,match,on=['matchId'],how='left').reset_index()
    # print('Data_dup:',data_dup.shape,list(data_dup))
    del match
    gc.collect()
    return data_dup


# **Load Input Data**

print('Input Train Data..')
train_data = pd.read_csv('../input/train_V2.csv')
print(train_data.shape)
# train_data= train_data.dropna(subset=['winPlacePerc'])
train_data = train_data.dropna()
print(train_data.shape)
train_data[train_data == np.Inf] = np.NaN
# train_data[train_data == np.NINF] = np.NaN
print('Removing Nan..')
# print("Removing Na's From DF")
train_data.fillna(0, inplace=True)
# test_data.fillna(0,inplace = True)

train_data = normalize_data(train_data,0)
y_train = train_data['winPlacePerc']
x_train = train_data.iloc[:,:-1]
# x_test = test_data.copy()
print(x_train.shape,y_train.shape)
# print(x_test.shape)


# **Split training  into test_train and train_train**
# Split in ratio 0.15:0.85
sp = np.random.rand(len(train_data)) < 0.85
train_train = train_data[sp]
test_train = train_data[~sp]


# **Some EDA**
#Getting some statistics about average of some features
#Many plots are the same kind, except for the variables -change wherever needed 
data = train_train.copy()
train = train_train.copy()
print('Average no.of healing items used : {:.3f} '.format(train['heals'].mean()))
print('Average no.of kills : {:.3f} '.format(train['kills'].mean()))
print('Average no.of assists : {:.3f} '.format(train['assists'].mean()))
print('Average walking Distance : {:.3f} '.format(train['walkDistance'].mean()))
print('Average swim Distance : {:.3f} '.format(train['swimDistance'].mean()))
print('Average Ride Distance : {:.3f} '.format(train['rideDistance'].mean()))
print('Average no.of head shot kills : {:.3f} '.format(train['headshotKills'].mean()))
print('Average damage dealt : {:.3f} '.format(train['damageDealt'].mean()))

#Plot scatter plots  
#valx and valy change accordingly
valx = 'walkDistance'
valy = 'Heals'
y = data[valy]
x = data[valx]
plt.scatter(x,y)
plt.xlabel(valx)
plt.ylabel(valy)

#Heat map - Ref from Kaggle
plt.subplots(figsize=(15, 15))
sns.heatmap(train_train.corr(), annot=True)
plt.show()

#Target Distribution Plots
val = 'winPlacePerc'
data = data[data[val] < train_train[val].quantile(0.99)]
plt.figure(figsize=(8,6))
plt.title("Target Distribution",fontsize=15)
sns.distplot(data[val])
plt.show()

#Plotting point plots - Ref from Kaggle 
data = data[data['heals'] < data['heals'].quantile(0.99)]
data = data[data['boosts'] < data['boosts'].quantile(0.99)]

f,ax1 = plt.subplots(figsize =(10,5))
sns.pointplot(y='heals',x='winPlacePerc',data=data,color='red',alpha=0.8)
sns.pointplot(y='boosts',x='winPlacePerc',data=data,color='blue',alpha=0.8)
plt.text('Heals',color='red')
plt.text('Boosts',color='blue')
plt.ylabel('Number of items')
plt.xlabel('winPlacePerc')
plt.title('Heals and Boosts')
plt.grid()
plt.show()

#Change Parameters wherever needed
df = train_train.copy()
sns.pointplot(df["heals"],df["winPlacePerc"], color = 'blue',linestyles="-")
sns.pointplot(df["boosts"],df["winPlacePerc"], color = "red", linestyles="--")
plt.xlabel("Heals and Boosts")
plt.legend(["heals","boosts"]) 
plt.show()

df = train.copy()
sns.pointplot(df["heals"],df["walkDistance"], color = 'blue',linestyles="-")
sns.pointplot(df["boosts"],df["walkDistance"], color = "red", linestyles="--")
plt.xlabel("Heals and Boosts")
plt.legend(["heals","boosts"]) 
plt.show()


# **Use train_train for training**

train_data = train_train
gc.collect()
print('Reducing Memory..')
train_data = reduce_mem_usage(train_data)
train_data = addl_features(train_data)
# test_data = addl_features(test_data)
train_data_feat = feat_engg(train_data,0)
# train_data_feat = train_data
del train_data
gc.collect()

columns_tr = list(train_data_feat)
features_rem_tr = {'Id','groupId','matchId','matchType'}
features_rem_tr = [col for col in columns_tr if col not in features_rem_tr]
x_train = train_data_feat[features_rem_tr]
x_train = reduce_mem_usage(x_train)
gc.collect()
y_train = x_train['winPlacePerc']
x_train = reduce_mem_usage(x_train)

x_train.drop(columns=['winPlacePerc'],inplace = True)
gc.collect()
# list(x_train)


# *Comment all other models when running a particular model*

# **Model 1 : Light GBM**

import lightgbm as lgb
#Light GBM Model
def run_lgb(tr_x, tr_y, val_x, val_y):
    params = {"objective" : "regression", "metric" : "mae", 'n_estimators':20000, 'early_stopping_rounds':200,"num_leaves" : 27, "learning_rate" : 0.07, "bagging_fraction" : 0.7,"bagging_seed" : 0, "num_threads" : 4,"colsample_bytree" : 0.7,"max_depth" : 12}
    lgb_train = lgb.Dataset(tr_x, label=tr_y)
    lgb_val = lgb.Dataset(val_x, label=val_y)
    model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_val], early_stopping_rounds=200, verbose_eval=1000)
#    pred_y = model.predict(ts_x, num_iteration=model.best_iteration)
    return model


# **Model 2 : Linear Regression**
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x_tr1,x_val1,y_tr1,y_val1 = train_test_split(x_train,y_train, train_size = 0.85,test_size = 0.15)
linreg = LinearRegression()
linreg.fit(x_tr1,y_tr1)
y_pred_val = linreg.predict(x_val1)
#print('MAE:',mae(y_val1,y_pred_val))


# **Model 3 : RF Regressor**
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
x_tr1,x_val1,y_tr1,y_val1 = train_test_split(x_train,y_train, train_size = 0.85,test_size = 0.15)
#final values
model = RandomForestRegressor(n_estimators = 30, max_depth = 7)
model.fit(x_tr1,y_tr1)
y_pred_val = model.predict(x_val1)
#print('MAE:',mae(y_val1,y_pred_val))


# **Model 4 : XG Boost Regressor**
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
x_tr1,x_val1,y_tr1,y_val1 = train_test_split(x_train,y_train, train_size = 0.85,test_size = 0.15)
model = XGBRegressor(max_depth=5, learning_rate = 0.07)
#final values
model.fit(x_tr1,y_tr1)
y_pred_val = model.predict(x_val1)
#print('MAE:',mae(y_val1,y_pred_val))


# **Model 5 : Cat Boost Regressor**
import catboost
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
x_train = x_train.select_dtypes(include = [np.number])
x_tr1,x_val1,y_tr1,y_val1 = train_test_split(x_train,y_train, train_size = 0.85,test_size = 0.15)
#final values
model = CatBoostRegressor(learning_rate=0.07,iterations = 20000)
model.fit(x_tr1,y_tr1)
y_pred_val = model.predict(x_val1)
#print('MAE:',mae(y_val1,y_pred_val))


# **Calculating on the training data on Validation Sets**

from sklearn.model_selection import train_test_split
xtr,xval,ytr,yval = train_test_split(x_train,y_train,test_size = 0.25)
del x_train,y_train
gc.collect()
# pred, model = run_lgb(xtr,ytr, xval, yval, x_test)
model = run_lgb(xtr,ytr, xval, yval)


# **Calculation out of sample performance metrics on train_test**
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2score
train_test_data = test_train
train_test_data = normalize_data(train_test_data,0)
gc.collect()
train_test_data = addl_features(train_test_data)
train_test_data = feat_engg(train_test_data,0)
# del train_test_data
gc.collect()
#Remove categorical feat
columns_ts = list(train_test_data)
y_train_test = train_test_data['winPlacePerc']
features_rem_ts = {'Id','groupId','matchId','matchType','winPlacePerc'}
features_rem_ts = [col for col in columns_ts if col not in features_rem_ts]
# print('Removed cat feat')
x_train_test = train_test_data[features_rem_ts]
#
gc.collect()
x_train_test = reduce_mem_usage(x_train_test)
gc.collect()

y_train_test_pred = model.predict(x_train_test,num_iteration=model.best_iteration)
print('MAE Score :',mae(y_train_test,y_train_test_pred))
print('R2 Score :',r2score(y_train_test,y_train_test_pred))
del train_test_data,x_train_test,y_train_test
gc.collect()


# **Load Test Data and predict values**
print('Test:')
test_data = pd.read_csv('../input/test_V2.csv')
print('Reducing Memory..')
test_data = reduce_mem_usage(test_data)
test_data = normalize_data(test_data,1)
gc.collect()
test_data = addl_features(test_data)
test_data = feat_engg(test_data,1)
# del test_data
gc.collect()

#Remove categorical feat
test_data_feat = test_data.copy()
columns_ts = list(test_data_feat)
features_rem_ts = {'Id','groupId','matchId','matchType'}
features_rem_ts = [col for col in columns_ts if col not in features_rem_ts]
# print('Removed cat feat')
x_test = test_data_feat[features_rem_ts]
del test_data_feat
gc.collect()
x_test = reduce_mem_usage(x_test)
gc.collect()


# **Get predictions for the test data**
pred = model.predict(x_test,num_iteration=model.best_iteration)


# **Write to the file**
#Kaggle Reference
test_data = pd.read_csv('../input/test_V2.csv')
for i in range(len(test_data)):
    percentage = pred[i]
    maxplace_value = int(test_data.iloc[i]['maxPlace'])
    if maxplace_value == 0:
        percentage = 0.0
    elif maxplace_value == 1:
        percentage = 1.0
    else:
        tmp = 1.0 / (maxplace_value - 1)
        percentage = round(percentage / tmp) * tmp
    
    if percentage < 0: 
        percentage = 0.0
    if percentage > 1: 
        percentage = 1.0    
    pred[i] = percentage
    
test_id = test_data['Id']
sub_csvfile = pd.DataFrame({'Id': test_id, "winPlacePerc": pred} , columns=['Id', 'winPlacePerc'])
sub_csvfile.to_csv("submission_lgbm.csv", index = False)

