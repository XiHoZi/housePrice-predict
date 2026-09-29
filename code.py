import pandas as pd
import numpy as np
import torch
import d2l.torch as d2l
import random
train_data = pd.read_csv('e:/vscode_code/Python/DL/train.csv')
test_data = pd.read_csv('e:/vscode_code/Python/DL/test.csv')
sample_submission = pd.read_csv('e:/vscode_code/Python/DL/sample_submission.csv')
def PreProcessData(df):
    #.columns是在获取前面的列名
    numData = df.select_dtypes(include=['int64','float64']).columns()
    for data in numData:
        median = df[data].median()
        df[data] = df[data].fillna(median)
    return df
train_data = PreProcessData(train_data)
test_data = PreProcessData(test_data)
 
def chuLiYiChangZhi(df,columns):
    for data in columns:
        meanVal = df[data].mean()
        stdVal = df[data].std()
        lowerOnes = meanVal - 3*stdVal
        upperOnes = meanVal + 3*stdVal
        medianVal = df[data].median()
        df.loc[df[data] < lowerOnes,data] = medianVal
        df.loc[df[data] > upperOnes,data] = medianVal
    return df
def DuiShuBianHuan(y):
    return np.log1p(y)
choseFeatures = ['OverallQual', 'OverallCond','MSSubClass','LotFrontage', 'LotArea','MasVnrArea',                     'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF','1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 
                     'GrLivArea','BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath','BedroomAbvGr', 'KitchenAbvGr', 
                     'TotRmsAbvGrd','Fireplaces','GarageYrBlt', 'GarageCars', 'GarageArea','WoodDeckSF', 'OpenPorchSF', 
                     'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea','MiscVal','MoSold', 'YrSold']

train_data = chuLiYiChangZhi(train_data,choseFeatures)
test_data = chuLiYiChangZhi(test_data,choseFeatures)
X_train = train_data[choseFeatures].values
y_train = train_data['SalePrice'].values
X_test = test_data[choseFeatures].values
y_train = DuiShuBianHuan(y_train)
def biaoZhunHuaFeatures(X_train,X_test):
    mean = np.mean(X_train,axis=0)
    std = np.std(X_train,axis=0)
    std[std == 0] = 1
    X_train_process = (X_train-mean)/std
    X_test_process = (X_test-mean)/std
    return  X_train_process,X_test_process

X_train_down,X_test_down = biaoZhunHuaFeatures(X_train,X_test)
X_train_tensor = torch.tensor(X_train_down,dtype=torch.float32)
y_train_tensor = torch.tensor(y_train,dtype=torch.float32)
y_train_tensor = y_train_tensor.view(-1,1)
X_test_tensor = torch.tensor(X_test_down,dtype=torch.float32)
test11 = X_train_down.shape()
featuresNum = test11[1]
w = torch.normal(0,0.01,size=(featuresNum,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)
lr = 0.01
num_epochs = 100
batch_size = 32
def data_iter(batch_size,features,labels):
    numExamples = len(features)
    indices = list(range(numExamples))
    random.shuffle(indices)
    for i in range(0,numExamples,batch_size):
        batch_indices = indices[i:min(i+batch_size,numExamples)]
        batch_indices_tensor = torch.tensor(batch_indices)
        batch_features = features[batch_indices_tensor]
        batch_labels = labels[batch_indices_tensor]
        yield batch_features,batch_labels
for epoch in range(num_epochs):
    iter111 = data_iter(batch_size,X_train_tensor,y_train_tensor)
    for X,y in iter111:
        y_hat = d2l.linreg(X,w,b)
        loss = d2l.squared_loss(y_hat,y).sum()
        loss.backward()
        d2l.sgd([w,b],lr,batch_size)
    if(epoch+1)%10 == 0:
        with torch.no_grad():
            train_l = d2l.squared_loss(d2l.linreg(X_train_tensor,w,b),y_train_tensor)
            print(f'epoch {epoch+1},loss {float(train_l.mean()):f}')
with torch.no_grad():
    y_test_pred = d2l.linreg(X_test_tensor,w,b)
    y_test_pred_numpy = y_test_pred.numpy()
    y_test_pred_real = np.expm1(y_test_pred_numpy)
    test_predictions = y_test_pred_real.flatten()
submission = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': test_predictions
})
submission.to_csv('e:/vscode_code/Python/DL/submission.csv', index=False)
