import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
import copy
import torch
import math
import networkx as nx
import torch_geometric
import torch.nn.functional as F
import pywt
import os
import re
import numpy.fft as fft
import random

from scipy.signal import savgol_filter
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from torch import nn
# from dataset1 import Mydataset
from torch.utils.data import DataLoader
# from earlystop import EarlyStopping
from torch_cluster import knn_graph
from torch_geometric.utils import degree,to_undirected,to_networkx
from torch_geometric.nn import GCNConv,BatchNorm
from scipy import special
from torch.utils.data import Dataset

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Mydataset(Dataset):
    def __init__(self, data,data_end,label):
        self.data = data
        self.data_end=data_end
        self.label = label
            
    # 根据索引获取data和label
    def __getitem__(self, index):
        return self.data[index], self.data_end[index], self.label[index]  # 以元组的形式返回

    # 获取数据集的大小
    def __len__(self):
        return len(self.data)



## 考虑了阈值V的不确定性，theta的不确定性和扩散系数的不确定性
seed=933#随机种子

random.seed(seed)
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
    
device = torch.device("cuda")    

## 读取数据

c1=np.load('c1_features_0.npy').astype('float32')
c2=np.load('c1_features_1.npy').astype('float32')
c3=np.load('c1_features_2.npy').astype('float32')
c4=np.load('c1_features_3.npy').astype('float32')
c5=np.load('c1_features_4.npy').astype('float32')
c6=np.load('c1_features_5.npy').astype('float32')
c7=np.load('c1_features_6.npy').astype('float32')
c8=np.load('c1_features_7.npy').astype('float32')
c9=np.load('c1_features_8.npy').astype('float32')
c10=np.load('c1_features.npy').astype('float32')
c11=np.load('c4_features_0.npy').astype('float32')
c12=np.load('c4_features_1.npy').astype('float32')
c13=np.load('c4_features_2.npy').astype('float32')
c14=np.load('c4_features_3.npy').astype('float32')
c15=np.load('c4_features_4.npy').astype('float32')
c16=np.load('c4_features_5.npy').astype('float32')
c17=np.load('c4_features_6.npy').astype('float32')
c18=np.load('c4_features_7.npy').astype('float32')
c19=np.load('c4_features_8.npy').astype('float32')
c20=np.load('c4_features.npy').astype('float32')
c21=np.load('c6_features_0.npy').astype('float32')
c22=np.load('c6_features_1.npy').astype('float32')
c23=np.load('c6_features_2.npy').astype('float32')
c24=np.load('c6_features_3.npy').astype('float32')
c25=np.load('c6_features_4.npy').astype('float32')
c26=np.load('c6_features_5.npy').astype('float32')
c27=np.load('c6_features_6.npy').astype('float32')
c28=np.load('c6_features_7.npy').astype('float32')
c29=np.load('c6_features_8.npy').astype('float32')
c30=np.load('c6_features.npy').astype('float32')

# 根据皮尔逊系数删去与退化趋势不太相关的退化特征
delete_index=np.array([ 1,  3,  5,  8, 10, 11, 12, 13, 14, 16, 17, 20])
c1=np.delete(c1, delete_index, 2)
c2=np.delete(c2, delete_index, 2)
c3=np.delete(c3, delete_index, 2) 
c4=np.delete(c4, delete_index, 2)
c5=np.delete(c5, delete_index, 2)
c6=np.delete(c6, delete_index, 2)
c7=np.delete(c7, delete_index, 2)
c8=np.delete(c8, delete_index, 2)
c9=np.delete(c9, delete_index, 2)
c10=np.delete(c10, delete_index, 2)
c11=np.delete(c11, delete_index, 2)
c12=np.delete(c12, delete_index, 2)
c13=np.delete(c13, delete_index, 2)
c14=np.delete(c14, delete_index, 2)
c15=np.delete(c15, delete_index, 2)
c16=np.delete(c16, delete_index, 2)
c17=np.delete(c17, delete_index, 2)
c18=np.delete(c18, delete_index, 2)
c19=np.delete(c19, delete_index, 2)
c20=np.delete(c20, delete_index, 2)
c21=np.delete(c21, delete_index, 2)
c22=np.delete(c22, delete_index, 2)
c23=np.delete(c23, delete_index, 2)
c24=np.delete(c24, delete_index, 2)
c25=np.delete(c25, delete_index, 2)
c26=np.delete(c26, delete_index, 2)
c27=np.delete(c27, delete_index, 2)
c28=np.delete(c28, delete_index, 2)
c29=np.delete(c29, delete_index, 2)
c30=np.delete(c30, delete_index, 2)

# 归一化
for i in range(7):
    for j in range(16):
        g1=np.expand_dims(c1[:,i,j],1)
        g2=np.expand_dims(c2[:,i,j],1)
        g3=np.expand_dims(c3[:,i,j],1)
        g4=np.expand_dims(c4[:,i,j],1)
        g5=np.expand_dims(c5[:,i,j],1)
        g6=np.expand_dims(c6[:,i,j],1)
        g7 = np.expand_dims(c7[:, i, j], 1)
        g8 = np.expand_dims(c8[:, i, j], 1)
        g9 = np.expand_dims(c9[:, i, j], 1)
        g10 = np.expand_dims(c10[:, i, j], 1)
        g11 = np.expand_dims(c11[:, i, j], 1)
        g12 = np.expand_dims(c12[:, i, j], 1)
        g13 = np.expand_dims(c13[:, i, j], 1)
        g14 = np.expand_dims(c14[:, i, j], 1)
        g15 = np.expand_dims(c15[:, i, j], 1)
        g16 = np.expand_dims(c16[:, i, j], 1)
        g17 = np.expand_dims(c17[:, i, j], 1)
        g18 = np.expand_dims(c18[:, i, j], 1)
        g19 = np.expand_dims(c19[:, i, j], 1)
        g20 = np.expand_dims(c20[:, i, j], 1)
        g21 = np.expand_dims(c21[:, i, j], 1)
        g22 = np.expand_dims(c22[:, i, j], 1)
        g23 = np.expand_dims(c23[:, i, j], 1)
        g24 = np.expand_dims(c24[:, i, j], 1)
        g25 = np.expand_dims(c25[:, i, j], 1)
        g26 = np.expand_dims(c26[:, i, j], 1)
        g27 = np.expand_dims(c27[:, i, j], 1)
        g28 = np.expand_dims(c28[:, i, j], 1)
        g29 = np.expand_dims(c29[:, i, j], 1)
        g30 = np.expand_dims(c30[:, i, j], 1)
        c1[:,i,j] = np.squeeze(MinMaxScaler().fit_transform(g1),1)  # 归一化
        c2[:,i,j] = np.squeeze(MinMaxScaler().fit_transform(g2),1)  # 归一化
        c3[:,i,j] = np.squeeze(MinMaxScaler().fit_transform(g3),1)  # 归一化
        c4[:,i,j] = np.squeeze(MinMaxScaler().fit_transform(g4),1)  # 归一化
        c5[:,i,j] = np.squeeze(MinMaxScaler().fit_transform(g5),1)  # 归一化
        c6[:,i,j] = np.squeeze(MinMaxScaler().fit_transform(g6),1)  # 归一化
        c7[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g7),1)# 归一化
        c8[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g8),1)# 归一化
        c9[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g9),1)# 归一化
        c10[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g10),1)# 归一化
        c11[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g11),1)# 归一化
        c12[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g12),1)# 归一化
        c13[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g13),1)# 归一化
        c14[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g14),1)# 归一化
        c15[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g15),1)# 归一化
        c16[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g16),1)# 归一化
        c17[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g17),1)# 归一化
        c18[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g18),1)# 归一化
        c19[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g19),1)# 归一化
        c20[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g20),1)# 归一化
        c21[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g21),1)# 归一化
        c22[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g22),1)# 归一化
        c23[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g23),1)# 归一化
        c24[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g24),1)# 归一化
        c25[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g25),1)# 归一化
        c26[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g26),1)# 归一化
        c27[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g27),1)# 归一化
        c28[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g28),1)# 归一化
        c29[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g29),1)# 归一化
        c30[:, i, j] = np.squeeze(MinMaxScaler().fit_transform(g30),1)# 归一化


# ## KNN建图部分 ##
edge_index=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
         5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6],
        [3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 0, 1, 2, 4, 5, 6, 0, 1, 2, 3, 5, 6,
         0, 1, 2, 3, 4, 6, 0, 1, 2, 3, 4, 5]])
edge_index=edge_index.to(device)
# print(edge_index)



    
# 固定窗口长度为115，设置每把刀的首点取值范围为1-201
c1_600=c1[0:200]
c2_600=c2[0:200]
c3_600=c3[0:200]
c4_600=c4[0:200]
c5_600=c5[0:200]
c6_600=c6[0:200]
c7_600=c7[0:200]
c8_600=c8[0:200]
c9_600=c9[0:200]
c10_600=c10[0:200]
c11_600=c11[0:200]
c12_600=c12[0:200]
c13_600=c13[0:200]
c14_600=c14[0:200]
c15_600=c15[0:200]
c16_600=c16[0:200]
c17_600=c17[0:200]
c18_600=c18[0:200]
c19_600=c19[0:200]
c20_600=c20[0:200]
c21_600=c21[0:200]
c22_600=c22[0:200]
c23_600=c23[0:200]
c24_600=c24[0:200]
c25_600=c25[0:200]
c26_600=c26[0:200]
c27_600=c27[0:200]
c28_600=c28[0:200]
c29_600=c29[0:200]
c30_600=c30[0:200]

# 将上面的数据转换成张量形式
c1_600=torch.tensor(c1_600).repeat(1,1,1).to(device)
c2_600=torch.tensor(c2_600).repeat(1,1,1).to(device)
c3_600=torch.tensor(c3_600).repeat(1,1,1).to(device)
c4_600=torch.tensor(c4_600).repeat(1,1,1).to(device)
c5_600=torch.tensor(c5_600).repeat(1,1,1).to(device)
c6_600=torch.tensor(c6_600).repeat(1,1,1).to(device)
c7_600=torch.tensor(c7_600).repeat(1,1,1).to(device)
c8_600=torch.tensor(c8_600).repeat(1,1,1).to(device)
c9_600=torch.tensor(c9_600).repeat(1,1,1).to(device)
c10_600=torch.tensor(c10_600).repeat(1,1,1).to(device)
c11_600=torch.tensor(c11_600).repeat(1,1,1).to(device)
c12_600=torch.tensor(c12_600).repeat(1,1,1).to(device)
c13_600=torch.tensor(c13_600).repeat(1,1,1).to(device)
c14_600=torch.tensor(c14_600).repeat(1,1,1).to(device)
c15_600=torch.tensor(c15_600).repeat(1,1,1).to(device)
c16_600=torch.tensor(c16_600).repeat(1,1,1).to(device)
c17_600=torch.tensor(c17_600).repeat(1,1,1).to(device)
c18_600=torch.tensor(c18_600).repeat(1,1,1).to(device)
c19_600=torch.tensor(c19_600).repeat(1,1,1).to(device)
c20_600=torch.tensor(c20_600).repeat(1,1,1).to(device)
c21_600=torch.tensor(c21_600).repeat(1,1,1).to(device)
c22_600=torch.tensor(c22_600).repeat(1,1,1).to(device)
c23_600=torch.tensor(c23_600).repeat(1,1,1).to(device)
c24_600=torch.tensor(c24_600).repeat(1,1,1).to(device)
c25_600=torch.tensor(c25_600).repeat(1,1,1).to(device)
c26_600=torch.tensor(c26_600).repeat(1,1,1).to(device)
c27_600=torch.tensor(c27_600).repeat(1,1,1).to(device)
c28_600=torch.tensor(c28_600).repeat(1,1,1).to(device)
c29_600=torch.tensor(c29_600).repeat(1,1,1).to(device)
c30_600=torch.tensor(c30_600).repeat(1,1,1).to(device)

#设置训练集
data=torch.cat((c1_600,c2_600,c3_600,c4_600,c5_600,c6_600,c7_600,c10_600,c11_600,c12_600,c13_600,c14_600,c15_600,c16_600,c17_600,c20_600,c21_600,c22_600,c23_600,c24_600,c25_600,c26_600,c27_600,c28_600),0)


# 固定窗口长度为115，设置每把刀的尾点取值范围为115-315
c1_315th=c1[115:315]
c2_315th=c2[115:315]
c3_315th=c3[115:315]
c4_315th=c4[115:315]
c5_315th=c5[115:315]
c6_315th=c6[115:315]
c7_315th=c7[115:315]
c8_315th=c8[115:315]
c9_315th=c9[115:315]
c10_315th=c10[115:315]
c11_315th=c11[115:315]
c12_315th=c12[115:315]
c13_315th=c13[115:315]
c14_315th=c14[115:315]
c15_315th=c15[115:315]
c16_315th=c16[115:315]
c17_315th=c17[115:315]
c18_315th=c18[115:315]
c19_315th=c19[115:315]
c20_315th=c20[115:315]
c21_315th=c21[115:315]
c22_315th=c22[115:315]
c23_315th=c23[115:315]
c24_315th=c24[115:315]
c25_315th=c25[115:315]
c26_315th=c26[115:315]
c27_315th=c27[115:315]
c28_315th=c28[115:315]
c29_315th=c29[115:315]
c30_315th=c30[115:315]

# 将上面的数据转换成张量形式
c1_315th=torch.tensor(c1_315th).repeat(1,1,1).to(device)
c2_315th=torch.tensor(c2_315th).repeat(1,1,1).to(device)
c3_315th=torch.tensor(c3_315th).repeat(1,1,1).to(device)
c4_315th=torch.tensor(c4_315th).repeat(1,1,1).to(device)
c5_315th=torch.tensor(c5_315th).repeat(1,1,1).to(device)
c6_315th=torch.tensor(c6_315th).repeat(1,1,1).to(device)
c7_315th=torch.tensor(c7_315th).repeat(1,1,1).to(device)
c8_315th=torch.tensor(c8_315th).repeat(1,1,1).to(device)
c9_315th=torch.tensor(c9_315th).repeat(1,1,1).to(device)
c10_315th=torch.tensor(c10_315th).repeat(1,1,1).to(device)
c11_315th=torch.tensor(c11_315th).repeat(1,1,1).to(device)
c12_315th=torch.tensor(c12_315th).repeat(1,1,1).to(device)
c13_315th=torch.tensor(c13_315th).repeat(1,1,1).to(device)
c14_315th=torch.tensor(c14_315th).repeat(1,1,1).to(device)
c15_315th=torch.tensor(c15_315th).repeat(1,1,1).to(device)
c16_315th=torch.tensor(c16_315th).repeat(1,1,1).to(device)
c17_315th=torch.tensor(c17_315th).repeat(1,1,1).to(device)
c18_315th=torch.tensor(c18_315th).repeat(1,1,1).to(device)
c19_315th=torch.tensor(c19_315th).repeat(1,1,1).to(device)
c20_315th=torch.tensor(c20_315th).repeat(1,1,1).to(device)
c21_315th=torch.tensor(c21_315th).repeat(1,1,1).to(device)
c22_315th=torch.tensor(c22_315th).repeat(1,1,1).to(device)
c23_315th=torch.tensor(c23_315th).repeat(1,1,1).to(device)
c24_315th=torch.tensor(c24_315th).repeat(1,1,1).to(device)
c25_315th=torch.tensor(c25_315th).repeat(1,1,1).to(device)
c26_315th=torch.tensor(c26_315th).repeat(1,1,1).to(device)
c27_315th=torch.tensor(c27_315th).repeat(1,1,1).to(device)
c28_315th=torch.tensor(c28_315th).repeat(1,1,1).to(device)
c29_315th=torch.tensor(c29_315th).repeat(1,1,1).to(device)
c30_315th=torch.tensor(c30_315th).repeat(1,1,1).to(device)

#设置训练集
data_end=torch.cat((c1_315th,c2_315th,c3_315th,c4_315th,c5_315th,c6_315th,c7_315th,c10_315th,c11_315th,c12_315th,c13_315th,c14_315th,c15_315th,c16_315th,c17_315th,c20_315th,c21_315th,c22_315th,c23_315th,c24_315th,c25_315th,c26_315th,c27_315th,c28_315th),0)

# 设置寿命的标签
label=np.expand_dims(np.flip(np.arange(116,316,1)),1).copy()
label=np.array(label,dtype='float32')
label=torch.tensor(label).repeat(24,1).to(device)

## 构建训练数据集
train_data = Mydataset(data,data_end,label)
train_size=100
train_data = DataLoader(train_data, batch_size=train_size,shuffle=True)

# 创建网络模型
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.gcn1=GCNConv(16, 8)
        self.gcn2=GCNConv(8, 8)
        # self.gcn3=GCNConv(8, 8)
        
        self.line=nn.Sequential(
            nn.Flatten(),
            nn.Linear(56, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )
        # 阈值的期望
        self.miu_w = nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        #阈值的方差
        self.sigma_w=nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        # 初始化网络参数
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.miu_w.size(0))
        # 设置阈值的期望初值范围为1.5-2
        self.miu_w.data.uniform_(1.5, 2)
        # 设置阈值方差的初值
        self.sigma_w.data.uniform_(0, stdv)

    def forward(self, x1, edge_index):
        x1 = self.gcn1(x1, edge_index)
        x1 = F.relu(x1)
        x1 = self.gcn2(x1, edge_index)
        x1 = F.relu(x1)

        x1 = self.line(x1)
        return x1

#     调用GCN的类
gcn = GCN()
gcn=gcn.to(device)

# 损失函数
loss_fn = nn.MSELoss()
loss_fn=loss_fn.to(device)

# 优化器
learning_rate = 2e-3
optimizer = torch.optim.Adam(gcn.parameters(),lr=learning_rate,weight_decay=0.05)


# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch =800

# 初始化训练、测试的损失和测试的准确率
loss_train =[]
loss_test  =[]
accuracy_test=[]

a1=0

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))
    
    # 训练步骤开始
    gcn.train()
    total_train_loss=0
    total_train_accuracy=0
    
    jishu=0
    for x in train_data:
        jishu=jishu+1
        data,data_end,label=x
        # 首点的复合退化性能指标
        data=gcn(data, edge_index)
        # 尾点的复合退化性能指标
        data_end=gcn(data_end, edge_index)
        # 计算漂移系数的期望
        miu_theta=(data_end-data)/115
        # 计算寿命的预测值
        T_est=(gcn.miu_w-data)/miu_theta
        
        
        ##计算准确度.
        error_train=label-T_est
        zero = torch.zeros_like(error_train)
        one = torch.ones_like(error_train)
        error_train = torch.where(error_train < -10, zero, error_train)
        error_train = torch.where(error_train > 13, zero, error_train)
        train_accuracy = torch.count_nonzero(error_train)
        total_train_accuracy = total_train_accuracy + train_accuracy.item()  

        #损失函数
        loss = loss_fn((T_est, label)/train_size \
            +(1/(100*torch.maximum(torch.tensor(0).to(device),torch.mean(data_end-gcn.miu_w))+0.0001))\
            +c+0.0001)
        # if i>=25:
        #     loss = torch.sigmoid(gcn.gamma)*loss_fn(T_est, label_train)/train_size + (1-torch.sigmoid(gcn.gamma))*torch.log10(1+loss_fn(std,T_std))\
        #         +(1/(100*torch.maximum(torch.tensor(0).to(device),torch.mean(data_train_mowei-gcn.miu_w))+0.01)) 
        # + 0.01*(1-torch.sigmoid(gcn.gamma))*loss_fn(V_est,V)/train_size


        # 训练总损失
        total_train_loss = total_train_loss + loss.item()

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 50 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))

    # （训练集数量*滑窗次数）/100
    total_train_accuracy = total_train_accuracy/48
    # if (total_train_accuracy>a1):
    #     a1=total_train_accuracy
    #     a2=i
    # total_test_accuracy = total_test_accuracy*100/5000
    print("训练集准确率 {}%".format(total_train_accuracy))
    # print("测试集准确率 {}%".format(total_test_accuracy))


    # 存储训练的总损失
    loss_train.append(total_train_loss)
    # # loss_test.append(total_test_loss)
    # # accuracy_test.append(total_test_accuracy)
    
    if i % 5 == 0:
        c=[c30]
        Te=np.zeros([300,1])
        i=0
        dao_rul=np.flip(np.arange(1,316,1)).copy()
        dao_rul=torch.unsqueeze(torch.tensor(dao_rul), 1).to(device)

        for d in c:
            dao=torch.tensor(d).to(device)
            dao_end=torch.unsqueeze(torch.tensor(d[314,:,:]),0).to(device)

            d1b=gcn(dao,edge_index)
            d1e=gcn(dao_end,edge_index)

            miu=(d1e-d1b)/dao_rul
            T=((gcn.miu_w-d1b)/miu)
            Te[:,i]=np.squeeze(T[0:300].cpu().detach().numpy(),1)
            i=i+1

        dao_rua=dao_rul[0:300].cpu().detach().numpy()

        plt.figure(1)
        plt.title('刀的寿命曲线')
        plt.subplot(211)
        plt.plot(Te[:,0])
        plt.plot(dao_rua)
        plt.xlabel('c1走刀数')
        plt.ylabel('c1寿命')
        
        
        test1=c30
        lll=test1.shape[0]
        onehot=np.ones((lll, 1))
    
        test1=torch.Tensor(test1).to(device)
        xx2 = gcn(test1,edge_index)
        xx2=torch.Tensor.cpu(xx2).detach().numpy()
        threshold=torch.Tensor.cpu(gcn.miu_w).detach().numpy()
        onehot=threshold*onehot
    
        plt.subplot(212)
        plt.plot(xx2)
        plt.plot(onehot)
        plt.show()
    
    
    

# # 设置测试集
# test = c30
# lll=test.shape[0]
# onehot=np.ones((lll, 1))
# test=torch.Tensor(test).to(device)
# # 计算测试集的复合退化性能指标
# xx2 = gcn(test,edge_index)
# xx2=torch.Tensor.cpu(xx2).detach().numpy()
# #计算失效阈值
# threshold=torch.Tensor.cpu(gcn.miu_w).detach().numpy()
# onehot=threshold*onehot
# # 作出失效阈值和复合退化性能指标关系曲线
# plt.plot(xx2)
# plt.plot(onehot)
# plt.figure(1)
# plt.show()

# # 计算剩余寿命
# c=[c30]
# Te=np.zeros([300,1])
# i=0
# dao_rul=np.flip(np.arange(1,316,1)).copy()
# dao_rul=torch.unsqueeze(torch.tensor(dao_rul), 1).to(device)

# for d in c:
#     dao=torch.tensor(d).to(device)
#     dao_end=torch.unsqueeze(torch.tensor(d[314,:,:]),0).to(device)
#     d1b=gcn(dao,edge_index)
#     d1e=gcn(dao_end,edge_index)
#     miu=(d1e-d1b)/dao_rul
#     T=((gcn.miu_w-d1b)/miu)
#     Te[:,i]=np.squeeze(T[0:300].cpu().detach().numpy(),1)
#     i=i+1
# dao_rua=dao_rul[0:300].cpu().detach().numpy()
# # 作出实际寿命与预测寿命关系曲线
# plt.figure(2)
# plt.title('刀的寿命曲线')
# plt.plot(Te[:,0])
# plt.plot(dao_rua)
# plt.xlabel('c30走刀数')
# plt.ylabel('c30寿命')
# plt.show()

print("CHI",d1b.cpu().detach().numpy())#CHI
print("真实寿命",dao_rua)
print("预测寿命",Te)
print("阈值",gcn.miu_w)
# 计算漂移系数的期望
data_end = c30[314,:,:]
data = c30[0,:,:]
data_end=torch.unsqueeze(torch.tensor(data_end),0).to(device)
data=torch.unsqueeze(torch.tensor(data),0).to(device)
data_end = gcn(data_end,edge_index)
data = gcn(data,edge_index)
miu_theta=(data_end-data)/315
print("lambda期望",miu_theta)
# 计算性能评估指标
err=dao_rua-Te
####RMSE
MSE=np.sum(err**2)/100
####Accuracy
jishu=np.zeros([300,1])
acc = np.where(err < -10, jishu, err)
acc = np.where(err >13, jishu, acc)
acc = np.count_nonzero(acc)
Accuracy= acc*100/300
####Score
scoz=err[np.where(err>0)]
scof=err[np.where(err<0)]
sz=np.sum(np.exp(scoz/10)-1)
sf=np.sum(np.exp(-scof/10)-1)
score=(sz+sf)
print(MSE,score,Accuracy)

