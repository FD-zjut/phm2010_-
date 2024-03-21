import numpy as np
import pandas as pd
import pywt
import os
import re
import numpy.fft as fft


## 提取28个特征
features=np.zeros((315,7,28))
path = "D:/中间估计器/2023年度工作/数模联动/RUL/PHM 2010/c1"  # 读取csv文件目录路径
# listdir()--返回path指定 的 文件夹中包含的文件或者文件夹名字 的 列表
FileNames = os.listdir(path)# 因此Filename是一个列表

jishu=0
for fn in FileNames:
    # re.search(pattern, string, flags=0) 扫描整个字符串并返回第一个成功的匹配
    # pattern：匹配的正则表达式
    # string：要匹配的字符串
    # flags：表达式，用于控制正则表达式的匹配方式
    if re.search(r'\.csv$', fn):
        fullfilename = os.path.join(path, fn)
        df = pd.read_csv(fullfilename,encoding='utf-8',on_bad_lines='skip',index_col=False, header=None)

        #时域量纲特征
        df_max=df.max()      #最大值0
        df_min=df.min()      #最小值1
        df_peak=df.abs().max() #峰值2
        df_aa=df.abs().mean() #绝对平均值3
        df_ppv=df_max-df_min #峰峰值4
        df_mean=df.mean()    #平均值5
        df_var=df.var()      #方差6
        df_std=df.std()      #标准差7
        df_rms=np.sqrt((df**2).mean()) #有效值   8
        df_ra=np.sqrt(df.abs()).mean() #方根幅值9

        #时域无量纲特征
        df_skew=df.skew()              #偏度10
        df_kurt=df.kurt()              #峭度11
        df_boxing=df_rms/df_aa         #波形因子12
        df_fengzhi=df_max/df_rms       #峰值因子13
        df_maichong=df_max/df_aa       #脉冲因子14
        df_yudu = df_max/df_ra         #裕度因子15
        
        #频域特征
        Fs=50000 #采样频率
        T = 1 / Fs  # 采样周期
        L = len(df)  # 信号长度
        t = [i * T for i in range(L)]
        t = np.array(t)
        
        S=np.zeros([4,7])
        E=np.zeros([8,7])
        for i in range(len(df.T)):
            # 快速傅里叶变换，返回结果为1000个复数，复数的模为幅值；复数的角度为相频
            complex_array = fft.fft(df[i])
            freqs = fft.fftfreq(t.size, t[1] - t[0])  # 从0开始,既有正的频率，又有负的，只取一半即可

            # # 复数的模为信号的振幅（能量大小）
            pows = 2*np.abs(complex_array) / L
            pows[freqs == 0] = pows[freqs == 0] / 2  # 频率为0的重复累积了两次，需要减半处理
            pows = pows[freqs >= 0]  # 取出对应的正频率的幅值
            freqs = freqs[freqs >= 0]
            
            ps_values = pows**2 #功率谱

            #重心频率16
            S[0,i]=np.dot(ps_values,freqs)/np.sum(ps_values) 
            #均方根频率17
            S[1,i]=np.sqrt(S[0,i])
            #频率方差18
            S[2,i]=np.dot(np.multiply(freqs-S[0,i],freqs-S[0,i]),ps_values)/np.sum(ps_values)
            #频率标准差19
            S[3,i]=np.sqrt(S[2,i])
            
            #小波包能量系数20-27
            wp = pywt.WaveletPacket(data=df[i], wavelet='db1',mode='symmetric',maxlevel=3) #选用db1小波，分解层数为3

            n = 3
            ce = []  #第n层所有节点的分解系数
            for k in [node.path for node in wp.get_level(n, 'freq')]:
                ce.append(wp[k].data)
                #第n层能量特征
            en = []
            for kk in ce:
                en.append(pow(np.linalg.norm(kk,ord=None),2))
            E[:,i]=en    
        
        
        feature= pd.DataFrame([df_max, df_min, df_peak,df_aa,df_ppv,df_mean,df_var,df_std,df_rms,df_ra \
                                ,df_skew,df_kurt,df_boxing,df_fengzhi,df_maichong,df_yudu])
        feature=np.array(feature)
        feature=np.concatenate([feature,S,E]).T
        features[jishu,:,:]=feature
        jishu=jishu+1
        
# np.save('c6_features',features)


# c1=np.load('c1_features.npy').astype('float32') 
# 皮尔逊系数选取特征
score=np.zeros([1,28])
t=np.flip(np.arange(1,316,1))
c1_features=np.load('c1_features.npy')
# for i in range(c1_features.shape[2]):
#     score[:,i]=np.dot(t-np.mean(t),c1_features[:,0,i]-np.mean(c1_features[:,0,i]))\
#               /np.sqrt((np.sum((t-np.mean(t))**2)*np.sum((c1_features[:,0,i]-np.mean(c1_features[:,0,i]))**2)))
# score=np.abs(score)

delete_index=np.array([ 1,  3,  5,  8, 10, 11, 12, 13, 14, 16, 17, 20])

c1_features=np.delete(c1_features, delete_index, 2)

# import scipy.io
# scipy.io.savemat('C6.mat', mdict={'data': c6_features})

