#encoding=utf8
import scipy.io as io
import os
import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch.optim as optim
import time
from nptdms import TdmsFile
from mpl_toolkits.mplot3d import Axes3D

class DownSample():
    def __init__(self,mat,TimeDownX,SampleDownX):
        '''
        :param mat:  输入数据矩阵
        :param TimeDownX: 时间序列降采样倍数, TimeDownX=10: 5kHz->500Hz
        :param SampleDownX: 样本数降采样倍数, SampleDownX=10: 每10个样本取1个
        '''
        self.data = mat
        self.TimeDownX = TimeDownX
        self.SampleDownX = SampleDownX

    def RowsDown(self,data):
        rows = range(0,data.shape[0],self.SampleDownX)  #样本数
        data = data[rows]
        return data

    def ColsDown(self,data):
        cols = range(0,data.shape[1],self.TimeDownX)  #时间序列降采样
        data = data[:,cols]
        return data

    def Down(self):
        data = self.data
        data = self.RowsDown(data)
        data = self.ColsDown(data)
        return data

def Letf_Right_Filp(mat):
    data = mat[::-1]
    return data

def cut(mat):
    #s时空矩阵填充
    mat[:,0:3000] = np.random.randn(100,3000)
    mat[:,12000:15000] = np.random.randn(100, 3000)
    return mat

# 数据准备
class Dataset(Data.Dataset):
    def __init__(self,dfPath):
        self.dfPath = dfPath
        self.paths = dfPath["path"].tolist()
        self.labels = dfPath["label"].tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        matpath = self.paths[item]
        mat = io.loadmat(matpath)["data"]

        mat = DownSample(mat,TimeDownX=10,SampleDownX=4).Down() #时空信号降采样

        mat = torch.from_numpy(mat).float()
        label = self.labels[item]
        #print("{}_th,mat size:{},label:{}".format(item,mat.size(),label))
        return mat,label

def collate_fn(batch):
    mats = []
    label = []

    for sample in batch:
        mats.append(sample[0])
        label.append(sample[1])

    mats_stack = torch.stack(mats,0)  #batch,mat.shape[0],mat.shape[1]
    mats_stack = torch.unsqueeze(mats_stack,1)  #batch, channels, mat.shape[0], mat.shape[1]
    label = torch.from_numpy(np.array(label))

    return mats_stack, label

def GenPath(path,label):
    PathList = []
    NameList = os.listdir(path)
    for name in NameList:
        matPath = os.path.join(path,name)
        PathList.append(matPath)
    df_path = pd.DataFrame(PathList,columns=["path"])
    df_path["label"] = label
    return df_path

def mat_sum(Mat):
    rows, cols = Mat.shape
    sum = 0
    for row in range(0,rows):
        for col in range(0,cols):
            sum += Mat[row,col]
    return sum

def Acc(Mat):

    rows,cols = Mat.shape
    false_mat1,false_mat2 = Mat[0:rows/2, cols/2:], Mat[rows/2:, 0:cols/2]
    sum, sum1, sum2 = mat_sum(Mat),mat_sum(false_mat1), mat_sum(false_mat2)
    acc_2_class = 1-float((sum1+sum2))/sum
    sum_diag = 0
    for i in range(rows):
        for j in range(cols):
            if i==j:
                sum_diag += Mat[i,j]
    acc_8_class = float(sum_diag)/sum

    return acc_8_class

def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp,axis=1,keepdims=True)
    s = x_exp/x_sum
    return s

def Predict(mat,model):

    threshold = 5.0
    if max(np.max(mat,axis=0)) <threshold:
        return 0

    data_cuda = torch.from_numpy(mat).float().cuda()
    data_cuda = data_cuda.unsqueeze(0)
    data_cuda = data_cuda.unsqueeze(0)
    output = model(data_cuda)
    prob = nn.functional.softmax(output).data.flatten().cpu().numpy()
    #print(prob)

    if max(prob)<0.6:  #抑制概率小于0.6的样本为噪声
        return 0

    label = torch.max(output,1)[1]
    return label

def plot3W(mat, name=None, savepath=None):
    size = mat.shape
    Y = np.arange(0, size[0], 1)
    X = np.arange(0, size[1], 1)
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, mat, cmap=plt.cm.hot)
    # plt.savefig(savepath+name+".jpg")
    # plt.close()
    plt.title("YaLu033331-250")
    plt.show()

if __name__ == "__main__":
    model = torch.load("../SaveModel/2d.pth")  #加载模型
    model.cuda()
    model.train(False)
    raw = "../test/"
    for file in os.listdir(raw):
        data = io.loadmat(raw+file)["data"]
        x = Predict(data,model)  #测试返回样本标签
        print(x)