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
from model import CLNN
import torch.optim as optim


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

        sample = range(0,100,4)
        mat_sample = mat[sample,:]

        mat = torch.from_numpy(mat_sample).float()
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

    mats_stack = torch.unsqueeze(mats_stack,1)  #batch, channels,mat.shape[0], mat.shape[1]

    label = torch.from_numpy(np.array(label))

    return mats_stack, label

# 训练/验证过程
def train(model,epoch_num,start_epoch,optimizer,criterion,data_set,data_loader,save_dir,ReturnLine=True,Confusion=True):

    def adjust_lr(optimizer):
        for param in optimizer.param_groups:
            param["lr"] = param["lr"] / 10
        return optimizer

    LossLine = []
    AccLine = []

    for epoch in range(start_epoch,epoch_num):

        if epoch in [5,10,15,20]:
            optimizer = adjust_lr(optimizer)

        model.train(True)

        TrainAcc = 0.
        ValAcc = 0.
        TotalBacth = 0

        for batch_cnt,data in enumerate(data_loader['train']):

            bacthX,batchY = data

            bacthX = Variable(bacthX.cuda())
            batchY = Variable(batchY.cuda())
            out = model(bacthX)

            loss = criterion(out,batchY)

            _,pred = torch.max(out, 1)

            TrainCorrect = torch.sum((pred==batchY)).data
            TrainAcc += TrainCorrect
            TotalBacth += len(pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            LossLine.append(loss.data)
            AccLine.append(float(TrainAcc.data)/TotalBacth)

            #print("epoch-iteration:{}-{}, loss:{:.3f}, accuracy:{:.3f}").format(epoch,batch_cnt,loss.data,float(TrainAcc.data)/TotalBacth)

        #print("第{}训练完成！".format(epoch))

        if epoch%5 == 0:

            model.train(False)

            ValAcc = 0.
            ValBatch = 0
            pred_lst = []
            label_lst = []

            for val_cnt,val_data in enumerate(data_loader["val"]):
                valX,valY = val_data
                valX = Variable(valX.cuda())
                valY = Variable(valY.cuda())

                val_out = model(valX)
                val_loss = criterion(val_out,valY)

                val_pred = torch.max(val_out,1)[1]
                val_correct = (val_pred==valY).sum()
                ValAcc += val_correct

                ValBatch += len(val_pred)

                label_lst.extend(valY)  #  绘制混淆矩阵
                pred_lst.extend(val_pred)

            acc = float(ValAcc)/ValBatch
            #print("{},,epoch:{},validatin accuracy:{:.3f},loss:{:.3f}").format("*"*10,epoch,acc,val_loss)
            torch.save(model, os.path.join(save_dir, "{:.3f}_{}.pth".format(acc,epoch)))

            if Confusion :

                label_lst = [int(x) for x in label_lst]
                pred_lst = [int(x) for x in pred_lst]

                con_mat = confusion_matrix(label_lst, pred_lst)
                print (con_mat)

    if ReturnLine:
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        plt.title("loss")
        plt.plot(LossLine)
        ax2 = fig.add_subplot(122)
        plt.title("accuracy")
        plt.plot(AccLine)
        fig.savefig("./Line.jpg")

def GenPath(path,label):
    PathList = []
    NameList = os.listdir(path)
    for name in NameList:
        matPath = os.path.join(path,name)
        PathList.append(matPath)
    df_path = pd.DataFrame(PathList,columns=["path"])
    df_path["label"] = label
    return df_path


def DataPrepare(all_pd):

    all_pd = all_pd.sample(frac=1)
    train_pd, val_pd = train_test_split(all_pd, test_size=0.15, random_state=43, stratify=all_pd["label"])

    print ("训练集样本：",Counter(train_pd["label"]))
    print ("测试集样本：", Counter(val_pd["label"]))

    data_set = {}
    data_set["train"] = Dataset(train_pd)
    data_set["val"] = Dataset(val_pd)

    dataloader = {}
    dataloader["train"] = torch.utils.data.DataLoader(data_set["train"],batch_size=16,
                                                      shuffle=True,num_workers=2,collate_fn=collate_fn)
    dataloader["val"] = torch.utils.data.DataLoader(data_set["val"],batch_size=16,
                                                      shuffle=True,num_workers=2,collate_fn=collate_fn)
    return data_set,dataloader


if __name__ == "__main__":

    DataRawPath = "../data/"

    data0 = GenPath(DataRawPath+"背景噪声",label=0)
    data1 = GenPath(DataRawPath + "履带挖掘", label=1)
    data2 = GenPath(DataRawPath + "轮式挖掘", label=1)
    data3 = GenPath(DataRawPath + "破路机", label=2)
    data4 = GenPath(DataRawPath + "人工触览", label=3)
    data5 = GenPath(DataRawPath + "挖机怠速", label=4)

    df_Path = pd.concat([data0,data1,data2,data3,data4,data5],axis=0,ignore_index=True)

    data_set,data_loader = DataPrepare(df_Path)
    print ("data preparing is over!!!")

    save_path = "./model"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model = CLNN(NumClass=5)
    model.cuda()

    base_lr = 0.0001
    optimizer = optim.Adam(model.parameters(),lr=base_lr, weight_decay=0.00001)
    criterion = torch.nn.CrossEntropyLoss()

    train(model=model, epoch_num=21, start_epoch=0, optimizer=optimizer, criterion=criterion,
          data_set=data_set,data_loader=data_loader,save_dir=save_path)
