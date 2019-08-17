#encoding=utf8

import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self,nIn,nHidden,nOut):
        super(BiLSTM,self).__init__()

        self.rnn = nn.LSTM(nIn,nHidden,bidirectional=True,dropout=0.3,batch_first=True)
        #self.classifier = nn.Linear(nHidden*2,nOut)   #T*h,num_classes
        self.classifier = nn.Linear(12800, nOut)  # T*h,num_classes

    def forward(self, x):

        recurrent,_ = self.rnn(x)
        b,T,h =recurrent.size()  # batch,seq_len,hidden

        outHidden = recurrent[:,:,:]  # batch,seq_len,hidden

        outHidden = outHidden.contiguous().view(b,-1)

        output = self.classifier(outHidden)
        return output


class CLNN(nn.Module):

    def __init__(self,NumClass):
        super(CLNN,self).__init__()

        self.NumClass = NumClass

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=(3,25),padding=(1,12),stride=(1,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pooling1 = nn.AvgPool2d(kernel_size=(1,25),stride=(1,25))  # (25,600)
        self.cnn1 = nn.Sequential(self.conv1,self.bn1,self.relu1,self.pooling1)

        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,25),padding=(1,12),stride=(1,1))
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pooling2 = nn.AvgPool2d(kernel_size=(1,25),stride=(1,25))
        self.cnn2 = nn.Sequential(self.conv2,self.bn2,self.relu2,self.pooling2) # (25,24)

        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,5),padding=(1,2),stride=(1,1))
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pooling3 = nn.AvgPool2d(kernel_size=(1,12),stride=(1,12))
        self.cnn3 = nn.Sequential(self.conv3,self.bn3,self.relu3,self.pooling3)  # (25,2)

        self.conv4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,5),padding=(1,2),stride=(1,1))
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pooling4 = nn.AvgPool2d(kernel_size=(1,2),stride=(1,2))  # (25,1)
        self.cnn4 = nn.Sequential(self.conv4,self.bn4,self.relu4,self.pooling4)

        self.rnn = BiLSTM(nIn=256,nOut=NumClass,nHidden=256)

    def forward(self, x):
        m1 = self.cnn1(x)
        m2 = self.cnn2(m1)
        m3 = self.cnn3(m2)
        m4 = self.cnn4(m3)

        b,c,h,w = m4.size()

        m4 = m4.squeeze(3)
        m4 = m4.permute(0,2,1)

        #features = torch.sum(m4,1)

        output = self.rnn(m4)

        return output

if __name__ == "__main__":
    model = CLNN(NumClass=5)
    print(model)
