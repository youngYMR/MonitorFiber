# MonitorFiber
model.py --定义模型结构

defrom_conv.py --引用可形变卷积

Layers.py --定义可形变卷积层

train.py --模型训练过程

predict.py --模型预测

prune.py --根据BN层的gamma值作为特征选择依据

参考论文
--《Deformable Convolutional Networks》
论文原作者github连接：https://github.com/oeway/pytorch-deform-conv
DAS硬件系统采集时空数据的时候，会产生偏移量，利用可变形卷积网络去学习这些偏移量

--《Networks Slimming-Learning Efficient Convolutional Networks through Network Slimming》
利用BN层gamma因子，作为channel的权重，做特征选择，裁剪网络后，迁移特征权重，重新训练模型。达到模型压缩，提高模型计算速度。
