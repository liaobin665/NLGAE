source of paper:
NLGAE: 一种基于改进网络结构及损失函数的图自编码器模型
（NLGAE: a Graph Auto-Encoder Model Based on Improved Network Structure and Loss Function）

依赖：
tensorflow：compat.v1
graphgallery：实验数据集来源
sklearn
scipy
pandas
numpy
xgboost


各文件夹功能：
getDataX:提取出各数据集的X
classfiy：分类性能实验
getEmbeddings：输入图数据，输出NLGAE模型的embedding
NLGAE：NLGAE模型---》config.py 模型配置；
                   nlgae.py 模型结构；
                   newtrainer.py 对trainer的功能强化；
                   saveBestEmbeddingtrainer.py 定制化的训练器，保存性能最优的embedding，并输出
parameterTurnning: 超参数扰动分析
saveBestEmbedding: 保存最优的嵌入信息
XGBresult：换分类器的实验，以（XGB）为例。