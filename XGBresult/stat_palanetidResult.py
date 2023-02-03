import argparse
from graphgallery.datasets import Planetoid
from graphgallery.datasets import KarateClub
from xgboost import XGBClassifier
from sklearn import preprocessing
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score,recall_score, f1_score,roc_auc_score,classification_report

# 设置为CPU运行
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from sklearn.model_selection import train_test_split

from NLGAE.nlgae.newtrainer import Model_Trainer as NLGAETrainer
from NLGAE.nlgae.config import get_NLGAE_config

# 得到NLGAE的算法超参数配置
configs = get_NLGAE_config()

# datasets = ['cora', 'citeseer', 'pubmed']

datasets = ['cora']

# 存放性能指标的数组
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

stat_pd = pd.read_csv("../../mydata/palanetoidStat/coraSata.csv")
# stat_pd = pd.read_csv("../../mydata/palanetoidStat/citeseerSata.csv")
# stat_pd = pd.read_csv("../../mydata/palanetoidStat/pubmedSata.csv")

# 得到节点全局统计信息数据
dataStatPD = stat_pd.drop(['node_id', 'label'], axis=1)
# 标准化处理
dataStatPD = preprocessing.minmax_scale(dataStatPD)

for dataName in datasets:

    # 得到图的原始数据，并从原始数据中得到结构信息G，节点特征信息X，节点标签信息Y
    # data = KarateClub('facebook', root="../../GraphData/datasets/", verbose=False)
    data = Planetoid(dataName, root="../../GraphData/datasets/", verbose=False)

    G = data.graph.adj_matrix
    # 将图原始属性 与 统计学特征 结合
    X = np.hstack((data.graph.node_attr, dataStatPD))
    X = np.matrix(X)
    Y = data.graph.node_label

    # 特征维度
    feature_dim = X.shape[1]

    configs.hidden_dims = [feature_dim] + configs.hidden_dims

    # 进一步得到图的结构信息：(indices, adj.data, adj.shape), adj.row, adj.col
    graphdataTuple, adjRow, adjCol = NLGAETrainer.prepare_graph_data(G)

    tf.reset_default_graph()
    # 创建模型训练器
    NLGAE_trainer = NLGAETrainer(configs)

    NLGAE_trainer(graphdataTuple, X, adjRow, adjCol)

    embeddings, attentions = NLGAE_trainer.infer(graphdataTuple, X, adjRow, adjCol)

    out =np.hstack((embeddings, Y.reshape(X.shape[0], 1)))

    total_pd = pd.DataFrame(out)

    total_pd = total_pd.sample(frac=1.0, random_state=99)

    # 训练数据Y
    Y = total_pd[total_pd.columns[-1]]
    # 训练数据X
    X = total_pd.iloc[:, 1:-1]

    # 标准化处理
    # x_scaled = preprocessing.scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=99)

    cls = XGBClassifier()
    cls.fit(X_train, y_train)

    report = classification_report(y_test, cls.predict(X_test), output_dict=True)

    df = pd.DataFrame(report).transpose()

    # 从classification_report读取到各性能指标
    accuracy_list.append(df.at['accuracy', 'f1-score'])
    precision_list.append(df.at['weighted avg', 'precision'])
    recall_list.append(df.at['weighted avg', 'recall'])
    f1_list.append(df.at['weighted avg', 'f1-score'])

    del NLGAE_trainer

performaceDF = pd.DataFrame({'dataName': datasets,
                                 'accuracy': accuracy_list,
                                 'precision': precision_list,
                                 'recall': recall_list,
                                 'f1': f1_list})

performaceFileName = "NAGAE_org_palanetid_XGB_result.csv"
performaceDF.to_csv(performaceFileName, index=True)




