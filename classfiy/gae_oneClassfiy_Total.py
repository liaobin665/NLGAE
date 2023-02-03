# 单算法的 详细 分类结果展示
#导入机器学习算法库
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score,recall_score, f1_score,roc_auc_score,classification_report

from sklearn.model_selection import train_test_split

# 读取带标签的数据
#total_pd = pd.read_csv("../getEmbeddings/cora512.csv")
# total_pd = pd.read_csv("../gae/getGAEembedding/gae_cora256.csv")
cora_pd = pd.read_csv("../gae/getGAEembedding/gae_cora32.csv")
citeseer_pd = pd.read_csv("../gae/getGAEembedding/gae_citeseer50.csv")
pubmed_pd = pd.read_csv("../gae/getGAEembedding/gae_pubmed32.csv")
# amazon_cs_pd = pd.read_csv("../../mydata/NPZEmbeding/amazon_cs_deepwalk_Embeding_256wei.csv")
dblp_pd = pd.read_csv("../gae/getGAEembedding/gae_dblp32.csv")
twitch_pd = pd.read_csv("../gae/getGAEembedding/gae_twitch_512.csv")
# uai_pd = pd.read_csv("../../mydata/NPZEmbeding/uai_deepwalk_Embeding_256wei.csv")

# dataSetNames = ['cora','citeseer']
dataSetNames = ['cora','citeseer','pubmed','dblp','twitch']

dataSets = []
dataSets.append(cora_pd)
dataSets.append(citeseer_pd)
dataSets.append(pubmed_pd)
# dataSets.append(amazon_cs_pd)
dataSets.append(dblp_pd)
dataSets.append(twitch_pd)
# dataSets.append(uai_pd)

# 存放性能指标的数组
accuracy_list = []
precision_list = []
precision_macro_list = []
recall_list = []
recall_macro_list = []
f1_list = []
f1_macro_list = []
trainTime_list = []

for total_pd in dataSets:
    # 按100%的比例抽样即达到打乱数据的效果
    total_pd = total_pd.sample(frac=1.0, random_state=99)
    # 训练数据Y
    Y = total_pd[total_pd.columns[-1]]
    # 训练数据X
    X = total_pd.iloc[:, 1:-1]
    # 标准化处理
    x_scaled = preprocessing.scale(X)
    X_train, X_test, y_train, y_test = train_test_split(x_scaled, Y, train_size=0.8, test_size=0.2, random_state=99)
    cls = LogisticRegression()
    cls.fit(X_train, y_train)
    report = classification_report(y_test, cls.predict(X_test), output_dict=True)

    df = pd.DataFrame(report).transpose()
    # 从classification_report读取到各性能指标
    accuracy_list.append(precision_score(y_test,cls.predict(X_test),average='micro'))

    # accuracy_list.append(df.at['accuracy', 'f1-score'])
    precision_list.append(df.at['weighted avg', 'precision'])
    precision_macro_list.append(df.at['macro avg', 'precision'])
    recall_list.append(df.at['weighted avg', 'recall'])
    recall_macro_list.append(df.at['macro avg', 'recall'])
    f1_list.append(df.at['weighted avg', 'f1-score'])
    f1_macro_list.append(df.at['macro avg', 'f1-score'])

    print(report)

performaceDF = pd.DataFrame({'datasetName': dataSetNames,
                                 'accuracy': accuracy_list,
                                 'precision': precision_list,
                                 'recall': recall_list,
                                 'f1': f1_list,
                                 'precision_macro': precision_macro_list,
                                 'recall_macro': recall_macro_list,
                                 'f1_macro': f1_macro_list})

performaceFileName = "gae+lr_result_3.csv"
performaceDF.to_csv(performaceFileName, index=True)


