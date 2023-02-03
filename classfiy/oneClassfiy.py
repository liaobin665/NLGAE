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
total_pd = pd.read_csv("../gae/getGAEembedding/gae_citeseer256.csv")

# 按100%的比例抽样即达到打乱数据的效果
total_pd=total_pd.sample(frac=1.0, random_state=99)


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

print(report)



