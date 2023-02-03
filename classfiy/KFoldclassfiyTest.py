# 用交叉验证的方式 测试 分类效果。
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

# 读取带标签的数据
#total_pd = pd.read_csv("../getEmbeddings/cora512.csv")
# total_pd = pd.read_csv("../gae/getGAEembedding/gae_cora256.csv")
total_pd = pd.read_csv("../XGBresult/cora_epoch_265_accuracy_0.8745387453874539.csv")

# 按100%的比例抽样即达到打乱数据的效果
total_pd=total_pd.sample(frac=1.0)


# 训练数据Y
Y = total_pd[total_pd.columns[-1]]
# 训练数据X
X = total_pd.iloc[:, 1:-1]
# 标准化处理
x_scaled = preprocessing.scale(X)

#设置kfold，交叉采样法拆分数据集
kfold=StratifiedKFold(n_splits=5)

#汇总不同模型算法
classifiers=[]
# classifiers.append(SVC())
classifiers.append(DecisionTreeClassifier())
classifiers.append(RandomForestClassifier())
classifiers.append(ExtraTreesClassifier())
classifiers.append(GradientBoostingClassifier())
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression())
classifiers.append(LinearDiscriminantAnalysis())
classifiers.append(MLPClassifier())
classifiers.append(XGBClassifier())
classifiers.append(LGBMClassifier())


#不同机器学习交叉验证结果汇总 scoring='accuracy'  f1_micro  precision_micro  recall_micro 可以，
cv_results=[]
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, x_scaled, Y,
                                      scoring='accuracy', cv=kfold, n_jobs=-1))

result_pd = pd.DataFrame(cv_results)

# 求出模型得分的均值和标准差
# 求出模型得分的均值和标准差
cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

print("模型均值为：")
print(cv_means)

print("模型标准差为：")
print(cv_std)

# 汇总数据
cvResDf = pd.DataFrame({'cv_mean': cv_means,
                        'cv_std': cv_std,
                        'algorithm': ['DecisionTreeCla', 'RandomForestCla', 'ExtraTreesCla',
                                      'GradientBoostingCla', 'KNN', 'LR', 'LinearDiscrimiAna', 'MLPClassifier','XGBClassifier','LGBMClassifier']})
cvResDf.to_csv("./cora_cora_epoch_265_accuracy_0.8745387453874539_result.csv")

# 可视化查看不同算法的表现情况
sns.barplot(data=cvResDf,x='cv_mean',y='algorithm',**{'xerr':cv_std})
plt.show()
