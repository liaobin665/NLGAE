# 定制化的训练器，保存性能最优的embedding，并输出
# 随着训练的进行，随时评估分类的质量。注意需要按需在构造函数中，更改数据的Y。

import tensorflow._api.v2.compat.v1 as tf
from graphgallery.datasets import Planetoid
from graphgallery.datasets import KarateClub
tf.disable_v2_behavior()
from sklearn.model_selection import train_test_split
import numpy as np
from NLGAE.nlgae.nlgae import NLGAE_MODEL
import scipy.sparse as sp
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")

def conver_sparse_tf2np(input):
    return [sp.coo_matrix((input[layer][1], (input[layer][0][:, 0], input[layer][0][:, 1])),
                          shape=(input[layer][2][0], input[layer][2][1])) for layer in input]


class Model_Trainer():

    def __init__(self, args):

        self.args = args
        self.build_placeholders()
        gate = NLGAE_MODEL(args.hidden_dims, args.lambda_)
        self.loss, self.H, self.C = gate(self.A, self.X, self.R, self.S)
        self.optimize(self.loss)
        self.build_session()
        # 训练的数据集名称
        self.dataName = 'cora'
        self.accuracyList = [0.0]
        data = Planetoid(self.dataName, root="../../GraphData/datasets/", verbose=False)
        self.Y = data.graph.node_label

    def build_placeholders(self):
        self.A = tf.sparse_placeholder(dtype=tf.float32)
        self.X = tf.placeholder(dtype=tf.float32)
        self.S = tf.placeholder(tf.int64)
        self.R = tf.placeholder(tf.int64)

    def build_session(self, gpu= True):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if gpu == False:
            config.intra_op_parallelism_threads = 0
            config.inter_op_parallelism_threads = 0
        self.session = tf.Session(config=config)
        self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])


    def optimize(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.args.gradient_clipping)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))


    def __call__(self, A, X, S, R):
        for epoch in range(self.args.n_epochs):
            loss = self.run_epoch(epoch, A, X, S, R)


    def run_epoch(self, epoch, A, X, S, R):

        loss, _ = self.session.run([self.loss, self.train_op],
                                         feed_dict={self.A: A,
                                                    self.X: X,
                                                    self.S: S,
                                                    self.R: R})

        if epoch % 5 == 0:
            Y = self.Y
            embeddings, attentions = self.infer(A, X, S, R)
            out = np.hstack((embeddings, Y.reshape(X.shape[0], 1)))
            total_pd = pd.DataFrame(out)

            total_pd = total_pd.sample(frac=1.0, random_state=99)
            # 训练数据Y
            Y = total_pd[total_pd.columns[-1]]
            # 训练数据X
            X = total_pd.iloc[:, 1:-1]

            # 标准化处理
            # x_scaled = preprocessing.scale(X)

            X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2,
                                                                random_state=99)

            # cls = XGBClassifier()
            cls = LogisticRegression()
            cls.fit(X_train, y_train)

            report = classification_report(y_test, cls.predict(X_test), output_dict=True)
            df = pd.DataFrame(report).transpose()
            # accuracy=df.at['accuracy', 'f1-score']
            #
            # #如果超越当前的最大值，则保存结果
            # if accuracy>max(self.accuracyList):
            #     self.accuracyList.append(accuracy)
            #     pd.DataFrame(out).to_csv(self.dataName + '_epoch_' + str(epoch) + '_accuracy_' + str(accuracy) + '.csv')

            print("-------Epoch,loss, accuracy, precision, recall, F1-----------------------")
            # print(epoch,loss,df.at['accuracy', 'f1-score'],df.at['weighted avg', 'precision'],df.at['weighted avg', 'recall'],df.at['weighted avg', 'f1-score'])

            # if epoch>=100:
            #     pd.DataFrame(out).to_csv(self.dataName+'_epoch_'+str(epoch)+'_accuracy_'+str(accuracy)+'.csv')


            # 尝试画图
            # tsne = TSNE()
            # tsneout = tsne.fit_transform(embeddings)
            # fig = plt.figure()
            # for i in range(len(set(Y))):
            #     indices = Y == i
            #     x, y = tsneout[indices].T
            #     plt.scatter(x, y, label=str(i))
            # plt.legend()
            # plt.show()

        print("Epoch: %s, Loss: %.2f" % (epoch, loss))
        return loss

    def infer(self, A, X, S, R):
        H, C = self.session.run([self.H, self.C],
                           feed_dict={self.A: A,
                                      self.X: X,
                                      self.S: S,
                                      self.R: R})
        return H, conver_sparse_tf2np(C)

    def prepare_graph_data(adj):
        num_nodes = adj.shape[0]
        adj = adj + sp.eye(num_nodes)  # self-loop
        # data =  adj.tocoo().data
        adj[adj > 0.0] = 1.0
        if not sp.isspmatrix_coo(adj):
            adj = adj.tocoo()
        adj = adj.astype(np.float32)
        indices = np.vstack((adj.col, adj.row)).transpose()
        return (indices, adj.data, adj.shape), adj.row, adj.col






