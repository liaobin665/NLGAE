# 得到原始数据的X
import graphgallery
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score,recall_score, f1_score,roc_auc_score,classification_report
from graphgallery.datasets import Planetoid
from graphgallery.datasets import KarateClub
from graphgallery.datasets import NPZDataset
from sklearn.preprocessing import label_binarize
import numpy as np
import datetime
import time
import pandas as pd

from graphgallery.gallery.gallery_model.tensorflow import APPNP
from graphgallery.gallery.gallery_model.tensorflow import PPNP
from graphgallery.gallery.gallery_model.tensorflow import ChebyNet
from graphgallery.gallery.gallery_model.tensorflow import DAGNN
from graphgallery.gallery.gallery_model.tensorflow import DenseGCN
from graphgallery.gallery.gallery_model.tensorflow import FastGCN
from graphgallery.gallery.gallery_model.tensorflow import GAT
from graphgallery.gallery.gallery_model.tensorflow import GCN
from graphgallery.gallery.gallery_model.tensorflow import GCNA
from graphgallery.gallery.gallery_model.tensorflow import GraphSAGE
from graphgallery.gallery.gallery_model.tensorflow import GWNN
from graphgallery.gallery.gallery_model.tensorflow import RobustGCN
from graphgallery.gallery.gallery_model.tensorflow import TAGCN
from graphgallery.gallery.gallery_model.tensorflow import SBVAT
from graphgallery.gallery.gallery_model.tensorflow import SGC
from graphgallery.gallery.gallery_model.tensorflow import OBVAT
from graphgallery.gallery.gallery_model.tensorflow import MLP

graphgallery.set_backend("tensorflow")

# datasets = ['cora', 'citeseer', 'pubmed']
# datasets = ['twitch']
datasets = ['amazon_cs',  'dblp', 'uai']
for dataName in datasets:
    # 得到数据
    # 得到数据
    # data = Planetoid(dataName, root="../../GraphData/datasets/", verbose=False)

    # data = KarateClub(dataName, root="../../GraphData/datasets/", verbose=False)

    data = NPZDataset(dataName, root="../../GraphData/datasets/", verbose=False)

    print()

    dataX =pd.DataFrame(data.g.node_attr)
    csvname= dataName+'_X.csv'
    dataX.to_csv(csvname)