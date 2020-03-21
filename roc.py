import numpy as np
import warnings
import itertools
import pandas as pd
from mlxtend.evaluate import confusion_matrix
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn import preprocessing,cross_validation,neighbors
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

#feature
data=pd.read_csv('train_data.csv')
X=np.array(data.drop(['class'],1))
#class label
Y=np.array(data['class'])
n_classes=5 

tY=label_binarize(Y,classes=[1,2,3,4,5])
tx_train,tx_test,ty_train,ty_test = cross_validation.train_test_split(X,tY,test_size=0.3)


clf=neighbors.KNeighborsClassifier()
y_score=clf.fit(tx_train,ty_train).predict(tx_test)
print(y_score)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(ty_test[:,i],y_score[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Class ')
    plt.legend(loc="lower right")
plt.show()
