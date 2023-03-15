# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:12:14 2023

@author: Administer
"""
## 本文件目标是将所有语料进行tfidf转换后，进行机器学习，标注给定的种类。
import pandas as pd
import jieba as jb   
import re
import numpy as np

## 1. 分词及语料库清理 
# 将下载完成的停用词载入
sw = pd.read_csv('E:/internship/Data label/stopwords-master/baidu_stopwords.txt', names=['stopword'], index_col=False)
stop_list = sw['stopword'].tolist()
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',4)
the_path = 'E:/internship/Data label/分表/Result/'
# 导入数据，保留文字部分
train1 = pd.read_csv('E:/internship/Data label/分表/榜单.csv',encoding='utf-8')
train2 = pd.read_csv('E:/internship/Data label/分表/负标签企业.csv', encoding='utf-8')

tt1 = train1[['企业名称', '企业简介']]
tt1 = pd.DataFrame(tt1)

tt2 = train2[['企业名称', '企业简介']]
tt2 = pd.DataFrame(tt2)
tt = pd.concat([tt1, tt2], axis = 0)

def cut_text(txt_in):
   clean = re.sub(u"\\（.*?）|\\{.*?}|\\[.*?]|\\【.*?】|[\d]|[a-zA-Z]|[\W]", '', txt_in)
   cutword = jb.cut(clean)
   return cutword
# 集成分词与删除停用词
def strcvt(txt_in):
    strresult = ' '.join(txt_in)
    return strresult
def cutrem(txt_in):
    cutrem = txt_in.apply(lambda x : [i for i in cut_text(x) if i not in stop_list])
    return cutrem

# 转为dataframe才能进行tfidf计算
tt = cutrem(tt['企业简介'])
tt.to_csv('Data_cut.csv',header=1,index=0)

tt=pd.read_csv('Data_cut.csv',header=0)
## 2. 文字向量化及tfidf计算(用于解决相同大类内不同子类企业的筛选更好)
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer

vect = TfidfVectorizer(ngram_range = (1, 2), min_df=10).fit(tt['企业简介'])
X = vect.transform(tt['企业简介'])
trainshape = X.toarray()
trainshape.shape
tt2['元宇宙企业'] = 0
tt1['元宇宙企业'] = 1

#检验是否成功转换
feature_names = vect.get_feature_names_out()
print("Number of features: {}".format(len(feature_names)))
print("First 20 features:\n{}".format(feature_names[:20]))
print("Features 210 to 230:\n{}".format(feature_names[210:230]))
print("Every 200th feature:\n{}".format(feature_names[::200]))


y_1 = np.array(tt1['元宇宙企业'])

y_2 = np.array(tt2['元宇宙企业'])

y= np.append(y_1, y_2)

print(X.shape)
print(y.shape)

## 3. tfidf结果
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

#非惩罚 逻辑回归
scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
print("Mean cross-validation accuracy: {:.2f}".format(np.mean(scores)))

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)

print("Test score: {:.2f}".format(grid.score(X_test, y_test)))

logreg = LogisticRegression(C = 10).fit(X_train, y_train)

print("Logistic regression")
print("Training set score: {:.2f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.2f}".format(logreg.score(X_test, y_test)))
# Kfold Cross Validation
print("Mean Cross Validation, KFold: {:.2f}".format(np.mean(cross_val_score(logreg, X_train, y_train, cv=10))))

## 4. 预测情况
data3 = pd.read_csv('E:/internship/Data label/分表/sumdata.csv',encoding='utf-8')
df3 = data3[['企业名称', '企业简介']]
df3 = pd.DataFrame(df3)
df3text = cutrem(df3['企业简介'])
df3text.to_csv('Data_cut.csv',header=1,index=0)
df3text=pd.read_csv('Data_cut.csv',header=0)

X_predict = vect.transform(df3text['企业简介'])
predictshape = X_predict.toarray()
predictshape.shape
y_predict = logreg.predict(X_predict)
df3['LR元宇宙企业'] = y_predict
df3lrlist = df3[df3['LR元宇宙企业'] == 1]
#df3list.to_csv(the_path + "Logregressionresult.csv", encoding ='utf_8_sig')




## 5. knn方法
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.model_selection import train_test_split
# import the cross validation packages
#from sklearn.model_selection import RepeatedKFold
#from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# import gridsearch and knn packages
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
#from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
#from sklearn import preprocessing

# use gridsearch to find the best parameter for n_neightbors
knn_pipe = make_pipeline(KNeighborsClassifier())
knn_param_grid = {'kneighborsclassifier__n_neighbors': range(1, 10)}
knn_grid = GridSearchCV(knn_pipe, knn_param_grid).fit(X_train, y_train)

print("KNN for regression")
print("Test set Score: {:.2f}".format(knn_grid.score(X_test, y_test)))
print("Best Parameter: {}".format(knn_grid.best_params_))

#The best parameter, in this case the k in KNN classifier, is chosen through GridsearchCV to test on a range of 1 to 10. 

knn = KNeighborsClassifier(n_neighbors = 1).fit(X_train, y_train)

print("KNN classifier (scaled data)")
print("Training set score: {:.2f}".format(knn.score(X_train, y_train)))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

# Kfold Cross Validation
print("Mean Cross Validation, KFold: {:.4f}".format(np.mean(cross_val_score(knn, X_train, y_train, cv=10))))


## 预测符合条件的企业数

y_knnpredict = knn.predict(X_predict)
df3['knn元宇宙企业'] = y_knnpredict
df3knnlist = df3[df3['knn元宇宙企业'] == 1]
#df3knnlist.to_csv(the_path + "knnresult.csv", encoding ='utf_8_sig')

## 6. SVM方法
# Support Vector Machine
from sklearn.svm import SVC

svm_param_grid = {'C': [1, 5, 10, 50],
              'gamma': [0.0001, 0.0005, 0.001, 0.005]}

svm_grid = GridSearchCV(SVC(kernel='rbf'), svm_param_grid).fit(X_train, y_train)

print(svm_grid.best_params_)
print(svm_grid.best_estimator_)
print(svm_grid.best_score_)

svm = SVC(kernel='rbf', C = 50, gamma = 0.005).fit(X_train, y_train)
print("SVM")
print("Training set score: {:.2f}".format(svm.score(X_train, y_train)))
print("Test set score: {:.2f}".format(svm.score(X_test, y_test)))

# Kfold Cross Validation
print("Mean Cross Validation, KFold: {:.4f}".format(np.mean(cross_val_score(svm, X_train, y_train, cv=10))))

## 预测符合条件的企业数
y_svmpredict = svm.predict(X_predict)
df3['svm元宇宙企业'] = y_svmpredict
df3svmlist = df3[df3['svm元宇宙企业'] == 1]
#df3svmlist.to_csv(the_path + "svmresult.csv", encoding ='utf_8_sig')


## 7. 决策树方法
# Bagged tree
from sklearn.ensemble import BaggingClassifier

bagtree_param_grid = {'n_estimators': list(range(1, 50))}

bagtree_grid = GridSearchCV(BaggingClassifier(), bagtree_param_grid).fit(X_train, y_train)

print("Bagging trees (not scaled data)")
print("Test set Score: {:.2f}".format(bagtree_grid.score(X_test, y_test)))
print("Best Parameter: {}".format(bagtree_grid.best_params_))

bagtree = BaggingClassifier(n_estimators = 21).fit(X_train, y_train)

print("Bagging trees (not scaled data)")
print("Training set score: {:.2f}".format(bagtree.score(X_train, y_train)))
print("Test set score: {:.2f}".format(bagtree.score(X_test, y_test)))

# Kfold Cross Validation
print("Mean Cross Validation, KFold: {:.4f}".format(np.mean(cross_val_score(bagtree, X_train, y_train, cv=10))))

## 预测符合条件的企业数
y_btpredict = bagtree.predict(X_predict)
df3['bagtree元宇宙企业'] = y_btpredict
df3bagtreelist = df3[df3['bagtree元宇宙企业'] == 1]
#df3bagtreelist.to_csv(the_path + "bagtreeresult.csv", encoding ='utf_8_sig')

## 8. 综合打分
#在本环境中，由于knn方法效果不佳，因此刨除该方法再单独计算一次总分
df3['除knn总分'] = df3['LR元宇宙企业'] + df3['svm元宇宙企业'] + df3['bagtree元宇宙企业']
df3['总分'] = df3['LR元宇宙企业'] + df3['knn元宇宙企业'] + df3['svm元宇宙企业'] + df3['bagtree元宇宙企业']
df3.to_csv(the_path + "四种机器学习方法综合结果.csv", encoding ='utf_8_sig')