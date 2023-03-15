# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:04:34 2023

@author: DELL
"""
## 本文件目标是基于传统的打分方法，筛选需要的高频词赋分，并通过特性挑选(RFE函数），knn,logistic regression等方法予以辅助。
import pandas as pd
import jieba as jb   
import nltk

## 1. 分词及语料库清理 
# 将下载完成的停用词载入
sw = pd.read_csv('E:/internship/Data label/stopwords-master/baidu_stopwords.txt', names=['stopword'], index_col=False)
stop_list = sw['stopword'].tolist()
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',4)

# 导入数据，保留文字部分
data1 = pd.read_csv('E:/internship/Data label/分表/榜单.csv',encoding='utf-8')
data2 = pd.read_csv('E:/internship/Data label/分表/区块链.csv',encoding='utf-8')
data3 = pd.read_csv('E:/internship/Data label/分表/sumdata.csv',encoding='utf-8')
df1 = data1[['企业名称', '企业简介']]
df1 = pd.DataFrame(df1)

df2 = data2[['企业名称', '企业简介']]
df2 = pd.DataFrame(df2)

df3 = data3[['企业名称', '企业简介']]
df3 = pd.DataFrame(df3)
# 将各种括号内内容、数字、英文及其他特殊字符（如中文标点）删除
def cut_text(txt_in):
   import re
   clean = re.sub(u"\\（.*?）|\\{.*?}|\\[.*?]|\\【.*?】|[\d]|[a-zA-Z]|[\W]", '', txt_in)
   cutword = jb.cut(clean)
   return cutword
# 集成分词与删除停用词
def cutrem(txt_in):
    cutrem = txt_in.apply(lambda x : [i for i in cut_text(x) if i not in stop_list])
    return cutrem

# 输入指定列，输出分词和清理后的结果
df1['namecut'] = cutrem(df1['企业名称'])
df1['introcut'] = cutrem(df1['企业简介'])
df2['namecut'] = cutrem(df2['企业名称'])
df2['introcut'] = cutrem(df2['企业简介'])
df3['namecut'] = cutrem(df3['企业名称'])
df3['introcut'] = cutrem(df3['企业简介'])
# 分词后删除停用词
#df1['introcut'] = df1['企业简介'].apply(lambda x : [i for i in cut_text(x) if i not in stop_list])
#df1['scopecut'] = df1['经营范围'].apply(lambda x : [i for i in cut_text(x) if i not in stop_list])

#df2['introcut'] = df2['企业简介'].apply(lambda x : [i for i in cut_text(x) if i not in stop_list])
#df2['scopecut'] = df2['经营范围'].apply(lambda x : [i for i in cut_text(x) if i not in stop_list])

## 2. 词频统计（用于解决大类企业的筛选）
#输入特定分词列，输出词频
def freq(txt_in):
# 将所有分词归入一张表
    df = []
    for content in txt_in:
     df.extend(content)
     
# 如果仍有空格，用此句补充删除空格
#introcut1 = [x.strip() for x in introcut1 if x.strip() != '']

#词频生成   
    freq = nltk.FreqDist(df).most_common()
    pd_freq = pd.DataFrame(freq, columns = ["Token","raw_frequency"])
    return pd_freq

megauniname =  freq(df1['namecut'])
megauniintro = freq(df1['introcut'])

blockchainname =  freq(df2['namecut'])
blockchainintro = freq(df2['introcut'])

sumdataname =  freq(df3['namecut'])
sumdataintro = freq(df3['introcut'])

## 3. 文字向量化及tfidf计算(用于解决相同大类内不同子类企业的筛选)
from sklearn.feature_extraction.text import TfidfVectorizer
the_path = 'E:/internship/Data label/分表/Result/'

# 输入第二部分分词完成的列，取消list嵌套list,转为字符串
def strcvt(txt_in):
    strresult = sum(txt_in, [])
    strresult = ' '.join(strresult)
    return strresult

# 将两个用于比较的列合成可供向量化的dataframe
def cp(a, b):
    test = strcvt(a)
    test2 = strcvt(b)
    testpd = pd.DataFrame()
    testpd['word'] = [test, test2]
    return testpd

# 计算模块，可以调试最大词组数量，以及max_df比例

# 获取各dataframe名字，在tfidf功能中将分词数量转为字符串保存，方便输出后区别查看
def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name
# tfidf计算功能
def my_tf_idf_fun(df_in, m, n, q):
    import pandas as pd
    vectorizer = TfidfVectorizer(ngram_range=(m, n), max_df = q)
    #name = get_df_name(df_in) + str(m) + '-' + str(n) + 'max_df' + str(q)
    inputcorpus = df_in['word']
    my_tf_idf_t = pd.DataFrame(vectorizer.fit_transform(inputcorpus).toarray())
    my_tf_idf_t.columns = vectorizer.vocabulary_
    my_tf_idf_t = my_tf_idf_t.transpose()
    my_tf_idf_t.columns = ['比较类', '参照类']
    my_tf_idf_t['比较类>参照类'] = my_tf_idf_t['比较类'] - my_tf_idf_t['参照类']
    my_tf_idf_t = my_tf_idf_t.sort_values(by = "比较类>参照类", ascending  = False)
    #my_tf_idf_t.to_csv(the_path + name + 'tfidf.csv', encoding='utf_8_sig')
    return my_tf_idf_t

# 向量化计算功能
def my_vec_fun(df_in, m, n, q):
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd
    vectorizer = CountVectorizer(ngram_range=(m, n), max_df = q)
    name = get_df_name(df_in) + str(m) + '-' + str(n) + 'max_df' + str(q)
    inputcorpus = df_in['word']
    my_vec_t = pd.DataFrame(vectorizer.fit_transform(inputcorpus).toarray())
    my_vec_t.columns = vectorizer.vocabulary_
    my_vec_t = my_vec_t.transpose()
    my_vec_t.columns = ['比较类', '参照类']
    my_vec_t['比较类>参照类'] = my_vec_t['比较类'] - my_vec_t['参照类']
    my_vec_t = my_vec_t.sort_values(by = "比较类>参照类", ascending  = False)
    my_vec_t.to_csv(the_path + name + 'vec.csv', encoding='utf_8_sig')
    return my_vec_t

#namecorpus = cp(df1['namecut'], df2['namecut'])
#introcorpus = cp(df1['introcut'], df2['introcut'])

#sumnamecorpus = cp(df1['namecut'], df3['namecut'])
#sumintrocorpus = cp(df1['introcut'], df3['introcut'])

#testtfidfintro = my_tf_idf_fun(introcorpus['word'], 1, 2, 2)
#testvecintro = my_vec_fun(introcorpus['word'], 1, 2, 2)
#testtfidfintro = my_tf_idf_fun(introcorpus['word'], 1, 1, 2)
#testvecintro = my_vec_fun(introcorpus['word'], 1, 1, 2)

## 4. 将全体企业库高频词前n位在子类企业高频词库中删除，并保存为csv
def filternword(df_in, df_sum, n):
    #name = get_df_name(df_in) + str(n) 
    secstop_list = df_sum['Token'].head(n).tolist()
    filterword = df_in[~df_in["Token"].isin(secstop_list)]
    #filterword.to_csv(the_path  + name + 'filter.csv', encoding='utf_8_sig')
    return filterword

megauniintrofilter30 = filternword(megauniintro, sumdataintro, 30)
megauniintrofilter30 = filternword(megauniintro, sumdataintro, 20)
megauniintrofilter30 = filternword(megauniintro, sumdataintro, 10)

## 5. 将若干高频词逐一筛选，按该企业是否含有该词赋值，然后打分
#该文本可供试验功能可用性
#test = ['中国数字研发平台',
#        '全球云研发科技',
#       '提供发展创新平台',
#        '实验基地']
#testpd = []
#testpd = pd.DataFrame(testpd)
#testpd['test'] = test
#testpd['cut'] = cutrem(testpd['test'])

def topword(df_in, source, n):
    top = source[source['raw_frequency'] > n]
    toplist = top['Token'].tolist()
    df = []
    df = pd.DataFrame(df)
    for word in toplist:
        name = str(word)
        #df[name] = testpd['cut'].apply(lambda x : 1 if name in x else 0)
        df[name] = df_in['introcut'].apply(lambda x : 1 if name in x else 0)
    return df
megauniintrodata = topword(df1, megauniintrofilter30, 100)   

def score(data, n, p, q):
    sumscore = p * n + q * (len(data.iloc[1, :]) - n) 
    data['score'] = data.apply(lambda x: x.sum(), axis=1)
    topn = data.iloc[:, 0 : n]
    data['topnscore'] = topn.apply(lambda x: x.sum(), axis=1) 
    data['adjustscore'] = (data['topnscore'] * p + (data['score'] - data['topnscore']) * q)/sumscore * 100
      
#dftest = pd.concat([df3, megauniintrodata.iloc[:, 17:20]], axis=1) 
#dftest.to_csv(the_path + '打分实验.csv', encoding='utf_8_sig')

## 6. 生成训练集
data4 = pd.read_csv('E:/internship/Data label/分表/负标签企业.csv', encoding='utf-8')
df4 = data4[['企业名称', '企业简介']]
df4 = pd.DataFrame(df4)
df4['namecut'] = cutrem(df4['企业名称'])
df4['introcut'] = cutrem(df4['企业简介'])
nmdataname =  freq(df4['namecut'])
nmdataintro = freq(df4['introcut'])

positive = filternword(megauniintro, nmdataintro, 30)
#negative = filternword(nmdataintro, megauniintro, 30)
megauniintrodata = topword(df1, positive, 100) 
score(megauniintrodata, 10, 10, 5) 
normalintrodata = topword(df4, positive, 100) 
score(normalintrodata, 10, 10, 5) 
df1test = pd.concat([df1.iloc[:, 0], megauniintrodata], axis=1) 
df4test = pd.concat([df4.iloc[:, 0], normalintrodata], axis=1) 

df4test['元宇宙企业'] = 0
df1test['元宇宙企业'] = 1
inidata = pd.concat([df1test, df4test])

## 7. knn方法
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
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
y = inidata['元宇宙企业']
X = inidata.iloc[:, 1:18]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 23)

#补充:  RFE to find several features that help model predict the best:

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

estimator = LinearRegression().fit(X_train, y_train) #model with all X variables


selector = RFE(estimator, n_features_to_select=9, step=1) # step tells RFE how many features to remove each time model features are evaluated

selector = selector.fit(X_train, y_train) # fit RFE estimator.

print("Num Features: "+str(selector.n_features_))
print("Selected Features: "+str(selector.support_)) # T/F for top five features
print("Feature Ranking: "+str(selector.ranking_))  # ranking for top five + features

# Transform X data for other use in this model or other models:

X_train = selector.transform(X_train) #reduces X to subset identified above
X_test = selector.transform(X_test)
print(X.columns[selector.support_ ]) # The most important features

# scale the data for KNN
#from sklearn import preprocessing

#scaler = preprocessing.StandardScaler().fit(X_train)
#X_train_scaled = scaler.transform(X_train)
#X_test_scaled = scaler.transform(X_test)

# use gridsearch to find the best parameter for n_neightbors
knn_pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
knn_param_grid = {'kneighborsclassifier__n_neighbors': range(1, 10)}
knn_grid = GridSearchCV(knn_pipe, knn_param_grid).fit(X_train, y_train)

print("KNN for regression (scaled data)")
print("Test set Score: {:.2f}".format(knn_grid.score(X_test, y_test)))
print("Best Parameter: {}".format(knn_grid.best_params_))

#The best parameter, in this case the k in KNN classifier, is chosen through GridsearchCV to test on a range of 1 to 10. 
#The result shows that 7 is the optimal parameter used to build the KNN model.

knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train)

print("KNN classifier (scaled data)")
print("Training set score: {:.2f}".format(knn.score(X_train, y_train)))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

# Kfold Cross Validation
print("Mean Cross Validation, KFold: {:.4f}".format(np.mean(cross_val_score(knn, X_train, y_train, cv=10))))


## 从结果来看，scaled data的结果更稳定
sumintrodata = topword(df3, positive, 100) 
score(sumintrodata, 10, 10, 5) 
df3test = pd.concat([df3.iloc[:, 0], sumintrodata], axis=1) 

## 预测符合条件的企业数
X_predict = df3test.iloc[:, 1:18]
X_predict = selector.transform(X_predict)
y_knnpredict = knn.predict(X_predict)
df3test['knn元宇宙企业'] = y_knnpredict
output = pd.concat([df3.iloc[:, 0:2], df3test.iloc[:, 20:22]], axis=1) 

## 8. Logistic Regression方法
from sklearn.model_selection import train_test_split
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

logreg = LogisticRegression(C = 1).fit(X_train, y_train)

print("Logistic regression")
print("Training set score: {:.2f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.2f}".format(logreg.score(X_test, y_test)))
# Kfold Cross Validation
print("Mean Cross Validation, KFold: {:.2f}".format(np.mean(cross_val_score(logreg, X_train, y_train, cv=10))))

## 预测情况
y_logpredict = logreg.predict(X_predict)
output['LR元宇宙企业'] = y_logpredict

## 9. SVM方法
# Support Vector Machine
from sklearn.svm import SVC

svm_param_grid = {'C': [1, 5, 10, 50],
              'gamma': [0.0001, 0.0005, 0.001, 0.005]}

svm_grid = GridSearchCV(SVC(kernel='rbf'), svm_param_grid).fit(X_train, y_train)

print(svm_grid.best_params_)
print(svm_grid.best_estimator_)
print(svm_grid.best_score_)

svm = SVC(kernel='rbf', C = 10, gamma = 0.005).fit(X_train, y_train)
print("SVM")
print("Training set score: {:.2f}".format(svm.score(X_train, y_train)))
print("Test set score: {:.2f}".format(svm.score(X_test, y_test)))

# Kfold Cross Validation
print("Mean Cross Validation, KFold: {:.4f}".format(np.mean(cross_val_score(svm, X_train, y_train, cv=10))))

## 预测符合条件的企业数
y_svmpredict = svm.predict(X_predict)
output['svm元宇宙企业'] = y_svmpredict

## 10. 决策树方法
# Bagged tree
from sklearn.ensemble import BaggingClassifier

bagtree_param_grid = {'n_estimators': list(range(1, 50))}

bagtree_grid = GridSearchCV(BaggingClassifier(), bagtree_param_grid).fit(X_train, y_train)

print("Bagging trees (not scaled data)")
print("Test set Score: {:.2f}".format(bagtree_grid.score(X_test, y_test)))
print("Best Parameter: {}".format(bagtree_grid.best_params_))

bagtree = BaggingClassifier(n_estimators = 36).fit(X_train, y_train)

print("Bagging trees (not scaled data)")
print("Training set score: {:.2f}".format(bagtree.score(X_train, y_train)))
print("Test set score: {:.2f}".format(bagtree.score(X_test, y_test)))

# Kfold Cross Validation
print("Mean Cross Validation, KFold: {:.4f}".format(np.mean(cross_val_score(bagtree, X_train, y_train, cv=10))))

## 预测符合条件的企业数
y_btpredict = bagtree.predict(X_predict)
output['bagtree元宇宙企业'] = y_btpredict

## 11. 总结
output['总分'] = output['LR元宇宙企业'] + output['knn元宇宙企业'] + output['svm元宇宙企业'] + output['bagtree元宇宙企业']
output.to_csv(the_path + "高频词打分法.csv", encoding ='utf_8_sig')