# -*- coding: utf-8 -*-
# 乳腺癌诊断分类
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

#加载数据
data = pd.read_csv('./data.csv')
#数据集中列数比较多,把列全部显示
pd.set_option('display.max_columns',None)
print(data.columns)
# print(data.head(5))
print(data.describe())

#将特征字段分成3组

features_mean = list(data.columns[2:12])
features_se = list(data.columns[12:22])
features_worst = list(data.columns[22:32])

#数据清洗
#不考虑ID列
data.drop('id',axis=1,inplace=True)
# 将结果数值化
data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})

# vis for the diagnosis result
# sns.countplot(data['diagnosis'],label= 'Count')
# plt.show()
# # 用热力图呈现feature_mean字段的相关性
# corr = data[features_mean].corr()
# plt.figure(figsize=(14,14))
# #annot = Ture 显示每个方格的数据
# sns.heatmap(corr,annot=True)
# plt.show()

#特征选择
features_remain = data.columns[1:31]
print(features_remain)
print('_'*100)
#划分数据集为 测试集、训练集
train,test = train_test_split(data,test_size=0.3)
train_x = train[features_remain]
train_y = train['diagnosis']
test_x = train[features_remain]
test_y = train['diagnosis']

#采用Z-score规范化数据
ss = StandardScaler()
train_x = ss.fit_transform(train_x)
test_x = ss.fit_transform(test_x)

#创建SVM 分类器
model =svm.LinearSVC()
#用数据集做训练
model.fit(train_x,train_y)
#用训练集做预测
prediction = model.predict(test_x)
print('准确率：',metrics.accuracy_score(prediction,test_y))
#准确率： 0.9899497487437185