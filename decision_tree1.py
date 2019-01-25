from sklearn.tree import DecisionTreeClassifier


#ID3 分类树采用 Entropy, CART 采用GINI
clf =DecisionTreeClassifier(criterion='entropy')
#
# DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
#             max_features=None, max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, presort=False, random_state=None,
#             splitter='best')
#数据探索
import pandas as pd
# 数据加载
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# 数据探索
# print(train_data.info())
# print('-'*30)
# print(train_data.describe())
# print('-'*30)
# print(train_data.describe(include=['O']))
# print('-'*30)
# print(train_data.head())
# print('-'*30)
# print(train_data.tail())

#result
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 891 entries, 0 to 890
# Data columns (total 12 columns):
# PassengerId    891 non-null int64
# Survived       891 non-null int64
# Pclass         891 non-null int64
# Name           891 non-null object
# Sex            891 non-null object
# Age            714 non-null float64
# SibSp          891 non-null int64
# Parch          891 non-null int64
# Ticket         891 non-null object
# Fare           891 non-null float64
# Cabin          204 non-null object
# Embarked       889 non-null object
# dtypes: float64(2), int64(5), object(5)
# memory usage: 83.6+ KB
# None
# ------------------------------
#        PassengerId    Survived     ...           Parch        Fare
# count   891.000000  891.000000     ...      891.000000  891.000000
# mean    446.000000    0.383838     ...        0.381594   32.204208
# std     257.353842    0.486592     ...        0.806057   49.693429
# min       1.000000    0.000000     ...        0.000000    0.000000
# 25%     223.500000    0.000000     ...        0.000000    7.910400
# 50%     446.000000    0.000000     ...        0.000000   14.454200
# 75%     668.500000    1.000000     ...        0.000000   31.000000
# max     891.000000    1.000000     ...        6.000000  512.329200
#
# [8 rows x 7 columns]
# ------------------------------
#                                                 Name   ...    Embarked
# count                                            891   ...         889
# unique                                           891   ...           3
# top     Jerwan, Mrs. Amin S (Marie Marthe Thuillard)   ...           S
# freq                                               1   ...         644
#
# [4 rows x 5 columns]
# ------------------------------
#    PassengerId  Survived  Pclass    ...        Fare Cabin  Embarked
# 0            1         0       3    ...      7.2500   NaN         S
# 1            2         1       1    ...     71.2833   C85         C
# 2            3         1       3    ...      7.9250   NaN         S
# 3            4         1       1    ...     53.1000  C123         S
# 4            5         0       3    ...      8.0500   NaN         S
#
# [5 rows x 12 columns]
# ------------------------------
#      PassengerId  Survived  Pclass    ...      Fare Cabin  Embarked
# 886          887         0       2    ...     13.00   NaN         S
# 887          888         1       1    ...     30.00   B42         S
# 888          889         0       3    ...     23.45   NaN         S
# 889          890         1       1    ...     30.00  C148         C
# 890          891         0       3    ...      7.75   NaN         Q
#
# [5 rows x 12 columns]

#数据清洗
# 使用平均年龄来填充年龄中的 nan 值
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
# 使用票价的均值填充票价中的 nan 值
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)
# 使用登录最多的港口来填充登录港口的 nan 值
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S',inplace=True)

# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]

#特征数值化
from sklearn.feature_extraction import DictVectorizer
dvec=DictVectorizer(sparse=False)
train_features=dvec.fit_transform(train_features.to_dict(orient='record'))


#决策树模型
from sklearn.tree import DecisionTreeClassifier
# 构造 ID3 决策树
clf = DecisionTreeClassifier(criterion='entropy')
# 决策树训练
clf.fit(train_features, train_labels)

test_features=dvec.transform(test_features.to_dict(orient='record'))
# 决策树预测
pred_labels = clf.predict(test_features)

#决策树评估1
# 得到决策树准确率
acc_decision_tree = round(clf.score(train_features, train_labels), 6)
print(u'score 准确率为 %.4lf' % acc_decision_tree)

# 决策树评估2 交叉验证
import numpy as np
from sklearn.model_selection import cross_val_score
# 使用 K 折交叉验证 统计决策树准确率
print(u'cross_val_score 准确率为 %.4lf' % np.mean(cross_val_score(clf, train_features, train_labels, cv=10)))




def test():
    print(clf)
    print(train_data['Embarked'].value_counts())
    print(dvec.feature_names_)
test()