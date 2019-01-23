from sklearn.model_selection import  train_test_split
from sklearn.metrics import  accuracy_score,mean_squared_error
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.datasets import load_iris,load_boston
# #准备数据集
# iris = load_iris()
# features = iris.data
# labels = iris.target
#
# #随机抽取33%数据集作为测试集，其余为训练集
# train_features,test_features,train_labels,test_labels = train_test_split(features,labels,test_size=0.33,random_state=0)
# #创建CART分类树
# clf = DecisionTreeClassifier(criterion='gini')
# #拟合 CART分类树
# clf = clf.fit(train_features,train_labels)
#
# #用CART 做预测
# test_predict = clf.predict(test_features)
# #预测与结果比较
# score = accuracy_score(test_labels,test_predict)
#
# print('CART分类树准确率 %.4f' % score)


##cart 回归树
boston  = load_boston()
print(boston.feature_names)
# 获取特征集和房价
features = boston.data
prices = boston.target
train_features,test_features,train_prices,test_prices = train_test_split(features,prices,test_size=0.33)
# 创建CART回归树
dtr = DecisionTreeRegressor()
dtr.fit(train_features,train_prices)
#预测房价
predict_prices = dtr.predict(test_features)
#测试集的结果评价
print('回归树的二乘偏差均值',mean_squared_error(test_prices,predict_prices))
print('回归树的绝对值偏差均值',mean_absolute_error(test_prices,predict_prices))

