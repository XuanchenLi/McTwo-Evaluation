import pandas as pd
import os
import numpy as np
from minepy import MINE
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn import tree
from sklearn import naive_bayes
from sklearn.neighbors import KNeighborsClassifier



DATA_PATH = './data_set/'


# 计算最大互信息系数
def mic(x, y):
    mine = MINE(0.6, 15)
    mine.compute_score(x, y)
    return mine.mic()


# 使用McOne算法筛选第一批特征
def McOne(F, C, r):
    # 初始化
    rowNum, colNum = F.shape
    numSubset = 0
    subset = [0]*colNum  # 初始化为0
    micFC = [0]*colNum
    # 对每个feature逐个计算MIC
    for idx in range(colNum):
        micFC[idx] = mic(F[:, idx], C)
        if micFC[idx] >= r:
            subset[numSubset] = idx
            numSubset = numSubset + 1
    # 递减顺序排序
    subset = subset[0:numSubset]  # 去除冗余项
    subset.sort(key=lambda x: micFC[x], reverse=True)
    select = [True] * numSubset  # 蒙版数组，花式索引
    for e in range(numSubset):
        if select[e]:
            for q in range(e+1, numSubset):
                if select[q] and mic(F[:, subset[e]], F[:, subset[q]]) >= micFC[subset[q]]:
                    select[q] = False

    subset = np.array(subset).astype('int')
    select = np.array(select).astype('bool')

    return F[:, np.array(subset)[select]]


# 计算BACC
# BACC = (Sn+Sp)/2
# Sn = TP/P
# Sp = TN/N
def bacc(F, C):
    rowNum, colNum = F.shape
    NN = KNeighborsClassifier(n_neighbors=1)
    prediction = []
    C = C.astype('int')
    # LOO计算NN
    for i in range(rowNum):
        NN.fit(F[[x for x in range(rowNum) if x != i]],
               C[[x for x in range(rowNum) if x != i]])
        prediction.append(NN.predict(F[[i]]).tolist()[0])
    prediction = np.array(prediction)
    return (np.mean(prediction[np.where(C == 0)] == C[np.where(C == 0)]) +
            np.mean(prediction[np.where(C == 1)] == C[np.where(C == 1)])) / 2


# 在McOne的基础上筛选最终的特征子集
def McTwo(FR, C):
    rowNum, colNum = FR.shape
    left = set([element for element in range(colNum)])
    selection = set([])
    maxBacc = 0.0

    while True:
        curBacc = -1.0
        targetIdx = -1
        for x in left:
            tmpBacc = bacc(FR[:, [x] + list(selection)], C)
            if tmpBacc > curBacc:
                curBacc = tmpBacc
                targetIdx = x
        if curBacc > maxBacc:
            maxBacc = curBacc
            selection.add(targetIdx)
            left.remove(targetIdx)
        else:
            break
    return FR[:, list(selection)]


# 将特征子集应用到各种分类算法中，比较准确率
def evaluation(F, C):
    C = C.astype('int')
    kf = KFold(n_splits=5)
    accSVM = accNN = accDT = accNBayes = 0.0
    for trainIdx, testIdx in kf.split(F):
        trainSet = F[trainIdx]
        testSet = F[testIdx]
        trainLabel = C[trainIdx]
        testLabel = C[testIdx]
        accSVM = max(accSVM, np.mean(SVC().fit(trainSet, trainLabel).predict(testSet) == testLabel))
        accDT = max(accDT, np.mean(tree.DecisionTreeClassifier().fit(trainSet, trainLabel).predict(testSet) == testLabel))
        accNN = max(accNN, np.mean(KNeighborsClassifier(n_neighbors=1).fit(trainSet, trainLabel).predict(testSet) == testLabel))
        accNBayes = max(accNBayes, np.mean(naive_bayes.GaussianNB().fit(trainSet, trainLabel).predict(testSet) == testLabel))
    accSet = {'SVM':accSVM, 'Nbayes':accNBayes, 'DTree':accDT, 'NN':accNN}
    return accSet



for dataName in os.listdir(DATA_PATH):
    dataArray = pd.read_table(os.path.join(DATA_PATH, dataName), header=None, low_memory=False, index_col=0).transpose().to_numpy()
    features = dataArray[:, 1:]  # 数据集的特征子集
    label = dataArray[:, 0]  # 数据集的标签子集
    for numLabel, originLabel in enumerate(list(set(label))):
        label[np.where(label == originLabel)] = numLabel  # 数字化标签
    oneRes = McOne(features, label, 0.3)  # Step1:McOne算法
    print(features.shape, oneRes.shape)
    twoRes = McTwo(oneRes, label)  # Step2:McTwo算法
    print(features.shape, twoRes.shape)
    print(evaluation(oneRes, label), evaluation(twoRes, label))


