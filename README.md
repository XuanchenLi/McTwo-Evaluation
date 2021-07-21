# McTwo-Evaluation
Codes and necessary datasets to reproduce the paper "McTwo: a two-step feature selection algorithm based on maximal information coefficient"

Long live open source.


Outline:
Comparative Analysis
指标：
分类准确度
选择特征数
比较对象：
Wrapper algorithm(Class1):PAM & RRF&CFS
Filter algorithm(Class2):TRank & WRank & ROCRank

FCBF:外部交叉验证

效能评估：分类算法SVM & NBayes & Dtree & NN
衡量指标：Sn Sp Acc Avc


1.McOne vs McTwo 
(1)Gas1和T1D的Acc(30次5-flod交叉验证)
(2)17个数据集的特征选择数和mAcc
2.McTwo vs Class1
(1)三元组（win/tie/loss）四种分类算法最大结果比较17个数据集上的mAcc
(2)EI比较模型复杂度和准确率

3.McTwo vs Class2
(1)挑选和McTwo所取特征数量一样的前p个特征用三元组法比较mAcc
4.外部交叉验证
(1)数据集ALL1 Gas1 Mye
(2)Class1:CFS FCBF McTwo PAM Rfe RRF
(3)Class2:McTwo RfeRank ROCRank TRank WRank
(4)Acc的框图
