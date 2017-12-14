# KNN 学习笔记
参考资料：http://cs231n.github.io/linear-classify/

# 文档说明
cifar-10-batches-py 文件夹保存数据
NearesrNeighbor.py 为近邻算法
kNearesrNeighbor.py 为k-近邻算法

## KNN 算法
1. 计算 每一个测试集 和 每一个训练集中的样本 的距离（L1, L2, L3）
2. 取前k个最小的距离
3. 统计这k个最小距离对应的测试集样本的类别概率
4. 选择概率最大类别的作为该样本的预测类别
5. 评估准确率

## 总结
1. 调整超参数k：使用验证集（validation_data），也就是将train_data的70%-90%作为真正的train_data，其余的作为validation_data。如果本身数据量较小，则可以使用交叉验证集（cross-validation_data）。选择合适的k值，之后固定k值，用test_data进行一次且唯一的一次验证，输出accurancy。
