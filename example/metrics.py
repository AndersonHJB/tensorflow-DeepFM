# 导入NumPy库，用于数值计算。
import numpy as np


# 定义gini函数，计算Gini系数。接受两个参数：actual（实际值）和pred（预测值）。
def gini(actual, pred):
    # 断言实际值和预测值的长度相等，确保每个实际值都有对应的预测值。
    assert (len(actual) == len(pred))
    # np.c_是一个索引对象，用于沿第二轴（列）连接数组。
    # 创建一个数组，包含实际值、预测值和原始索引。然后将其转换为浮点类型。
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    # 对数组进行排序。首先按原始索引排序，然后按预测值的降序排序。
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    # 计算所有实际值的总和。
    totalLosses = all[:, 0].sum()
    # 计算累积实际值的总和，并除以总实际值，得到归一化的累积实际值之和。
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    # 从归一化的Gini和中减去一个校正因子（实际值数量加一除以二）。
    giniSum -= (len(actual) + 1) / 2.
    # 返回Gini系数，即归一化Gini和除以实际值的数量。
    return giniSum / len(actual)


# 定义gini_norm函数，计算归一化Gini系数。
def gini_norm(actual, pred):
    # 计算预测值的Gini系数，并除以实际值对自身的Gini系数（理想情况下的最大Gini系数）。
    return gini(actual, pred) / gini(actual, actual)
