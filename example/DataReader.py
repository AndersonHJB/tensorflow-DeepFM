"""
A data parser for Porto Seguro's Safe Driver Prediction competition's dataset.
URL: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction
"""

# 导入pandas库，用于数据操作和分析。
import pandas as pd


# 定义FeatureDictionary类，用于创建特征字典。
class FeatureDictionary(object):
    # 初始化函数，接收训练和测试文件的路径，或已加载的DataFrame，以及数值型和忽略的列。
    def __init__(self, trainfile=None, testfile=None,
                 dfTrain=None, dfTest=None, numeric_cols=[], ignore_cols=[]):
        # 确保至少提供了训练文件或训练DataFrame中的一个。
        assert not ((trainfile is None) and (dfTrain is None)), "trainfile or dfTrain at least one is set"
        # 确保不会同时提供训练文件和训练DataFrame。
        assert not ((trainfile is not None) and (dfTrain is not None)), "only one can be set"
        # 确保至少提供了测试文件或测试DataFrame中的一个。
        assert not ((testfile is None) and (dfTest is None)), "testfile or dfTest at least one is set"
        # 确保不会同时提供测试文件和测试DataFrame。
        assert not ((testfile is not None) and (dfTest is not None)), "only one can be set"
        # 初始化类变量。
        self.trainfile = trainfile
        self.testfile = testfile
        self.dfTrain = dfTrain
        self.dfTest = dfTest
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        # 生成特征字典。
        self.gen_feat_dict()

    # 生成特征字典的方法。
    def gen_feat_dict(self):
        # 如果没有提供训练DataFrame，则从文件中加载。
        if self.dfTrain is None:
            dfTrain = pd.read_csv(self.trainfile)
        else:
            dfTrain = self.dfTrain
        # 如果没有提供测试DataFrame，则从文件中加载。
        if self.dfTest is None:
            dfTest = pd.read_csv(self.testfile)
        else:
            dfTest = self.dfTest
        # 合并训练和测试数据。
        df = pd.concat([dfTrain, dfTest])
        # 初始化特征字典和索引计数器。
        self.feat_dict = {}
        tc = 0
        # 遍历数据的列。
        for col in df.columns:
            # 忽略在ignore_cols中的列。
            if col in self.ignore_cols:
                continue
            # 对于数值型列，直接映射到一个索引。
            if col in self.numeric_cols:
                self.feat_dict[col] = tc
                tc += 1
            else:
                # 对于非数值型列，将唯一值映射到连续的索引。
                us = df[col].unique()
                self.feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
                tc += len(us)
        # 设置特征维度，即最大的特征索引值。
        self.feat_dim = tc


# 定义DataParser类，用于解析数据。
class DataParser(object):
    # 初始化函数，接收一个FeatureDictionary对象。
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict

    # 解析数据的方法，可以从文件或DataFrame中解析，并指定是否有标签列。
    def parse(self, infile=None, df=None, has_label=False):
        # 确保至少提供了文件路径或DataFrame中的一个。
        assert not ((infile is None) and (df is None)), "infile or df at least one is set"
        # 确保不会同时提供文件路径和DataFrame。
        assert not ((infile is not None) and (df is not None)), "only one can be set"
        # 如果提供了DataFrame，进行复制；否则从文件加载。
        if infile is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(infile)
        # 如果有标签列，提取标签并从数据中删除标签和ID列。
        if has_label:
            y = dfi["target"].values.tolist()
            dfi.drop(["id", "target"], axis=1, inplace=True)
        else:
            # 如果没有标签，只提取ID。
            ids = dfi["id"].values.tolist()
            dfi.drop(["id"], axis=1, inplace=True)
        # 创建一个副本用于存储特征值。
        dfv = dfi.copy()
        # 遍历列，进行特征索引和特征值的映射。
        for col in dfi.columns:
            # 忽略在ignore_cols中的列。
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            # 对于数值型列，使用固定的索引。
            if col in self.feat_dict.numeric_cols:
                dfi[col] = self.feat_dict.feat_dict[col]
            else:
                # 对于类别型列，使用映射后的索引，特征值设为1。
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                dfv[col] = 1.
        # 将DataFrame转换为列表格式，用于模型训练。
        Xi = dfi.values.tolist()
        Xv = dfv.values.tolist()
        # 根据是否有标签返回不同的数据结构。
        if has_label:
            return Xi, Xv, y
        else:
            return Xi, Xv, ids
