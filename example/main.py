# 导入os模块，用于处理操作系统级别的操作，如文件路径、环境变量等。
import os
# 导入sys模块，用于访问与Python解释器紧密相关的变量和函数。
import sys

# 导入numpy库，通常用于科学计算中的数组操作。
import numpy as np
# 导入pandas库，用于数据分析和处理。
import pandas as pd
# 导入tensorflow库，一个主要用于深度学习项目的强大库。
import tensorflow as tf
# 导入matplotlib的pyplot模块，用于绘图。
from matplotlib import pyplot as plt
# 从sklearn.metrics导入make_scorer，用于创建评分对象。
from sklearn.metrics import make_scorer
# 从sklearn.model_selection导入StratifiedKFold，用于分层抽样交叉验证。
from sklearn.model_selection import StratifiedKFold

# 导入本地config模块，通常包含配置信息。
import config
# 从metrics模块导入gini_norm，一个评估函数。
from metrics import gini_norm
# 从DataReader模块导入FeatureDictionary和DataParser，用于数据处理。
from DataReader import FeatureDictionary, DataParser

# 将父目录("..")添加到sys.path中，这样可以导入父目录中的模块。
sys.path.append("..")
# 从DeepFM模块导入DeepFM类。
from DeepFM import DeepFM

# 创建一个make_scorer的实例，用于计算gini系数，这是一个常见的模型评估指标。
gini_scorer = make_scorer(gini_norm, greater_is_better=True, needs_proba=True)


# 定义一个函数_load_data，用于加载和预处理数据。
def _load_data():
    # 从config配置中读取训练文件和测试文件的路径，读取数据为pandas的DataFrame。
    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)

    # 定义一个内部函数preprocess，用于数据预处理。
    def preprocess(df):
        # 从DataFrame中排除"id"和"target"列，生成新的列列表。
        cols = [c for c in df.columns if c not in ["id", "target"]]
        # 创建新列"missing_feat"，计算每行中值为-1的数量，表示缺失特征的数量。
        df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)
        # 创建新列"ps_car_13_x_ps_reg_03"，是两个特征列的乘积，可能是为了生成交互特征。
        df["ps_car_13_x_ps_reg_03"] = df["ps_car_13"] * df["ps_reg_03"]
        return df

    # 对训练和测试数据集应用preprocess函数。
    dfTrain = preprocess(dfTrain)
    dfTest = preprocess(dfTest)

    # 再次排除"id"和"target"列，以及config中指定的忽略列。
    cols = [c for c in dfTrain.columns if c not in ["id", "target"]]
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    # 提取训练集和测试集的特征和目标变量。
    X_train = dfTrain[cols].values
    y_train = dfTrain["target"].values
    X_test = dfTest[cols].values
    ids_test = dfTest["id"].values
    # 根据配置文件中定义的类别特征索引。
    cat_features_indices = [i for i, c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    # 返回处理后的数据集和相关变量。
    return dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices


# 定义一个函数_run_base_model_dfm，用于训练和评估DeepFM模型。
def _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params):
    # 实例化FeatureDictionary，用于创建特征字典，该字典将特征名称映射到其索引。
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS)
    # 实例化DataParser，用于解析数据，生成特征索引和特征值列表。
    data_parser = DataParser(feat_dict=fd)
    # 解析训练数据和测试数据。
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    Xi_test, Xv_test, ids_test = data_parser.parse(df=dfTest)

    # 设置DeepFM模型参数，包括特征大小和字段大小。
    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0])

    # 初始化用于保存训练期间和交叉验证结果的数组。
    y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
    y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)
    # 辅助函数_get，用于从列表中选择特定索引的元素。
    _get = lambda x, l: [x[i] for i in l]
    gini_results_cv = np.zeros(len(folds), dtype=float)
    gini_results_epoch_train = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    gini_results_epoch_valid = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    # 进行交叉验证，循环每个fold，训练并评估模型。
    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)

        # 初始化并训练DeepFM模型。
        dfm = DeepFM(**dfm_params)
        dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)

        # 保存训练和验证结果，进行预测。
        y_train_meta[valid_idx, 0] = dfm.predict(Xi_valid_, Xv_valid_)
        y_test_meta[:, 0] += dfm.predict(Xi_test, Xv_test)

        # 计算并保存Gini系数结果。
        gini_results_cv[i] = gini_norm(y_valid_, y_train_meta[valid_idx])
        gini_results_epoch_train[i] = dfm.train_result
        gini_results_epoch_valid[i] = dfm.valid_result

    # 平均测试结果。
    y_test_meta /= float(len(folds))

    # 根据模型类型保存结果。
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "DeepFM"
    elif dfm_params["use_fm"]:
        clf_str = "FM"
    elif dfm_params["use_deep"]:
        clf_str = "DNN"
    # 输出模型性能。
    print("%s: %.5f (%.5f)" % (clf_str, gini_results_cv.mean(), gini_results_cv.std()))
    # 生成结果文件名并保存预测结果。
    filename = "%s_Mean%.5f_Std%.5f.csv" % (clf_str, gini_results_cv.mean(), gini_results_cv.std())
    _make_submission(ids_test, y_test_meta, filename)

    # 绘制训练和验证结果的图形。
    _plot_fig(gini_results_epoch_train, gini_results_epoch_valid, clf_str)

    # 返回训练和测试元数据。
    return y_train_meta, y_test_meta


# 定义一个函数_make_submission，用于生成提交的结果文件。
def _make_submission(ids, y_pred, filename="submission.csv"):
    # 创建一个DataFrame，并保存为CSV文件。
    pd.DataFrame({"id": ids, "target": y_pred.flatten()}).to_csv(
        os.path.join(config.SUB_DIR, filename), index=False, float_format="%.5f")


# 定义一个函数_plot_fig，用于绘制模型的训练和验证Gini结果。
def _plot_fig(train_results, valid_results, model_name):
    # 设置颜色和样式。
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_results.shape[1] + 1)
    plt.figure()
    legends = []
    # 绘制每个fold的训练和验证结果。
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d" % (i + 1))
        legends.append("valid-%d" % (i + 1))
    # 设置图表标题和标签。
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s" % model_name)
    plt.legend(legends)
    # 保存图表并关闭。
    plt.savefig("./fig/%s.png" % model_name)
    plt.close()


# 调用_load_data函数，加载数据。
dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = _load_data()

# 生成folds，用于交叉验证。
folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                             random_state=config.RANDOM_SEED).split(X_train, y_train))

# 初始化DeepFM模型参数，并运行模型。
dfm_params = {
    "use_fm": True,  # 使用因子分解机特性。
    "use_deep": True,  # 使用深度网络特性。
    "embedding_size": 8,  # 嵌入层大小。
    "dropout_fm": [1.0, 1.0],  # 因子分解机层的dropout比率。
    "deep_layers": [32, 32],  # 深度网络层的结构。
    "dropout_deep": [0.5, 0.5, 0.5],  # 深度网络层的dropout比率。
    "deep_layers_activation": tf.nn.relu,  # 深度网络层的激活函数。
    "epoch": 30,  # 训练轮数。
    "batch_size": 1024,  # 每批训练的样本数。
    "learning_rate": 0.001,  # 学习率。
    "optimizer_type": "adam",  # 优化器类型。
    "batch_norm": 1,  # 是否使用批量归一化。
    "batch_norm_decay": 0.995,  # 批量归一化的衰减率。
    "l2_reg": 0.01,  # L2正则化系数。
    "verbose": True,  # 是否显示详细信息。
    "eval_metric": gini_norm,  # 评估指标。
    "random_seed": config.RANDOM_SEED  # 随机种子。
}
# 调用_run_base_model_dfm函数，执行模型训练和评估。
y_train_dfm, y_test_dfm = _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params)

# 设置仅使用因子分解机特性的参数，并运行模型。
fm_params = dfm_params.copy()
fm_params["use_deep"] = False
y_train_fm, y_test_fm = _run_base_model_dfm(dfTrain, dfTest, folds, fm_params)

# 设置仅使用深度网络特性的参数，并运行模型。
dnn_params = dfm_params.copy()
dnn_params["use_fm"] = False
y_train_dnn, y_test_dnn = _run_base_model_dfm(dfTrain, dfTest, folds, dnn_params)
