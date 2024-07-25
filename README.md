# tensorflow-DeepFM

你好，我是悦创。

本仓库为私教学员答疑仓库，有问题可以添加微信付费咨询：Jiabcdefh

## 1. 操作步骤

1. 抓取本仓库：

```bash
git clone git@github.com:AndersonHJB/tensorflow-DeepFM.git
```

2. 数据集下载：[Porto Seguro's Safe Driver Prediction competition on Kaggle](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction).

3. conda 安装正确的情况下，命令行进入到该项目：

```bash
conda env create -f environment.yml
```

4. 运行模型

激活环境：使用以下命令激活新创建的环境：

```bash
conda activate 环境名
```

```bash
python main.py
```

## 2. 建议

1. 推荐使用 Anaconda + Pycharm


## 3. 问题解答

### 1. 模型示意图中不同颜色的线、参数名、参数在代码中的变量名的对应关系

- **红色线**：表示一阶特征的线性部分，对应代码中的`self.y_first_order`，形状为`[None, F]`。
- **蓝色线**：表示二阶交互特征，对应代码中的`self.y_second_order`，形状为`[None, K]`。
- **绿色线**：表示深度网络部分，对应代码中的`self.y_deep`，在不同深度层之间形状会变化，最终输出形状为`[None, 1]`。

### 2. 总结整个训练+评估的流程

整个流程包括以下步骤：

1. **数据加载和预处理**：通过`_load_data()`函数加载训练和测试数据，进行必要的预处理。
2. **特征解析**：使用`FeatureDictionary`和`DataParser`类从数据中提取特征，生成特征索引和值。
3. **模型初始化和配置**：根据提供的参数初始化DeepFM模型，包括一阶特征、二阶特征和深度网络部分。
4. **交叉验证**：使用交叉验证方法（如StratifiedKFold）来训练和评估模型，每次训练结束后对验证集进行评估。
5. **模型训练**：通过`fit()`方法在训练数据上训练模型，可选择是否使用早停来避免过拟合。
6. **模型评估**：使用`evaluate()`方法计算验证集上的性能指标，如Gini系数或AUC。
7. **结果汇总和输出**：整合交叉验证中的结果，输出平均性能指标，并保存模型预测的结果。

### 3. `gini_results_epoch_train`的形状

根据代码初始化部分，`gini_results_epoch_train`的形状为`[len(folds), dfm_params["epoch"]]`，其中`len(folds)`是交叉验证的折数，`dfm_params["epoch"]`是迭代次数。

### 4. `feature_size`和`field_size`的值与区别

- **`feature_size`**：特征字典的总尺寸，即所有不同的特征的数量。在`FeatureDictionary`类中计算得到。
- **`field_size`**：数据中原始字段的数量，即数据集中不同的列数（忽略ID和目标列）。

二者的区别在于`feature_size`通常大于`field_size`，因为特征字典考虑了所有可能的特征值（尤其是类别特征），而`field_size`仅考虑字段本身。

### 5. 代码中使用的是几折交叉验证法？每一折数据的长度是多少？
根据`folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, ...).split(X_train, y_train))`的设置，使用的是`config.NUM_SPLITS`折交叉验证法。每一折的数据长度为数据总长度除以折数。

### 6. `self.embeddings`的形状

`self.embeddings`的形状为`[None, F, K]`，其中`F`是`field_size`，`K`是嵌入的维度（`embedding_size`）。

### 7. `self.y_first_order`对应FM方程中的哪部分？

`self.y_first_order`对应 FM 方程中的一阶线性部分，即线性关系的总和。

### 8. `self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)`对应的公式

这是 FM 模型二阶交互项的实现，对应的数学表达式为：

$\text{Second Order} = \frac{1}{2} \left( \left(\sum_{k=1}^K \left(\sum_{i=1}^n v_{ik} x_i \right)^2 \right) - \sum_{k=1}^K \sum_{i=1}^n v_{ik}^2 x_i^2 \right)$
这里 $v_{ik}$ 是第 $i$ 个特征在第$k$维的嵌入值，$x_i$ 是特征值。

### 9. DeepFM的`weights["concat_projection"]`形状

根据 DeepFM 的代码设置，如果同时使用FM和深度网络，则`input_size = self.field_size + self.embedding_size + self.deep_layers[-1]`。因此，`weights["concat_projection"]`的形状为`[input_size, 1]`。

### 10. 为什么损失函数选择 logloss 而不是 mse？

在分类任务中，特别是二分类问题如 CTR 预测，logloss（对数损失）提供了概率输出的好处，它直接关联到预测值和实际值之间的概率差异，而 mse（均方误差）更多用于回归问题。 logloss 对于模型性能的优化更为敏感和直接。

### 11. 对比`fit_on_batch`函数和`evaluate`函数

- `fit_on_batch`：这个函数主要用于模型的训练，通过输入批次数据来不断更新模型的权重。
- `evaluate`：这个函数用于评估模型的性能，通常在验证集或测试集上执行，不会改变模型的权重。

训练过程涉及权重更新以最小化损失函数，而验证过程则用于监控模型的泛化能力，确保没有过拟合。

### 拓展

- `tf.multiply`：逐元素相乘。
- `tf.matmul`：矩阵乘法。
- `tf.tensordot`：张量点积，可用于高维矩阵的点乘。
- `tf.reduce_sum`：对张量的元素进行求和。
- `tf.linalg.inner`：计算内积。
- `tf.keras.layers.Dot`：计算两个张量的点积。

这些函数和类在 TensorFlow 和 Keras 中用于不同类型的数学操作，它们在深度学习模型中用于实现各种层和操作，如特征交叉、权重计算等。
