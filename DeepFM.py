"""
Tensorflow implementation of DeepFM [1]

Reference:
[1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
"""

# 导入numpy库，用于处理数组和矩阵运算。
import numpy as np
# 导入tensorflow库，用于构建和训练深度学习模型。
import tensorflow as tf
# 从sklearn.base导入BaseEstimator和TransformerMixin，这两个类是构建scikit-learn兼容模型的基类。
from sklearn.base import BaseEstimator, TransformerMixin
# 从sklearn.metrics导入roc_auc_score，用于模型评估。
from sklearn.metrics import roc_auc_score
# 导入time库的time函数，用于计算代码段的运行时间。
from time import time
# 从tensorflow.contrib.layers导入batch_norm，用于批量归一化。
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
# 导入YFOptimizer，它是一个优化算法，用于训练神经网络。
from yellowfin import YFOptimizer


# 定义DeepFM类，继承自BaseEstimator和TransformerMixin。
class DeepFM(BaseEstimator, TransformerMixin):
    # 初始化函数，设置DeepFM模型的各种参数。
    def __init__(self, feature_size, field_size,
                 embedding_size=8, dropout_fm=[1.0, 1.0],
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer_type="adam",
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 use_fm=True, use_deep=True,
                 loss_type="logloss", eval_metric=roc_auc_score,
                 l2_reg=0.0, greater_is_better=True):
        # 断言条件，确保模型至少使用FM或深度网络中的一种。
        assert (use_fm or use_deep)
        # 断言条件，确保损失类型是"logloss"或"mse"。
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        # 初始化各种模型参数。
        self.feature_size = feature_size  # 特征字典的大小
        self.field_size = field_size  # 特征字段的大小
        self.embedding_size = embedding_size  # 嵌入层的大小

        self.dropout_fm = dropout_fm  # FM部分的dropout率
        self.deep_layers = deep_layers  # 深度层的结构
        self.dropout_deep = dropout_deep  # 深度层的dropout率
        self.deep_layers_activation = deep_layers_activation  # 深度层的激活函数
        self.use_fm = use_fm  # 是否使用FM组件
        self.use_deep = use_deep  # 是否使用深度网络组件
        self.l2_reg = l2_reg  # L2正则化参数

        self.epoch = epoch  # 迭代次数
        self.batch_size = batch_size  # 批量大小
        self.learning_rate = learning_rate  # 学习率
        self.optimizer_type = optimizer_type  # 优化器类型

        self.batch_norm = batch_norm  # 是否使用批量归一化
        self.batch_norm_decay = batch_norm_decay  # 批量归一化衰减率

        self.verbose = verbose  # 是否显示详细信息
        self.random_seed = random_seed  # 随机种子
        self.loss_type = loss_type  # 损失类型
        self.eval_metric = eval_metric  # 评估指标
        self.greater_is_better = greater_is_better  # 评估指标的方向
        self.train_result, self.valid_result = [], []  # 训练和验证结果

        self._init_graph()  # 初始化计算图

    # 初始化计算图的函数
    def _init_graph(self):
        self.graph = tf.Graph()  # 创建一个新的计算图
        with self.graph.as_default():  # 将此计算图设置为默认计算图

            tf.set_random_seed(self.random_seed)  # 设置随机种子，确保结果可复现

            # 定义输入占位符
            self.feat_index = tf.placeholder(tf.int32, shape=[None, None],
                                             name="feat_index")  # 特征索引，形状为[None, F]
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None],
                                             name="feat_value")  # 特征值，形状为[None, F]
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # 标签，形状为[None, 1]
            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")  # FM部分的dropout占位符
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None],
                                                    name="dropout_keep_deep")  # 深度部分的dropout占位符
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")  # 训练阶段的标志

            self.weights = self._initialize_weights()  # 初始化模型权重

            # 构建模型
            self.embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"],
                                                     self.feat_index)  # 查找输入特征的嵌入，形状为[None, F, K]
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])  # 将feat_value的形状调整为[None, F, 1]
            self.embeddings = tf.multiply(self.embeddings, feat_value)  # 将嵌入值和特征值相乘，模拟加权嵌入

            # ---------- first order term ----------
            self.y_first_order = tf.nn.embedding_lookup(self.weights["feature_bias"],
                                                        self.feat_index)  # 查找输入特征的偏置，形状为[None, F, 1]
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)  # 计算一阶项的和，形状为[None, F]
            self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_keep_fm[0])  # 应用dropout到一阶项

            # ---------- second order term ---------------
            # sum_square part
            self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)  # 计算嵌入的和，形状为[None, K]
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # 将和的平方，形状为[None, K]

            # square_sum part
            self.squared_features_emb = tf.square(self.embeddings)  # 计算嵌入的平方
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # 计算平方的和，形状为[None, K]

            # second order
            self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square,
                                                    self.squared_sum_features_emb)  # 计算二阶项，形状为[None, K]
            self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])  # 应用dropout到二阶项

            # ---------- Deep component ----------
            self.y_deep = tf.reshape(self.embeddings,
                                     shape=[-1, self.field_size * self.embedding_size])  # 将嵌入值重塑为[None, F*K]，为深度组件准备
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])  # 应用dropout到深度组件的输入层
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" % i]),
                                     self.weights["bias_%d" % i])  # 计算每个深度层的输出
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase,
                                                        scope_bn="bn_%d" % i)  # 应用批量归一化
                self.y_deep = self.deep_layers_activation(self.y_deep)  # 应用激活函数
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1 + i])  # 应用dropout到每个深度层

            # ---------- DeepFM ----------
            if self.use_fm and self.use_deep:
                concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep],
                                         axis=1)  # 如果同时使用FM和深度网络，将一阶、二阶和深度组件的输出合并
            elif self.use_fm:
                concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)  # 如果只使用FM，合并一阶和二阶输出
            elif self.use_deep:
                concat_input = self.y_deep  # 如果只使用深度网络，使用深度组件的输出作为最终输入
            self.out = tf.add(tf.matmul(concat_input, self.weights["concat_projection"]),
                              self.weights["concat_bias"])  # 计算最终的模型输出

            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)  # 如果是logloss，通过sigmoid函数将输出转换为概率
                self.loss = tf.losses.log_loss(self.label, self.out)  # 计算logloss
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))  # 如果是mse，计算均方误差
            # l2 regularization on weights
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["concat_projection"])  # 如果使用L2正则化，应用到投影权重
                if self.use_deep:
                    for i in range(len(self.deep_layers)):
                        self.loss += tf.contrib.layers.l2_regularizer(
                            self.l2_reg)(self.weights["layer_%d" % i])  # 应用L2正则化到每个深度层的权重

            # optimizer
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)  # 使用Adam优化器
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(
                    self.loss)  # 使用Adagrad优化器
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                    self.loss)  # 使用梯度下降优化器
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)  # 使用Momentum优化器
            elif self.optimizer_type == "yellowfin":
                self.optimizer = YFOptimizer(learning_rate=self.learning_rate, momentum=0.0).minimize(
                    self.loss)  # 使用Yellowfin优化器

            # init
            self.saver = tf.train.Saver()  # 创建Saver对象，用于保存模型
            init = tf.global_variables_initializer()  # 初始化所有变量
            self.sess = self._init_session()  # 创建会话
            self.sess.run(init)  # 在会话中运行变量初始化

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()  # 获取每个变量的形状
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value  # 计算每个变量的总参数量
                total_parameters += variable_parameters  # 累加总参数量
            if self.verbose > 0:
                print("#params: %d" % total_parameters)  # 如果verbose大于0，打印参数总数

    # 创建会话的函数
    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})  # 配置会话，不使用GPU
        config.gpu_options.allow_growth = True  # 允许GPU内存增长
        return tf.Session(config=config)  # 创建并返回会话

    # 初始化模型权重的函数
    def _initialize_weights(self):
        weights = dict()  # 创建一个空字典用于存储权重

        # embeddings
        weights["feature_embeddings"] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name="feature_embeddings")  # 初始化特征嵌入矩阵，形状为[feature_size, embedding_size]
        weights["feature_bias"] = tf.Variable(
            tf.random_uniform([self.feature_size, 1], 0.0, 1.0), name="feature_bias")  # 初始化特征偏置向量，形状为[feature_size, 1]

        # deep layers
        num_layer = len(self.deep_layers)  # 深度层的数量
        input_size = self.field_size * self.embedding_size  # 计算输入大小
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))  # 使用glorot初始化方法计算初始化标准差
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)  # 初始化第一层权重
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                        dtype=np.float32)  # 初始化第一层偏置
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))  # 为每一层计算glorot标准差
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                dtype=np.float32)  # 初始化深度层权重
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 初始化深度层偏置

        # final concat projection layer
        if self.use_fm and self.use_deep:
            input_size = self.field_size + self.embedding_size + self.deep_layers[-1]  # 如果同时使用FM和深度组件，计算总输入大小
        elif self.use_fm:
            input_size = self.field_size + self.embedding_size  # 如果只使用FM组件，计算总输入大小
        elif self.use_deep:
            input_size = self.deep_layers[-1]  # 如果只使用深度组件，计算总输入大小
        glorot = np.sqrt(2.0 / (input_size + 1))  # 计算glorot标准差
        weights["concat_projection"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
            dtype=np.float32)  # 初始化最终投影层权重
        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)  # 初始化最终投影层偏置

        return weights  # 返回权重字典

    # 批量归一化层的实现函数
    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)  # 训练阶段的批量归一化
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)  # 推理阶段的批量归一化
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)  # 根据train_phase选择使用训练还是推理阶段的批量归一化
        return z  # 返回批量归一化的结果

    # 获取批次数据的函数
    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size  # 计算批次开始位置
        end = (index + 1) * batch_size  # 计算批次结束位置
        end = end if end < len(y) else len(y)  # 如果结束位置超出数据范围，调整为数据长度
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]  # 返回批次数据

    # 同步洗牌三个列表的函数
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()  # 获取当前随机状态
        np.random.shuffle(a)  # 洗牌列表a
        np.random.set_state(rng_state)  # 重置随机状态
        np.random.shuffle(b)  # 洗牌列表b
        np.random.set_state(rng_state)  # 重置随机状态
        np.random.shuffle(c)  # 洗牌列表c

    # 批次训练的函数
    def fit_on_batch(self, Xi, Xv, y):
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.label: y,
                     self.dropout_keep_fm: self.dropout_fm,
                     self.dropout_keep_deep: self.dropout_deep,
                     self.train_phase: True}  # 创建feed字典
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)  # 运行计算图，获取损失和执行优化器
        return loss  # 返回损失值

    # 模型训练函数
    def fit(self, Xi_train, Xv_train, y_train,
            Xi_valid=None, Xv_valid=None, y_valid=None,
            early_stopping=False, refit=False):
        """
        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                         indi_j is the feature index of feature field j of sample i in the training set
        :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                         vali_j is the feature value of feature field j of sample i in the training set
                         vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
        :param y_train: label of each sample in the training set
        :param Xi_valid: list of list of feature indices of each sample in the validation set
        :param Xv_valid: list of list of feature values of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :param early_stopping: perform early stopping or not
        :param refit: refit the model on the train+valid dataset or not
        :return: None
        """
        has_valid = Xv_valid is not None  # 检查是否有验证数据集
        for epoch in range(self.epoch):  # 循环迭代次数
            t1 = time()  # 记录开始时间
            self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)  # 同步洗牌训练数据
            total_batch = int(len(y_train) / self.batch_size)  # 计算总批次数
            for i in range(total_batch):  # 循环每个批次
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)  # 获取批次数据
                self.fit_on_batch(Xi_batch, Xv_batch, y_batch)  # 执行批次训练

            # evaluate training and validation datasets
            train_result = self.evaluate(Xi_train, Xv_train, y_train)  # 评估训练数据集
            self.train_result.append(train_result)  # 保存训练结果
            if has_valid:
                valid_result = self.evaluate(Xi_valid, Xv_valid, y_valid)  # 评估验证数据集
                self.valid_result.append(valid_result)  # 保存验证结果
            if self.verbose > 0 and epoch % self.verbose == 0:  # 如果需要打印详细信息
                if has_valid:
                    print("[%d] train-result=%.4f, valid-result=%.4f [%.1f s]"
                          % (epoch + 1, train_result, valid_result, time() - t1))  # 打印训练和验证结果
                else:
                    print("[%d] train-result=%.4f [%.1f s]"
                          % (epoch + 1, train_result, time() - t1))  # 打印训练结果
            if has_valid and early_stopping and self.training_termination(self.valid_result):  # 如果使用早停
                break  # 如果满足早停条件，则停止训练

        # fit a few more epoch on train+valid until result reaches the best_train_score
        if has_valid and refit:  # 如果需要在训练+验证数据集上重新训练
            if self.greater_is_better:  # 如果评估指标越大越好
                best_valid_score = max(self.valid_result)  # 获取最佳验证结果
            else:
                best_valid_score = min(self.valid_result)  # 获取最佳验证结果
            best_epoch = self.valid_result.index(best_valid_score)  # 获取最佳验证结果的索引
            best_train_score = self.train_result[best_epoch]  # 获取对应的训练结果
            Xi_train = Xi_train + Xi_valid  # 合并训练和验证的特征索引
            Xv_train = Xv_train + Xv_valid  # 合并训练和验证的特征值
            y_train = y_train + y_valid  # 合并训练和验证的标签
            for epoch in range(100):  # 循环额外的迭代次数
                self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)  # 同步洗牌所有数据
                total_batch = int(len(y_train) / self.batch_size)  # 重新计算总批次数
                for i in range(total_batch):  # 循环每个批次
                    Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train,
                                                                 self.batch_size, i)  # 获取批次数据
                    self.fit_on_batch(Xi_batch, Xv_batch, y_batch)  # 执行批次训练
                # check
                train_result = self.evaluate(Xi_train, Xv_train, y_train)  # 评估所有数据
                if abs(train_result - best_train_score) < 0.001 or \
                        (self.greater_is_better and train_result > best_train_score) or \
                        ((not self.greater_is_better) and train_result < best_train_score):
                    break  # 如果训练结果接近最佳训练结果或者达到预期，则停止训练

    # 早停判断函数
    def training_termination(self, valid_result):
        if len(valid_result) > 5:  # 如果验证结果的数量大于5
            if self.greater_is_better:  # 如果评估指标越大越好
                if valid_result[-1] < valid_result[-2] and \
                        valid_result[-2] < valid_result[-3] and \
                        valid_result[-3] < valid_result[-4] and \
                        valid_result[-4] < valid_result[-5]:
                    return True  # 如果连续五次验证结果递减，则返回True
            else:
                if valid_result[-1] > valid_result[-2] and \
                        valid_result[-2] > valid_result[-3] and \
                        valid_result[-3] > valid_result[-4] and \
                        valid_result[-4] > valid_result[-5]:
                    return True  # 如果连续五次验证结果递增，则返回True
        return False  # 否则返回False

    # 预测函数
    def predict(self, Xi, Xv):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y
        dummy_y = [1] * len(Xi)  # 创建一个虚拟的标签列表
        batch_index = 0
        Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)  # 获取第一个批次数据
        y_pred = None
        while len(Xi_batch) > 0:  # 如果批次数据不为空
            num_batch = len(y_batch)  # 获取批次数据的数量
            feed_dict = {self.feat_index: Xi_batch,
                         self.feat_value: Xv_batch,
                         self.label: y_batch,
                         self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                         self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                         self.train_phase: False}  # 创建feed字典，关闭dropout和批量归一化
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)  # 运行计算图，获取输出

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))  # 如果是第一批次，初始化预测结果
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))  # 否则，将当前批次结果拼接到预测结果上

            batch_index += 1  # 批次索引自增
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)  # 获取下一个批次数据

        return y_pred  # 返回预测结果

    # 评估函数
    def evaluate(self, Xi, Xv, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        """
        y_pred = self.predict(Xi, Xv)  # 获取预测结果
        return self.eval_metric(y, y_pred)  # 计算评估指标
