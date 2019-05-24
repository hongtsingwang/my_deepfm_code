# coding=utf-8

import numpy as np
import tensorflow as tf
from time import time
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.metrics import roc_auc_score


class DeepFM(BaseEstimator, TransformerMixin):
    """
    使用BaseEstimator, TransformerMixin 两个作为基类，可以自动实现sklearn中的机器学习模型中的fit, transform, fit_transform
    基类：
    1. BaseEstimator: 可得到两个方法get_params()和set_params()
    2. 继承TransformerMixin 可以获得fit_transform功能
    """
    
    def __init__(self, feature_size, field_size, embedding_size=8, dropout_fm=[1.0, 1.0],
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],
                 deep_layer_activation=tf.nn.relu
                 , epoch=10
                 , batch_size=256
                 , learning_rate=0.001
                 , optimizer="adam"
                 , batch_norm=0
                 , batch_norm_decay=0.995
                 , verbose=False
                 , random_seed=42
                 , use_fm=True
                 , use_deep=True
                 , loss_type="loglss"
                 , eval_metric=roc_auc_score
                 , l2_reg=0.0
                 , greater_is_better=True):
        """
        类初始化
        :param feature_size:
        :param field_size:
        :param embedding_size:
        :param dropout_fm:
        :param deep_layers: deep部分每层神经元的个数 TODO 神经元的表述未必准确
        :param dropout_deep:
        :param deep_layer_activation:
        :param epoch:
        :param batch_size:
        :param learning_rate:
        :param optimizer:
        :param batch_norm:
        :param batch_norm_decay:
        :param verbose:
        :param random_seed: 设置随机数
        :param use_fm: 是否在模型中添加FM模块
        :param use_deep: 是否在模型中添加Deep模块
        :param loss_type:
        :param eval_metric:
        :param l2_reg: l2正则化的参数
        :param greater_is_better:
        :return:
        """
        # FM模型和Deep模型至少用一个
        assert (use_fm or use_deep)
        # mse: mean square error 回归中最常见的了， 不要惊慌
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"
        
        # 参数读进来
        # TODO feature_size 和field_size还是不太理解
        # TODO embedding_size
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        
        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers
        self.dropout_dep = dropout_deep
        self.deep_layers_activation = deep_layer_activation
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.l2_reg = l2_reg
        
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer
        
        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay
        
        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        
        # 存储训练的结果和验证集的结果
        self.train_result = []
        self.valid_result = []
        
        # 自有函数前加下划线， 好习惯
        self._init_graph()
    
    def _initialize_weights(self):
        """
        对模型中权重部分的初始化
        :return:
        """
        # embedding层
        weights = dict()
        # 这个是图中最下面一层的分布，从稀疏特征向dense feature size转化
        # W 高斯分布 维度 feature_size X embedding_size
        # 均值为0，方差0.01
        weights['feature_embeddings'] = tf.Variable(tf.random_normal([self.feature_size,
                                                                      self.embedding_size],
                                                                     0.0,
                                                                     0.01),
                                                    name='feature_embeddings')
        # B 高斯分布， 均值为0， 方差为1
        weights['feature_bias'] = tf.Variable(tf.random_normal([self.feature_size, 1], 0.0, 1.0), name='feature_bias')
        
        # deep layer层
        num_layer = len(self.deep_layers)  # 获得deep部分共有多少层
        # TODO field size 不太理解 每层神经元的个数， 好像不太对
        input_size = self.field_size * self.embedding_size
        
        # 每层权重初始化 TODO 初始化原理不太理解
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32
        )
        
        weights['bias_0'] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])), dtype=np.float32
        )
        
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]
        
        # TODO  看着意思是field_size 和 embedding_size属于FM， deep_layers[-1]属于DEEP
        if self.use_fm and self.use_deep:
            input_size = self.field_size + self.embedding_size + self.deep_layers[-1]
        elif self.use_fm:
            input_size = self.field_size + self.embedding_size
        elif self.use_deep:
            input_size = self.deep_layers[-1]
        
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights['concat_projection'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                                                   dtype=np.float32)
        weights['concat_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)
        return weights
    
    def _init_graph(self):
        # 指定TensorFlow空白图的对象
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            # 表示将新生成的图作为TensorFlow运行环境的默认图
            tf.set_random_seed(self.random_seed)
            
            # 指定feature对应的索引
            self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name="feature_index")
            
            # 指定feature该索引对应的值
            self.feat_value = tf.placeholder(tf.int32, shape=[None, None], name="feature_value")
            
            # 指定label
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
            
            # FM模型部分居然也有dropout, 而且还是数组 TODO 不太理解
            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_fm')
            
            # DEEP部分的dropout 为什么也是有维度的， 我理解是一个值呢， TODO
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name='dropout_deep_deep')
            
            # TODO train_phase未知
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')
            
            # 设置了模型， 该设置网络中权重的初始值了
            self.weights = self._initialize_weights()
            
            # model
            # 根据feature的index， 查询出对应的weight来
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.feat_index)  # N * F * K
            # 将feat_value转换数组形式
            # TODO 话说， 我一直对神经网络中的维度搞不太清楚。 为什么要做维度转换呢？
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
            
            # 对embeddings进行计算 multiply tensor中对应位置相乘
            self.embeddings = tf.multiply(self.embeddings, feat_value)
            
            # FM 模型部分
            # first order term
            # 查找bias
            self.y_first_order = tf.nn.embedding_lookup(self.weights['feature_bias'], self.feat_index)
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)
            self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_keep_fm[0])
            
            # second order term
            # sum-square-part
            self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)  # None * k
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K
            
            # squre-sum-part
            self.squared_features_emb = tf.square(self.embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K
            
            # second order
            self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)
            self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])
            
            # Deep 部分
            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size])
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
            
            # 下面这部分是前向神经网络的操作， 矩阵相乘， 后加偏置， 然后激活
            # 每层函数的dropout概率是不一样的。
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" % i]), self.weights["bias_%d" % i])
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[i + 1])
            
            # 根据参数确定 模型架构
            # 既用FM， 又用DEEP则为deepfm
            if self.use_deep and self.use_fm:
                concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)
            elif self.use_fm:
                concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)
            elif self.use_deep:
                concat_input = self.y_deep
            
            self.out = tf.add(tf.matmul(concat_input, self.weights['concat_projection']), self.weights['concat_bias'])
            
            # loss
            # 如果是logloss， 说明是分类问题， 最后一层加一个sigmoid函数
            # 如果是mse, 使用l2_loss来看结果
            
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
            # l2 regularization on weights
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["concat_projection"])
                # 如果有deep部分， 那么DNN网络中的参数也应该加进来。
                if self.use_deep:
                    for i in range(len(self.deep_layers)):
                        self.loss += tf.contrib.layers.l2_regularizer(
                            self.l2_reg)(self.weights["layer_%d" % i])
            
            # 指定优化器
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)
            
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
            
            # 计算网络中参量的总数
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)
    
    def get_batch(self, Xi, Xv, y, batch_size, index):
        # 根据batch_size获取对应的mini_batch 样本 进行训练
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]
    
    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c):
        # 打乱顺序 理解
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
    
    def evaluate(self, Xi, Xv, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        """
        # 根据预测的结果算分。
        y_pred = self.predict(Xi, Xv)
        return self.eval_metric(y, y_pred)
    
    def predict(self, Xi, Xv):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # 在一定程度上是理解了。 但肯定不够深刻
        # dummy y
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.feat_index: Xi_batch,
                         self.feat_value: Xv_batch,
                         self.label: y_batch,
                         self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                         self.dropout_keep_deep: [1.0] * len(self.dropout_dep),
                         self.train_phase: False}  # TODO train_phase含义是？
            # dropout_keep_fm， 与 dropout_keep_deep 为什么还是数组形式
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)
            # 将结果拼接起来
            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))
            
            batch_index += 1
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
    
    def fit_on_batch(self, Xi, Xv, y):
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.label: y,
                     self.dropout_keep_fm: self.dropout_fm,
                     self.dropout_keep_deep: self.dropout_dep,
                     self.train_phase: True}
        
        loss, opt = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        
        return loss
    
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
        has_valid = Xv_valid is not None
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
                self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
            
            # evaluate training and validation datasets
            train_result = self.evaluate(Xi_train, Xv_train, y_train)
            self.train_result.append(train_result)
            if has_valid:
                valid_result = self.evaluate(Xi_valid, Xv_valid, y_valid)
                self.valid_result.append(valid_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    print("[%d] train-result=%.4f, valid-result=%.4f [%.1f s]"
                          % (epoch + 1, train_result, valid_result, time() - t1))
                else:
                    print("[%d] train-result=%.4f [%.1f s]"
                          % (epoch + 1, train_result, time() - t1))
                if has_valid and early_stopping and self.training_termination(self.valid_result):
                    break
                # fit a few more epoch on train+valid until result reaches the best_train_score
                if has_valid and refit:
                    if self.greater_is_better:
                        best_valid_score = max(self.valid_result)
                    else:
                        best_valid_score = min(self.valid_result)
                    best_epoch = self.valid_result.index(best_valid_score)
                    best_train_score = self.train_result[best_epoch]
                    Xi_train = Xi_train + Xi_valid
                    Xv_train = Xv_train + Xv_valid
                    y_train = y_train + y_valid
                    for epoch in range(100):
                        self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
                        total_batch = int(len(y_train) / self.batch_size)
                        for i in range(total_batch):
                            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train,
                                                                         self.batch_size, i)
                            self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
                        # check
                        train_result = self.evaluate(Xi_train, Xv_train, y_train)
                        if abs(train_result - best_train_score) < 0.001 or \
                                (self.greater_is_better and train_result > best_train_score) or \
                                ((not self.greater_is_better) and train_result < best_train_score):
                            break
    
    def training_termination(self, valid_result):
        # 理解
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                        valid_result[-2] < valid_result[-3] and \
                        valid_result[-3] < valid_result[-4] and \
                        valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                        valid_result[-2] > valid_result[-3] and \
                        valid_result[-3] > valid_result[-4] and \
                        valid_result[-4] > valid_result[-5]:
                    return True
        return False
