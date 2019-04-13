### scaler
#### 标准化
去除均值和方差缩放：通过(X-X_mean)/std计算每个属性(每列)，进而使所有数据聚集在0附近，方差为1
* sklearn.preprocessing.scale()
  直接将给定数据进行标准化


### standardScaler
#### sklearn.preprocessing.StandardScaler()
* 可保存训练集中的均值、方差参数，然后直接用于转换测试集数据。


### MinMaxScaler

#### 缩放到指定范围

* 将属性缩放到一个指定的最大和最小值（通常是1-0）之间，这样处理可对方差非常小的属性增强其稳定性，也可维持稀疏矩阵中为0的条目。
    preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
计算公式：
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (max - min) + min
其中
    feature_range : tuple (min, max), default=(0, 1)

也可直接应用fit_transform(X)实现fit和transform功能。



### Normalizer

#### 正则化

* 对每个样本计算其p-范数，再对每个元素除以该范数，这使得每个处理后样本的p-范数（l1-norm,l2-norm）等于1。如果后续要使用二次型等方法计算两个样本之间的相似性会有用。
preprocessing.Normalizer(norm=’l2’, copy=True)

