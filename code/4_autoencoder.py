"""
自编码器（Autoencoder），顾名思义，即可以使用自身的高阶特征编码自己。也是一种神经网络，它的输入和输出是一致的，它借助稀疏编码的思想，
目的是使用稀疏的一些高阶特征重新组合来重构自己。因此，它的特征非常明显：第一，期望输入/输出一致；第二，希望使用高阶特征来重构自己，而不只是复制像素点。
"""
import numpy as np 
import sklearn.preprocessing as prep 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test
 
def xavier_init(fan_in, fan_out, constant=1):  #fan_in是输入节点的数量, fan_out是输出节点的数量
    low = -constant* np.sqrt(6.0/( fan_in + fan_out))
    high = constant* np.sqrt(6.0/( fan_in + fan_out))
    return tf.random_uniform((fan_in,fan_out),
                                  minval = low, maxval=high,
                                  dtype= tf.float32)
class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self,n_input, n_hidden, transfer_function=tf.nn.softplus,
                  optimizer= tf.train.AdamOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale =tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()

        self.weights = network_weights
 
        self.x = tf.placeholder(tf.float32,[None,self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(
                           self.x +scale*tf.random_normal((n_input,)),
                           self.weights['w1']),self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(
                                self.hidden, self.weights['w2']),self.weights['b2'])
 
 
#接下来定义自编码器的损失函数，这里直接使用平方误差（Squared Error)作为cost，即用tf.subtract计算输出（self.reconstruction)与输入（self.x)之差，
# 再使用tf.pow求差的平方，最后使用tf.reduce_sum求和即可得到平方误差。再定义训练操作为优化器self.optimizer对损失self.cost进行优化。最后创建Session,并初始化自编码器的全部模型参数。
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(
                                        self.reconstruction, self.x),2.0))
        self.optimizer = optimizer.minimize(self.cost)
 
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
    
    def  _initialize_weights(self):
        all_weights = dict()    #先创建一个名为all_weights的字典
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype= tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,self.n_input], dtype=tf.float32))    #对于输出层self.reconstruction，因为没有使用激活函数，这里将w2,b2全部初始化为0即可。
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights
    
    def partial_fit(self, X):     #函数partial_fit做的就是用一个batch数据进行训练并返回当前的损失cost。
        cost, opt = self.sess.run((self.cost, self.optimizer),    #让Session执行两个计算图的节点
            feed_dict = {self.x:X, self.scale: self.training_scale}  )
        return cost
 
    #这个函数是在自编码器训练完毕后，在测试集上对模型进行性能评测时会用到的，它不会像partial_fit那样触发训练操作。
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X,
                               self.scale: self.training_scale})
    #我们还定义了transform函数，它返回自编码器隐含层的输出结果。它的目的是提供一个接口来获取抽象后的特征，自编码器的隐含层的最主要功能就是学习出数据中的高阶特征。
    def transform(self, X):
        return self.sess.run( self.hidden, feed_dict = {self.x: X, self.scale: self.training_scale } )
 
    #我们再定义generate函数，它将隐含层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据。这个接口和前面的transform正好将整个自编码器拆分为两个部分
    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size= self.weights["b1"])
        return self.sess.run(self.reconstruction,
                             feed_dict={self.hidden:hidden })
 
    #接下来定义reconstruct函数，它整体运行一遍复原过程，包括提取高阶特征和通过高阶特征复原数据，即包括transform和generate两块。输入数据是原数据，输出数据是复原后的的数据
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x:X,
                                   self.scale: self.training_scale })
    #这里的getWeights函数是获取隐含层的权重w1
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
    #而getBiases函数则是获取隐含层的偏置系数b1
    def getBiases(self):
        return self.sess.run(self.weights['b1'])
    
    #先在训练数据上fit出一个共用的Scaler，方法是先减去均值，再除以标准差，让数据变成0均值，且标准差为1的分布。我们直接使用sklearn.preprossing的StandardScaler这个类
def standerd_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_test, X_train
    #再定义一个获取随机block数据的函数：
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]


#至此，去噪自编码器的class就全部定义完了，包括神经网络的设计，权重的初始化，以及几个常用的成员函数（transform,generate等，它们属于计算图中的子图），接下来使用定义好的AGN自编码器在MNIST数据集上进行一些简单的性能测试，看看模型对数据的复原效果究竟如何。
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
 
 
n_samples = int(mnist.train.num_examples)   #总训练样本数
training_epochs = 20
batch_size = 128
display_step = 1 #每隔一轮（epoch）就显示一次损失 （cost）
 
    #创建一个AGN自编码器的实例，同时将噪声的系数scale设为0.01
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784,
                                                  n_hidden= 200,
                                                  transfer_function= tf.nn.softplus,
                                                  optimizer = tf.train.AdamOptimizer(learning_rate=0.001),
                                                  scale=  0.01)
    #下面开始训练过程，在每一轮(epoch)循环开始时，我们将平均损失设为0，并计算总共需要的batch数
    #可以通过调整batch_size,epoch数，优化器，自编码器的隐含层数，隐含节点数等，来尝试获得更低的cost
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int (n_samples/batch_size)
    for i in range(total_batch):
        batch_xs= get_random_block_from_data(X_train,batch_size)
 
        cost= autoencoder.partial_fit(batch_xs)    # def partial_fit(self, X):     #函数partial_fit做的就是用一个batch数据进行训练并返回当前的损失cost。
 
        avg_cost += cost/n_samples*batch_size    
 
    if epoch % display_step ==0:
        print("Epoch:",'%04d'%(epoch +1),"cost=",
            "{:.9f}".format(avg_cost))
        
        #最后，对训练完的模型进行性能测试，评价指标是平方误差
    print("Total cost:" + str(autoencoder.calc_total_cost(X_test)))    #def calc_total_cost(self, X):
 
 
 
                                