#coding=utf-8
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def xavier_init(fan_in,fan_out, constant = 1):
    '''Yoshua Bengio指出深度学习模型的权重初始化太小，那么信号将在每层传递缩小而失去作用；
    太大将导致发散.Xavier初始化器就是让权重被初始化得不大不小，正好合适。
    从数学的角度看，Xavier就是满足均值为：0，方差为2/(n_in+n_out)的均匀或高斯分布'''
    low = -constant*np.sqrt(6.0/(fan_in+fan_out))
    high = constant*np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out), minval=low, maxval=high,dtype = tf.float32)
#去噪声自编码class
class AdditiveGaussianNoiseAutoencoder(object):

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        ''' n_input:输入变量数
            n_hidden:隐含层节点数
            transfer_function:隐含层激活函数,默认为softplus
            optimizer:优化器，默认为Adam
            scale:高斯噪声系数，默认为0.1'''
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        #参数初始化使用_initialize_weights()
        network_weights = self._initialize_weights()
        self.weights = network_weights

        #输入x
        self.x = tf.placeholder(tf.float32,[None,self.n_input])

        '''隐藏层hidden,首先输入x加上噪声：self.x+scale*tf.random_normal((n_input,))
            然后tf.matmul上式与隐含层权重w1，
            tf.add加上隐含层biases：b1，
            最后使用self.transfer对结果进行激活函数处理'''
        self.hidden = self.transfer(tf.add(tf.matmul(
                        self.x+scale*tf.random_normal((n_input,)),
                        self.weights['w1']), self.weights['b1']))
        '''输出层重构：reconstruction,不用激活函数
        tf.matmul隐含层输出和输出层权重w2再加上输出层偏置b2'''
        self.reconstruction = tf.add(tf.matmul(self.hidden,
                        self.weights['w2']), self.weights['b2'])

        '''cost:直接使用平方误差即tf.substract计算输出self.reconstruction与self.x之差，
        再使用tf.pow求差的平方'''
        self.cost = 0.5*tf.reduce_sum(tf.pow(tf.subtract(
                            self.reconstruction, self.x), 2.0))


        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        '''w1使用xavier_init函数初始化，传入输入节点数和隐含层节点数，
        它将返回一个比较适合softplus激活函数的权重初始分布'''
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,
                                                    self.n_hidden))
        #b1，w2，b2使用tf.zeros全部为0
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],
                                                  dtype = tf.float32))

        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,
                                                  self.n_input], dtype = tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))

        return all_weights

    def partial_fit(self, X):

        '''trian每一个batch数据并返回当年batch的cost
        Session执行两个计算图的节点，cost和训练过程optimizer,
        输入的feed_dict：输入数据x和噪声系数：scale'''

        cost, opt = self.sess.run((self.cost, self.optimizer),
            feed_dict = {self.x:X, self.scale: self.training_scale})

        return cost


    def calc_total_cost(self, X):
        #计算cost
        return self.sess.run(self.cost, feed_dict = {self.x:X,
            self.scale:self.training_scale
        })

    def transform(self, X):
        #计算抽象的特征，返回隐含层的输出结果
        return self.sess.run(self.hidden, feed_dict = {self.x:X,
            self.scale:self.training_scale
        })

    def generate(self, hidden = None):
        #将高阶抽象特征复原为原始数据
        if hidden is None:
            hidden = bp.random_normal(size = self.weights['b1'])

        return self.sess.run(self.reconstruction, feed_dict = {self.hidden:hidden})


    def reconstruction(self, X):
        '''整体运行一遍复原过程，包括提取高阶特征和用高阶特征复原原始数据
        输入：原数据 输出：复原后的数据'''
        return self.sess.run(self.reconstruction, feed_dict = {self.x:X,
            self.scale:self.training_scale
        })


    def getWeights(self):
        #获取隐含层权重w1
        return self.sess.run(self.weights['w1'])

    def getBiases(self):

        #获取隐含层的偏置系数b1
        return self.sess.run(self.weights['b1'])

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

def standard_scale(X_train, X_test):
    '''对训练、测试data进行标准化处理（让数据变成均值为0,标准差为1的分布）'''
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

def get_random_block_from_data(data, batch_size):
    '''随机获取block数据：取一个0到len(data)-batch_size之间的随机整数
    再以这个随机数作为block的起始位置，然后顺序取batch_size的数据'''
    start_index = np.random.randint(0,len(data)-batch_size)

    return data[start_index:(start_index+batch_size)]

X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1


autoencoder = AdditiveGaussianNoiseAutoencoder(n_input = 784,
                n_hidden = 200,
                transfer_function = tf.nn.softplus,
                optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),
                scale = 0.01
                )

for epoch in range(training_epochs):

    avg_cost = 0.
    total_batch = int(n_samples/batch_size)

    for i in range(total_batch):
        batch_xs =  get_random_block_from_data(X_train, batch_size)


        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost/n_samples*batch_size

    if epoch%display_step == 0:
        print("Epoch:", '%04d'%(epoch+1), "cost = ",
                "{:.9f}".format(avg_cost))

print("Total cost:"+str(autoencoder.calc_total_cost(X_test)))
