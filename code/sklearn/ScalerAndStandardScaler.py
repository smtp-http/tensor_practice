from sklearn import preprocessing 
import numpy as np 
X = np.array([[ 1., -1.,  2.],[ 2.,  0.,  0.],[ 0.,  1., -1.]]) 


'''
    scale
'''
X_scaled = preprocessing.scale(X)

print(X_scaled)
#array([[ 0.        , -1.22474487,  1.33630621],
#       [ 1.22474487,  0.        , -0.26726124],
#       [-1.22474487,  1.22474487, -1.06904497]])


print(X_scaled.mean(axis=0))
#array([ 0.,  0.,  0.])

print(X_scaled.std(axis=0))
#array([ 1.,  1.,  1.])

'''
    StandardScaler
'''

scaler = preprocessing.StandardScaler().fit(X)
print(scaler.transform(X))


print(scaler.mean_)
print(scaler.var_)


'''
    MinMaxScaler
'''

minmaxscaler = preprocessing.MinMaxScaler().fit(X)

print(minmaxscaler.transform(X))

print(minmaxscaler.scale_)

print(minmaxscaler.min_)



'''
    Normalizer
'''

normalizer = preprocessing.Normalizer().fit(X)
print(normalizer.transform(X))
