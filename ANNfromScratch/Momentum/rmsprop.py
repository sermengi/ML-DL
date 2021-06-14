from utils import process_data, yreshape, cost, error_rate
from mlp import forward, derivative_w1, derivative_w2, derivative_b1, derivative_b2
import numpy as np

Xtrain, Xtest, Ytrain, Ytest = process_data()
Ytrain_ind = yreshape(Ytrain)
Ytest_ind = yreshape(Ytest)
N,D = Xtrain.shape
K = Ytest_ind.shape[1]
print("Number of unique class: ",K)

batch_size = 500
num_of_batches = int(np.ceil(N/batch_size))
M = 300 # number of neurons in hidden layer
W1 = np.random.randn(D,M) / 28 # hidden layer weights
b1 = np.zeros(M)
W2 = np.random.randn(M,K) / np.sqrt(M) # weights for output neuron
b2 = np.zeros(K)
lr = 0.001
reg = 0.01
print_res = 40
max_iter = 20
dw2, dw1, db2, db1 = 0, 0, 0, 0
cache_gw2,cache_gb2, cache_gw1, cache_gb1 = 1, 1, 1, 1
decay = 0.999
eps = 1e-8
losses_batch = []
errors_batch = []

for i in range(max_iter):
    for j in range(num_of_batches):
        Xtmp = Xtrain[j*batch_size:(j+1)*batch_size,:]
        Ytmp = Ytrain_ind[j*batch_size:(j+1)*batch_size,:]
        py_batch, Z = forward(Xtmp,W1,W2,b1,b2)
        gw2, gb2 = derivative_w2(Z,Ytmp,py_batch)+reg*W2, derivative_b2(Ytmp,py_batch)+reg*b2
        gw1, gb1 = derivative_w1(Xtmp,Z,Ytmp,py_batch,W2)+reg*W1, derivative_b1(Z,Ytmp,py_batch,W2)+reg*b1

        cache_gw2 = decay*cache_gw2 + (1-decay)*gw2**2
        cache_gb2 = decay*cache_gb2 + (1-decay)*gb2**2
        cache_gw1 = decay*cache_gw1 + (1-decay)*gw1**2
        cache_gb1 = decay*cache_gb1 + (1-decay)*gb1**2
        W2 -= lr*gw2 / np.sqrt(eps+cache_gw2)
        b2 -= lr*gb2 / np.sqrt(eps+cache_gb2)
        W1 -= lr*gw1 / np.sqrt(eps+cache_gw1)
        b1 -= lr*gb1 / np.sqrt(eps+cache_gb1)

        if (j+1) % print_res == 0:
            p_y, _ = forward(Xtest,W1,W2,b1,b2)
            l = cost(Ytest_ind,p_y)
            losses_batch.append(l)
            e = error_rate(Ytest_ind,p_y)
            errors_batch.append(e)
            print(f"iteration i: {i} batch: {j+1} Cost: {l:.6f} Error rate: {e:.6f}")