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
lr = 0.00004
reg = 0.01
print_res = 40
max_iter = 20
dw2, dw1, db2, db1 = 0, 0, 0, 0
losses_batch = []
errors_batch = []

# Without momentum
for i in range(max_iter):
    for j in range(num_of_batches):
        Xtmp = Xtrain[j*batch_size:(j+1)*batch_size,:]
        Ytmp = Ytrain_ind[j*batch_size:(j+1)*batch_size,:]
        py_batch, Z = forward(Xtmp,W1,W2,b1,b2)
        gw2, gb2 = derivative_w2(Z,Ytmp,py_batch), derivative_b2(Ytmp,py_batch)
        gw1, gb1 = derivative_w1(Xtmp,Z,Ytmp,py_batch,W2), derivative_b1(Z,Ytmp,py_batch,W2)

        W2 -= lr*(gw2 + reg*W2)
        b2 -= lr*(gb2 + reg*b2)
        W1 -= lr*(gw1 + reg*W1)
        b1 -= lr*(gb1 + reg*b1)

        if (j+1) % print_res == 0:
            p_y, _ = forward(Xtest,W1,W2,b1,b2)
            l = cost(Ytest_ind,p_y)
            losses_batch.append(l)
            e = error_rate(Ytest_ind,p_y)
            errors_batch.append(e)
            print(f"iteration i: {i} batch: {j+1} Cost: {l:.6f} Error rate: {e:.6f}")

# dw1 = 0
# dw2 = 0
# db1 = 0
# db2 = 0
# mu = 0.9
#
# # With momentum method
# for i in range(max_iter):
#     for j in range(num_of_batches):
#         Xtmp = Xtrain[j*batch_size:(j+1)*batch_size,:]
#         Ytmp = Ytrain_ind[j*batch_size:(j+1)*batch_size,:]
#         py_batch, Z = forward(Xtmp,W1,W2,b1,b2)
#         gw2, gb2 = derivative_w2(Z,Ytmp,py_batch)+reg*W2, derivative_b2(Ytmp,py_batch)+reg*b2
#         gw1, gb1 = derivative_w1(Xtmp,Z,Ytmp,py_batch,W2)+reg*W1, derivative_b1(Z,Ytmp,py_batch,W2)+reg*b1
#
#         # update velocities
#         dw1 = mu*dw1 - lr*gw1
#         dw2 = mu*dw2 - lr*gw2
#         db1 = mu*db1 - lr*gb1
#         db2 = mu*db2 - lr*gb2
#
#         # update with momentum
#         W2 += dw2
#         b2 += db2
#         W1 += dw1
#         b1 += db1
#
#         if (j+1) % print_res == 0:
#             p_y, _ = forward(Xtest,W1,W2,b1,b2)
#             l = cost(Ytest_ind,p_y)
#             losses_batch.append(l)
#             e = error_rate(Ytest_ind,p_y)
#             errors_batch.append(e)
#             print(f"iteration i: {i} batch: {j+1} Cost: {l:.6f} Error rate: {e:.6f}")

# db1 = 0
# dw1 = 0
# db1 = 0
# db2 = 0
# mu = 0.9
#
# # With Nesterov momentum
# for i in range(max_iter):
#     for j in range(num_of_batches):
#         Xtmp = Xtrain[j*batch_size:(j+1)*batch_size,:]
#         Ytmp = Ytrain_ind[j*batch_size:(j+1)*batch_size,:]
#         py_batch, Z = forward(Xtmp,W1,W2,b1,b2)
#         gw2, gb2 = derivative_w2(Z,Ytmp,py_batch)+reg*W2, derivative_b2(Ytmp,py_batch)+reg*b2
#         gw1, gb1 = derivative_w1(Xtmp,Z,Ytmp,py_batch,W2)+reg*W1, derivative_b1(Z,Ytmp,py_batch,W2)+reg*b1
#
#         # update velocities
#         dw2 = mu*dw2 - lr*gw2
#         dw1 = mu*dw1 - lr*gw1
#         db1 = mu*db1 - lr*gb1
#         db2 = mu*db2 - lr*gb2
#
#         # update with Nesterov momentum
#
#         W2 += mu*dw2 - lr*gw2
#         b2 += mu*db2 - lr*gb2
#         W1 += mu*dw1 - lr*gw1
#         b1 += mu*db1 - lr*gb1
#
#         if (j+1) % print_res == 0:
#             p_y, _ = forward(Xtest,W1,W2,b1,b2)
#             l = cost(Ytest_ind,p_y)
#             losses_batch.append(l)
#             e = error_rate(Ytest_ind,p_y)
#             errors_batch.append(e)
#             print(f"iteration i: {i} batch: {j+1} Cost: {l:.6f} Error rate: {e:.6f}")







