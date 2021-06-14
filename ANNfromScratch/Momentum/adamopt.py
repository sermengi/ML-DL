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
W1 = np.random.randn(D,M) / np.sqrt(D) # hidden layer weights
b1 = np.zeros(M)
W2 = np.random.randn(M,K) / np.sqrt(M) # weights for output neuron
b2 = np.zeros(K)
lr = 0.001
reg = 0.01
print_res = 40
max_iter = 20
m_gw2, m_gb2, m_gw1, m_gb1 = 0, 0, 0, 0
v_gw2, v_gb2, v_gw1, v_gb1 = 0, 0, 0, 0
beta1 = 0.9 # decay for 1st momentum
beta2 = 0.999 # decay for 2nd momentum
eps = 1e-8
t = 1
losses_batch = []
errors_batch = []

for i in range(max_iter):
    for j in range(num_of_batches):
        Xtmp = Xtrain[j*batch_size:(j+1)*batch_size,:]
        Ytmp = Ytrain_ind[j*batch_size:(j+1)*batch_size,:]
        py_batch, Z = forward(Xtmp,W1,W2,b1,b2)
        gw2, gb2 = derivative_w2(Z,Ytmp,py_batch)+reg*W2, derivative_b2(Ytmp,py_batch)+reg*b2
        gw1, gb1 = derivative_w1(Xtmp,Z,Ytmp,py_batch,W2)+reg*W1, derivative_b1(Z,Ytmp,py_batch,W2)+reg*b1

        # momentum
        m_gw2 = beta1*m_gw2 + (1-beta1)*gw2
        m_gb2 = beta1*m_gb2 + (1-beta1)*gb2
        m_gw1 = beta1*m_gw1 + (1-beta1)*gw1
        m_gb1 = beta1*m_gb1 + (1-beta1)*gb1

        # velocity
        v_gw2 = beta2*v_gw2 + (1-beta2)*gw2*gw2
        v_gb2 = beta2*v_gb2 + (1-beta2)*gb2*gb2
        v_gw1 = beta2*v_gw1 + (1-beta2)*gw1*gw1
        v_gb1 = beta2*v_gb1 + (1-beta2)*gb1*gb1

        # bias correction
        corr1 = 1 - beta1**t
        m_hat_gw2 = m_gw2 / corr1
        m_hat_gb2 = m_gb2 / corr1
        m_hat_gw1 = m_gb1 / corr1
        m_hat_gb1 = m_gb1 / corr1

        corr2 = 1 - beta2**t
        v_hat_gw2 = v_gw2 / corr2
        v_hat_gb2 = v_gb2 / corr2
        v_hat_gw1 = v_gw1 / corr2
        v_hat_gb1 = v_gb1 / corr2

        t += 1

        W2 -= lr*m_hat_gw2 / (np.sqrt(v_hat_gw2) + eps)
        b2 -= lr*m_hat_gb2 / (np.sqrt(v_hat_gb2) + eps)
        W1 -= lr*m_hat_gw1 / (np.sqrt(v_hat_gw1) + eps)
        b1 -= lr*m_hat_gb1 / (np.sqrt(v_hat_gb1) + eps)

        if (j+1) % print_res == 0:
            p_y, _ = forward(Xtest,W1,W2,b1,b2)
            l = cost(Ytest_ind,p_y)
            losses_batch.append(l)
            e = error_rate(Ytest_ind,p_y)
            errors_batch.append(e)
            print(f"iteration i: {i} batch: {j+1} Cost: {l:.6f} Error rate: {e:.6f}")