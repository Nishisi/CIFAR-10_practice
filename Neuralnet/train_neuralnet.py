import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.dataset import load_data
from two_layer_net import TwoLayerNet

# Date input
(x_train,t_train),(x_test,t_test) = load_data(normalize=True,one_hot_label=True)

network = TwoLayerNet(input_size=3072,hidden_size=10,output_size=10)

iters_num = 100
train_size = x_train.shape[0]
batch_size = 100
learning_rage = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size,1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 誤差逆伝播法によって勾配を求める
    grad = network.gradient(x_batch,t_batch)
    print(grad)

    for key in ('W1','b1','W2','b2'):
        network.params[key] -= learning_rage * grad[key]
