import os
import pickle
import numpy as np
from PIL import Image

def img_show(img):
    img = img.reshape(3,32,32).transpose(1,2,0)
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def unpickle(filename):
    with open(filename,"rb") as fp:
        data = pickle.load(fp,encoding="latin-1")
    return data

def _change_one_hot_label(X):
    T = np.zeros((X.size,10))
    for idx,row in enumerate(T):
        row[X[idx]] = 1
    return T

def load_data(one_hot_label=False):
    """ Read CIFAR-10 Dataset

    Parameters
    ----------
    one_hot_label :
        if one_hot_label is True, this function returns label of ont_hot array.


    Returns
    -------
    (train_image,train_label),(test_image,test_label)
    """
    X_train = None
    y_train = []
    path = os.path.dirname(os.path.abspath(__file__))

    for i in range(1,6):
        data_dic = unpickle(path + "/data_batch_{}".format(i))
        if i == 1:
            X_train = data_dic['data']
        else :
            X_train = np.vstack((X_train,data_dic['data']))
        y_train += data_dic['labels']

    test_data_dic = unpickle(path + "/test_batch")
    X_test = test_data_dic['data']
    y_test = test.data.dic['labels']
    y_train = np.array(y_train)
    y_test = np.array(y_teset)
    
    if one_hot_label:
        y_train = _change_one_hot_label(y_train)
        y_test = _change_one_hot_label(y_test)

    return X_train
