# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os
import tarfile
import pickle
import numpy as np
from PIL import Image

url_base = 'https://www.cs.toronto.edu/~kriz/'
file_name = 'cifar-10-python.tar.gz'

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/cifar10.pkl"

def _download(file_name):
    file_path = dataset_dir + "/" + file_name

    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ...")
    urllib.request.urlretrieve(url_base + file_name,file_path)
    print("Done")

def download_cifar10():
    _download(file_name)

def _unzip():
    file_path = dataset_dir + "/" + file_name

    print("Unziping " + file_name + " ...")
    tar = tarfile.open(file_path)
    tar.extractall()
    tar.close()
    print("End")

def unzip_cifar10():
    _unzip()

#def img_show(img):
#    img = img.reshape(3,32,32).transpose(1,2,0)
#    pil_img = Image.fromarray(np.uint8(img))
#    pil_img.show()

def _unpickle(filename):
    with open(filename,"rb") as fp:
        data = pickle.load(fp,encoding="latin-1")
    return data

def _load_data():
    #X_train = None
    dataset = {}
    y_train = []
    path = dataset_dir + "/cifar-10-batches-py"
    
    for i in range(1,6):
        data_dic = _unpickle(path + "/data_batch_{}".format(i))
        if i == 1:
            #X_train = data_dic['data']
            dataset['train_img'] = data_dic['data']
        else :
            #X_train = np.vstack((X_train,data_dic['data']))
            dataset['train_data'] = np.vstack((dataset['train_img'],data_dic['data']))
        y_train += data_dic['labels']

    test_data_dic = _unpickle(path + "/test_batch")
    dataset['test_img'] = test_data_dic['data']
    y_test = test_data_dic['labels']
    dataset['train_label'] = np.array(y_train)
    dataset['test_label'] = np.array(y_test)

    return dataset

def init_cifar10():
    download_cifar10()
    unzip_cifar10()
    dataset = _load_data()
    print("Creating pickle file ...")
    with open(save_file,'wb') as f:
        pickle.dump(dataset,f,-1)
    print("Done!")

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

    if not os.path.exists(save_file):
        init_cifar10()

    with open(save_file,'rb') as f:
        dataset = pickle.load(f)

    #dataset = _load_data()
    
    if one_hot_label:
        y_train = _change_one_hot_label(y_train)
        y_test = _change_one_hot_label(y_test)

    return (dataset['train_img'],dataset['train_label']),(dataset['test_img'],dataset['test_label'])

if __name__ == '__main__':
    init_cifar10()
