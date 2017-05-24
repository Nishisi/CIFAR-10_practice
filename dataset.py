import os
import pickle

def unpickle(filename):
    with open(filename,"rb") as fp:
        data = pickle.load(fo,encoding="latin-1")
    return data

def load_data():
    path = os.path.dirname(os.path.abspath(__file__))
    for i in range(1,6):
        data_dic = unpickle(path + "data_batch_{}".format(i))
    
