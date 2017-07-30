import os
import gzip
import pickle
import numpy as np


def _print_info(name, data_set):
    x, y = data_set
    print("""{}
  X::
    shape:{}
    min:{} mean:{:5.2f} max:{:5.2f}
  Y::
    shape:{}
    min:{} mean:{:5.2f} max:{}
    """.format(name,
               x.shape, x.min(), x.mean(), x.max(),
               y.shape, y.min(), y.mean(), y.max(),))
def _load_mnist():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(data_dir, "scenes4.pkl")
    f = open(data_file,'rb')
    up = pickle.Unpickler(f)

    training_x, training_y, testing_x, testing_y = up.load()
    f.close()
    training_x= training_x.astype('float32')
    training_y=training_y.astype('float32')
    testing_x= testing_x.astype('float32')
    testing_y=testing_y.astype('float32')
    
    return training_x, training_y, testing_x, testing_y



training_x, training_y, testing_x, testing_y = _load_mnist()


if __name__ == '__main__':
    _print_info("Training Data Set:", (training_x, training_y))
    _print_info("Test Data Set:", (testing_x, testing_y))
