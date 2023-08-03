from utils import load_data
import pandas as pd


train_X, train_y = load_data(is_train=True, flatten=False)

print(train_X)