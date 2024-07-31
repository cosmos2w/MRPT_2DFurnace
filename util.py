
import numpy as np 
import torch 
import pickle
import os 
from constant import DataSplit 

def get_2d_position():
	filepath = os.path.join(os.path.dirname(__file__), 'Boiler/data_split/data_split_240.pic')
	with open(filepath, 'rb') as fp:
		data_split = pickle.load(fp)
		Y_train = data_split.Y_train
		# print(data_split.U_train.shape)
		return Y_train[[0], :, :]

if __name__ == '__main__':
	Y_2d = get_2d_position() 
	print(Y_2d.shape)