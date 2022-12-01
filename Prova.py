import numpy as np
import pickle
import keras
import keras.callbacks as cb
from keras import Model
from tensorflow.keras.models import load_model
from CustomMetricsLosses import *
import argparse
from scipy.io import loadmat
import matplotlib.pyplot as plt
from random import sample
from scipy.interpolate import Rbf,interp2d
from scipy.signal import resample
from Model_Unet import *
import GPUtil
import os

GPU = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
print('GPU selected:', str(GPU))

import tensorflow as tf
from tensorflow.compat.v1.keras.backend import clear_session

config=tf.compat.v1.ConfigProto()
session = tf.compat.v1.Session(config=config)

num_x_points = 64
num_freqs = 1024

def normalize(in_content):
	in_content = np.abs(in_content)
	max_el = in_content.max()
	in_content_norm = in_content/max_el
	return in_content_norm


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath',type=str,required=False,default='./Plots/2D/xf/64/Regular/down8/Plot_xf_down8')
    parser.add_argument('--outdir_metrics',type=str,required=False,
    	default='./Metrics/2D/xf/64/Paper/Unets/Regular/metrics_2D_Unet_downtrain8_downtest_12.5%%data')
    parser.add_argument('--outdir_plots',type=str,required=False,
    	default='./Plots/2D/xf/64/Paper/Unets/Regular/Plots_2D_Unet_downtrain8_downtest_12.5%%data')
    parser.add_argument('--modeldir',type=str,required=False,
    	default='./ModelCheckpoint/2D/xf/64/Regular/down8/super_res_xf_down8.h5')


    args = parser.parse_args()

    with open(args.datapath, 'rb') as data:
    	dati = pickle.load(data)

    Y_test = []
    X_test = []

    for idx in range(len(dati)):
    	X_test.append(dati[idx][0]) 
    	Y_test.append(dati[idx][1])

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_test = X_test.reshape(len(X_test),num_x_points,num_freqs,1)
    Y_test = Y_test.reshape(len(Y_test),num_x_points,num_freqs,1)

    opt = keras.optimizers.Adam(learning_rate=0.0004)

    uNet = load_model(args.modeldir, custom_objects = {'loss': mask_mse(batch_size=1,num_x_points=num_x_points),'NMSE': NMSE, 'ncc': ncc, 'ReflectionPadding2D':ReflectionPadding2D})
    uNet.compile(loss=mask_mse(batch_size=1,num_x_points=num_x_points), optimizer=opt, metrics=[NMSE, ncc])

    score = uNet.evaluate(X_test, Y_test, verbose=1, batch_size=1)
    probs = uNet.predict(X_test, verbose=1, batch_size=1)

    print("Calculating U-net NMSE and NCC for predictions")

    list_metrics_Unet = []
    list_plots_Unet = []

    for idx in range(len(Y_test)):
    	down = X_test[idx][:,:,0]
    	ground_truth = Y_test[idx][:,:,0]
    	prediction = probs[idx][:,:,0]

    	nmse_Unet = nmse(ground_truth,prediction)
    	ncc_Unet = NCC(ground_truth,prediction)
    	list_metrics_Unet.append((nmse_Unet,ncc_Unet))
    	list_plots_Unet.append((ground_truth, down, prediction))

    with open(args.outdir_metrics,'wb') as output:
    	pickle.dump(list_metrics_Unet,output)

    '''with open(args.outdir_plots,'wb') as output:
    	pickle.dump(list_plots_Unet,output)'''

if __name__=='__main__':
	main()