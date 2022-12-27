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

def normalize(in_content):
    in_content = np.abs(in_content)
    max_el = in_content.max()
    in_content_norm = in_content/max_el
    return in_content_norm

def prepareTestSet(init,end,x_down_factor,num_x_points,num_freqs,imgs_x_file,downsampling):

    zero_lines_idxs = []
    X_test = []
    Y_test = []

    x = np.arange(0,num_x_points,1).tolist()

    counter_array = np.arange(init,end+1,1)

    datapath = '../dataset/Dataset_complex/dataset_complex_'

    for count in counter_array:
        with open(datapath+str(count), 'rb') as data:
            dati = pickle.load(data)

        print('')
        print('Preparing '+datapath+str(count))
        print('')

        list_images = sample(np.arange(0,len(dati),1).tolist(),k=imgs_x_file)

        for step,img_idx in enumerate(list_images):
            target_img = np.array(dati[img_idx][6])
            input_img = np.zeros((num_x_points,num_freqs))

            if downsampling=='regular':
                sampled_list = x[::int(1/x_down_factor)]
            else:
                sampled_list = sample(x,k=int(num_x_points*x_down_factor))
                sampled_list.sort()

            i=0
            for idx in x:
                if i==int(num_x_points*(x_down_factor)):
                    break
                elif idx==sampled_list[i]:
                    input_img[idx,:]=target_img[idx,:]
                    i=i+1                    

            if np.mean(target_img**2)>1e-10:
                zero_lines_idxs.append(sampled_list)
                X_test.append(normalize(input_img))
                Y_test.append(normalize(target_img))

                '''plt.subplot(121), plt.title('Input xy image')
                plt.imshow(normalize(input_img), cmap='bone', aspect='auto'), plt.colorbar()
                plt.xlabel('X [m]'), plt.ylabel('Y [m]')
                #plt.grid(None)
                plt.subplot(122), plt.title('Target xy image')
                plt.imshow(normalize(target_img), cmap='bone', aspect='auto'), plt.colorbar()
                plt.xlabel('X [m]'), plt.ylabel('Y [m]')
                #plt.grid(None)
                plt.show()'''

    print('')
    print('Test set composed by --> '+str(len(X_test))+' images')
    print('')

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    ### ADD CHANNEL DIMENSION
    X_test = X_test.reshape(len(X_test),num_x_points,num_freqs,1)
    Y_test = Y_test.reshape(len(X_test),num_x_points,num_freqs,1)
    return X_test,Y_test, zero_lines_idxs

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--init',type=int,required=False,default=10)
    parser.add_argument('--end',type=int,required=False,default=)
    parser.add_argument('--num_x_points',type=int,required=False,default=64)
    parser.add_argument('--num_freqs',type=int,required=False,default=1024)
    parser.add_argument('--lr',type=float,required=False,default=0.0004)
    parser.add_argument('--imgs_x_file',type=int,required=False,default=30)
    parser.add_argument('--interp_func',type=str,required=False,default='Bicubic')
    parser.add_argument('--downsampling',type=str,required=False,default='regular')
    parser.add_argument('--method',type=str,required=False,default='Unet')

    args = parser.parse_args()

    if args.downsampling=='regular':
        x_down_factors = [0.125,0.25,0.5]
    else:
        x_down_factors = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    for count in range(len(x_down_factors)):

        x_down_factor = x_down_factors[count]

        X_test,Y_test,zero_row_idxs = prepareTestSet(args.init,args.end,x_down_factor,args.num_x_points,args.num_freqs,args.imgs_x_file,args.downsampling)

        print('')
        print('X_test dimensions: '+ str(np.shape(X_test)))
        print('')
        print('Y_test dimensions: '+ str(np.shape(Y_test)))

        num_x_points = args.num_x_points
        num_freqs = args.num_freqs
        x = np.arange(0,num_x_points,1).tolist() # x-axis
        freq = np.arange(0,num_freqs,1).tolist()  #frequency axis
        ds_matrix_points = int(num_x_points*x_down_factor)*num_freqs

        if args.method=='interp':

            list_metrics_interp = []
            list_plots_interp = []

            for idx in range(len(Y_test)):
                #print('Image n° '+str(idx))
                target_img = Y_test[idx][:,:,0]
                ds_img = np.zeros((int(num_x_points*x_down_factor),num_freqs))
                zero_row_idx=0
                count=0

                x_ds = np.linspace(0,num_x_points,int(num_x_points*x_down_factor)).tolist()

                for i in x:
                    if i==zero_row_idxs[idx][zero_row_idx]:
                        ds_img[count,:] = target_img[i,:]
                        count=count+1
                        zero_row_idx=zero_row_idx+1
                    if zero_row_idx==int(num_x_points*(x_down_factor)):
                        break

                if args.interp_func=='Rbf':
                    # Rbf Interpolator
                    if idx==0:
                        print('')
                        print('Testing Rbf Interpolation on '+args.downsampling+' down images with '+str(x_down_factor*100)+' %% of original data')
                        print('')
                    values = np.zeros((ds_matrix_points))
                    rows = np.zeros((ds_matrix_points))
                    columns = np.zeros((ds_matrix_points))
                    count=0

                    for i in range(int(num_x_points*x_down_factor)):
                        for j in range(num_freqs):
                            values[count] = ds_img[i,j]
                            rows[count] = i
                            columns[count] = j
                            count = count+1

                    rbf = Rbf(columns, rows, values, function='cubic')  # radial basis function interpolator instance
                    XI, YI = np.meshgrid(freq, x/args.down_factor)
                    interp_img = rbf(XI, YI)

                elif args.interp_func=='Fourier':
                    # ResSample Interpolator:
                    if idx==0:
                        print('')
                        print('Testing Fourier-based Interpolation on '+args.downsampling+' down images with '+str(x_down_factor*100)+' %% of original data')
                        print('')
                    interp_img= resample(ds_img,num_x_points,axis=0)

                elif args.interp_func=='Bicubic':
                    # Bicubic Interpolator:
                    if idx==0:
                        print('')
                        print('Testing Bicubic Interpolation on '+args.downsampling+' down images with '+str(x_down_factor*100)+' %% of original data')
                        print('')
                    interp = interp2d(freq, x_ds, ds_img, kind='cubic')
                    interp_img = interp(freq,x)

                else:
                    print('ERRORE! CONTROLLA PARAMETRO interp_func')
                    exit()

                nmse_interp = nmse(target_img,interp_img)
                ncc_interp = NCC(target_img,interp_img)
                list_metrics_interp.append((nmse_interp,ncc_interp))
                list_plots_interp.append((target_img,interp_img))

            print("Calculating Interp NMSE and NCC for reconstructions")

            if args.downsampling=='regular':
                with open('./Metrics/2D/xf/64/Paper/Interps/Regular/metrics_2D_'+args.interp_func+'_interp_downtest'+str(x_down_factor*100)+'%%data','wb') as output:
                    pickle.dump(list_metrics_interp,output)
            else:
                with open('./Metrics/2D/xf/64/Paper/Interps/Random/metrics_2D_'+args.interp_func+'_interp_downtest'+str(x_down_factor*100)+'%%data','wb') as output:
                    pickle.dump(list_metrics_interp,output)

        elif args.method=='Unet':

            opt = keras.optimizers.Adam(learning_rate=args.lr)

            down_factors = np.array([2,4,8])

            for step in range(len(down_factors)):
                print('')
                print('Testing 2D U-net trained on '+args.downsampling+' down '+str(down_factors[step])+' images, on images with '+str(x_down_factor*100)+' %% of original data')
                print('')

                if args.downsampling=='random':
                    uNet = load_model('./ModelCheckpoint/2D/xf/64/Random/down'+str(down_factors[step])+'/super_res_xf_random_down'+str(down_factors[step])+'.h5', 
                        custom_objects = {'loss': mask_mse(batch_size=1,num_x_points=args.num_x_points),'NMSE': NMSE, 'ncc': ncc, 'ReflectionPadding2D':ReflectionPadding2D})

                elif args.downsampling=='regular':
                    uNet = load_model('./ModelCheckpoint/2D/xf/64/Regular/down'+str(down_factors[step])+'/super_res_xf_down'+str(down_factors[step])+'.h5', 
                        custom_objects = {'loss': mask_mse(batch_size=1,num_x_points=args.num_x_points),'NMSE': NMSE, 'ncc': ncc, 'ReflectionPadding2D':ReflectionPadding2D})

                uNet.compile(loss=mask_mse(batch_size=1,num_x_points=args.num_x_points), optimizer=opt, metrics=[NMSE, ncc])

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

                if args.downsampling=='regular':
                    with open('./Metrics/2D/xf/64/Paper/Unets/Regular/metrics_2D_Unet_downtrain'+str(down_factors[step])+'_downtest'+str(x_down_factor*100)+'%%data','wb') as output:
                        pickle.dump(list_metrics_Unet,output)
                else:
                    with open('./Metrics/2D/xf/64/Paper/Unets/Random/metrics_2D_Unet_downtrain'+str(down_factors[step])+'_downtest'+str(x_down_factor*100)+'%%data','wb') as output:
                        pickle.dump(list_metrics_Unet,output)

if __name__=='__main__':
    main()
