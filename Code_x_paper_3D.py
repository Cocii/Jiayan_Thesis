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
from scipy.interpolate import Rbf,griddata,RegularGridInterpolator
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

#inizializza variabili globali
num_x_points = 64
num_y_points = 16
num_freqs = 512
num_dimensions = 3
x = np.arange(0,num_x_points,1).tolist() # x-axis
y = np.arange(0,num_y_points,1).tolist() # y-axis

def normalize(in_content):
	in_content = np.abs(in_content)
	max_el = in_content.max()
	in_content_norm = in_content/max_el
	return in_content_norm

def prepareTestSet(init,end,x_down_factor,y_down_factor,num_freqs,tens_x_file,downsampling):
    zero_lines_idxs = []
    X_test = []
    Y_test = []
    x = np.arange(0,num_x_points,1).tolist()
    counter_array = np.arange(init,end+1,1)
    datapath = './dataset/DatasetFiles/Dataset_xyf/Dataset_xyf_'

    for count in counter_array:
        with open(datapath+str(count), 'rb') as data:
            dati = pickle.load(data)
        tens_init_freq = np.arange(0,1024,num_freqs)

        print('')
        print('Preparing '+datapath+str(count))
        print('')

        list_tens = sample(np.arange(0,len(dati),1).tolist(),k=tens_x_file)

        for step,tens_idx in enumerate(list_tens):

            for i,init_freq in enumerate(tens_init_freq):

                if init_freq>1024-num_freqs:
                    break
                end_freq = init_freq+num_freqs
                freq = np.arange(init_freq,end_freq,1)

                target_tens = np.array(dati[tens_idx][5][:,:,init_freq:end_freq])
                input_tens = np.array(dati[tens_idx][5][:,:,init_freq:end_freq])

                if downsampling=='regular':
                    x_sampled_list = x[::int(1/x_down_factor)]
                    y_sampled_list = y[::int(1/y_down_factor)]
                else:
                    x_sampled_list = sample(x,k=int(num_x_points*(x_down_factor)))
                    x_sampled_list.sort()
                    y_sampled_list = sample(y,k=int(num_y_points*(y_down_factor)))
                    y_sampled_list.sort()

                sampled_list = x_sampled_list+y_sampled_list

                if downsampling=='random':
                    i=0
                    for xx in x:
                        if i==len(x_sampled_list):
                            input_tens[:,xx:num_x_points,:]=np.zeros((num_y_points,num_x_points-xx,num_freqs))
                            break
                        elif xx!=sampled_list[i]:
                            input_tens[:,xx,:]=np.zeros((num_y_points,num_freqs))
                        else:
                            i=i+1
                    for yy in y:
                        if i==len(sampled_list):
                            input_tens[yy:num_y_points,:,:]=np.zeros((num_y_points-yy,num_x_points,num_freqs))
                            break
                        elif yy!=sampled_list[i]:
                            input_tens[yy,:,:]=np.zeros((num_x_points,num_freqs))
                        else:
                            i=i+1

                elif downsampling=='regular':
                    for xx in x:
                        if xx%int(1/x_down_factor)!=0:
                            input_tens[:,xx,:]=np.zeros((num_y_points,num_freqs))
                    for yy in y:
                        if yy%int(1/y_down_factor)!=0:
                            input_tens[yy,:,:]=np.zeros((num_x_points,num_freqs))

                zero_lines_idxs.append(sampled_list)
                X_test.append(normalize(input_tens))
                Y_test.append(normalize(target_tens))

                '''plt.subplot(121), plt.title('Input xy image')
                plt.imshow(input_tens[:,:,120], cmap='bone', aspect='auto'), plt.colorbar()
                plt.xlabel('X [m]'), plt.ylabel('Y [m]')
                #plt.grid(None)
                plt.subplot(122), plt.title('Target xy image')
                plt.imshow(target_tens[:,:,120], cmap='bone', aspect='auto'), plt.colorbar()
                plt.xlabel('X [m]'), plt.ylabel('Y [m]')
                #plt.grid(None)
                plt.show()'''


    print('')
    print('Test set composed by --> '+str(len(X_test))+' tensors')
    print('')

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
    ### ADD CHANNEL DIMENSION
    X_test = X_test.reshape(len(X_test),num_y_points,num_x_points,num_freqs,1)
    Y_test = Y_test.reshape(len(X_test),num_y_points,num_x_points,num_freqs,1)
    return X_test,Y_test, zero_lines_idxs

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--init',type=int,required=False,default=1)
    parser.add_argument('--end',type=int,required=False,default=1)
    parser.add_argument('--lr',type=float,required=False,default=0.0004)
    parser.add_argument('--tens_x_file',type=int,required=False,default=1)
    parser.add_argument('--downsampling',type=str,required=False,default='regular')
    parser.add_argument('--method',type=str,required=False,default='interp')
    parser.add_argument('--interp_func',type=str,required=False,default='linear')

    args = parser.parse_args()

    if args.downsampling=='regular':
        x_down_factors = np.array([0.25, 0.5, 0.5])
        y_down_factors = np.array([0.5, 0.5, 1])
    else:
        x_down_factors = np.array([0.2,0.4,0.5,0.5,0.5,0.6,0.7,0.8,0.9])
        y_down_factors = np.array([0.5,0.5,0.6,0.8,1,1,1,1,1])

    for count in range(len(x_down_factors)):

        x_down_factor = x_down_factors[count]
        y_down_factor = y_down_factors[count]

        X_test,Y_test,zero_row_idxs = prepareTestSet(args.init,args.end,x_down_factor,y_down_factor,num_freqs,args.tens_x_file,args.downsampling)

        print('')
        print('X_test dimensions: '+ str(np.shape(X_test)))
        print('')
        print('Y_test dimensions: '+ str(np.shape(Y_test)))

        list_metrics_interp = []
        list_plots_interp = []
        freq = np.arange(0,num_freqs,1).tolist()  #frequency axis
        grid_y, grid_x, grid_freq = np.mgrid[ 0:num_y_points:1, 0:num_x_points:1, 0:num_freqs:1]
        x_ds = np.linspace(0,num_x_points,int(num_x_points*(x_down_factor))).tolist()
        y_ds = np.linspace(0,num_y_points,int(num_y_points*(y_down_factor))).tolist()
        ds_matrix_points = len(x_ds)*len(y_ds)*len(freq)

        if args.method=='interp':

            for idx in range(len(Y_test)):
                #print('Tensor n°: '+str(idx))

                target_tens = Y_test[idx][:,:,:,0]
                down = X_test[idx][:,:,:,0]

                x_ds_tens = np.zeros((num_y_points,int(num_x_points*x_down_factor),num_freqs))

                zero_row_idx=0
                count=0

                for j in x:
                    if j==zero_row_idxs[idx][zero_row_idx]:
                        x_ds_tens[:,count,:] = down[:,j,:]
                        count=count+1
                        zero_row_idx=zero_row_idx+1
                    if zero_row_idx==int(num_x_points*(x_down_factor)):
                        break
                    
                count=0
                ds_tens = np.zeros((int(num_y_points*y_down_factor),int(num_x_points*x_down_factor),num_freqs))

                for i in y:
                    if i==zero_row_idxs[idx][zero_row_idx]:
                        ds_tens[count,:,:] = x_ds_tens[i,:,:]
                        count=count+1
                        zero_row_idx=zero_row_idx+1
                    if zero_row_idx==len(zero_row_idxs[idx]):
                        break

                if args.interp_func=='Fourier' and idx==0:
                    # RESAMPLE --> FOURIER-BASED INTERPOLATOR
                    if idx==0:
                        print('')
                        print('Testing Fourier-based Interpolation on '+args.downsampling+' down tensors with '+str(x_down_factor*y_down_factor*100)+' %% of original data')
                        print('')
                    interp_y= resample(ds_tens,num_y_points,axis=0)
                    interp_tens= resample(interp_y,num_x_points,axis=1)

                elif args.interp_func=='NN' and idx==0:
                    # NN INTERPOLATOR
                    if idx==0:
                        print('')
                        print('Testing Nearest-Neighbour Interpolation on '+args.downsampling+' down tensors with '+str(x_down_factor*y_down_factor*100)+' %% of original data')
                        print('')
                    num_points = len(x_ds)*len(y_ds)*len(freq)
                    points = np.zeros((num_points,num_dimensions))
                    values = np.zeros((num_points))
                    count = 0

                    for i in range(len(y_ds)):
                        for j in range (len(x_ds)):
                            for k in range(len(freq)):
                                points[count,:] = (y_ds[i], x_ds[j], freq[k])
                                values[count] = ds_tens[i,j,k]
                                count = count+1

                    interp_tens = griddata(points, values, (grid_y, grid_x, grid_freq), method='nearest')

                elif args.interp_func=='Linear':
                    # LINEAR INTERPOLATOR
                    if idx==0:
                        print('')
                        print('Testing Linear Interpolation on '+args.downsampling+' down tensors with '+str(x_down_factor*y_down_factor*100)+' %% of original data')
                        print('')
                    interp = RegularGridInterpolator((y_ds, x_ds, freq), ds_tens, method='linear')
                    interp_tens = interp((grid_y, grid_x, grid_freq))

                elif args.interp_func=='Rbf':
                    # Rbf INTERPOLATOR
                    if idx==0:
                        print('')
                        print('Testing Rbf Interpolation on '+args.downsampling+' down tensors with '+str(x_down_factor*y_down_factor*100)+' %% of original data')
                        print('')
                    values = np.zeros((ds_matrix_points))
                    rows = np.zeros((ds_matrix_points))
                    columns = np.zeros((ds_matrix_points))
                    freqs_values = np.zeros((ds_matrix_points))
                    count=0

                    for i in range(int(num_y_points*y_down_factor)):
                        for j in range(int(num_x_points*x_down_factor)):
                            for k in range(args.num_freqs):
                                values[count] = ds_tens[i,j,k]
                                rows[count] = i
                                columns[count] = j
                                freqs_values[count] = k
                                count = count+1

                    rbf = Rbf(freqs_values, columns, rows, values, function='cubic')  # radial basis function interpolator instance
                    XI, YI, ZI = np.meshgrid(freq, x*x_down_factor,y*y_down_factor)
                    interp_tens = rbf(XI, YI, ZI)

                else:
                    print('ERRORE! CONTROLLA PARAMETRO interp_func')
                    exit()
                
                nmse_interp = nmse(target_tens,interp_tens)
                ncc_interp = NCC(target_tens,interp_tens)
                list_metrics_interp.append((nmse_interp,ncc_interp))

            print("Calculating Interp NMSE and NCC for reconstructions")

            if args.downsampling=='regular':
                with open('./Metrics/3D/Paper/Interps/Regular/metrics_3D_'+args.interp_func+'_interp_downtest'+str(x_down_factor*y_down_factor*100)+'%%data','wb') as output:
                    pickle.dump(list_metrics_interp,output)
            else:
                with open('./Metrics/3D/Paper/Interps/Random/metrics_3D_'+args.interp_func+'_interp_downtest'+str(x_down_factor*y_down_factor*100)+'%%data','wb') as output:
                    pickle.dump(list_metrics_interp,output)

        elif args.method=='Unet':

            opt = keras.optimizers.Adam(learning_rate=args.lr)

            down_factors = np.array([2,4,8])
            #down_factors = np.array([4,8])

            for step in range(len(down_factors)):

                print('')
                print('Testing 3D U-net trained on '+args.downsampling+' down '+str(down_factors[step])+' tensors, on tensors with '+str(x_down_factor*y_down_factor*100)+' %% of original data')
                print('')

                if args.downsampling=='regular':
                    if step==0:
                        uNet = load_model('./ModelCheckpoint/3D/down_x/Regular/super_res_3D_down'+str(down_factors[step])+'.h5', 
                            custom_objects = {'loss': mask_mse_3D(batch_size=1, num_freqs=num_freqs),'NMSE': NMSE, 'ncc': ncc})
                    else:
                        uNet = load_model('./ModelCheckpoint/3D/down_xy/Regular/super_res_3D_down'+str(down_factors[step])+'.h5', 
                            custom_objects = {'loss': mask_mse_3D(batch_size=1, num_freqs=num_freqs),'NMSE': NMSE, 'ncc': ncc})

                    uNet.compile(loss=mask_mse_3D(batch_size=1, num_freqs=args.num_freqs), optimizer=opt, metrics=[NMSE, ncc])

                elif args.downsampling=='random':
                    if step==0:
                        uNet = load_model('./ModelCheckpoint/3D/down_x/Random/super_res_3D_random_down'+str(down_factors[step])+'.h5', 
                            custom_objects = {'mask_mse_3D': mask_mse_3D,'NMSE': NMSE, 'ncc': ncc})
                        uNet.compile(loss=mask_mse_3D, optimizer=opt, metrics=[NMSE, ncc])
                    else:
                        uNet = load_model('./ModelCheckpoint/3D/down_xy/Random/super_res_3D_random_down'+str(down_factors[step])+'.h5', 
                            custom_objects = {'loss': mask_mse_3D(batch_size=1, num_freqs=num_freqs),'NMSE': NMSE, 'ncc': ncc})
                        uNet.compile(loss=mask_mse_3D(batch_size=1, num_freqs=num_freqs), optimizer=opt, metrics=[NMSE, ncc])

                score = uNet.evaluate(X_test, Y_test, verbose=1, batch_size=1)
                probs = uNet.predict(X_test, verbose=1, batch_size=1)

                print("Calculating U-net NMSE and NCC for predictions")

                list_metrics_Unet = []
                list_plots_Unet = []

                for idx in range(len(Y_test)):
                    down = X_test[idx][:,:,:,0]
                    ground_truth = Y_test[idx][:,:,:,0]
                    prediction = probs[idx][:,:,:,0]

                    nmse_Unet = nmse(ground_truth,prediction)
                    ncc_Unet = NCC(ground_truth,prediction)
                    list_metrics_Unet.append((nmse_Unet,ncc_Unet))

                if args.downsampling=='regular':
                    with open('./Metrics/3D/Paper/Unets/Regular/metrics_3D_Unet_downtrain'+str(down_factors[step])+'_downtest'+str(x_down_factor*y_down_factor*100)+'%%data','wb') as output:
                     pickle.dump(list_metrics_Unet,output)
                else:
                    with open('./Metrics/3D/Paper/Unets/Random/metrics_3D_Unet_downtrain'+str(down_factors[step])+'_downtest'+str(x_down_factor*y_down_factor*100)+'%%data','wb') as output:
                     pickle.dump(list_metrics_Unet,output)

if __name__=='__main__':
    main()


