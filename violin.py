import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from random import sample
from scipy.signal import resample
from scipy.interpolate import interp2d,griddata
import argparse
from CustomMetricsLosses import *
import GPUtil
import keras
import keras.callbacks as cb
from keras import Model
from tensorflow.keras.models import load_model
from Model_Unet import *

GPU = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
print('GPU selected:', str(GPU))

import tensorflow as tf
from tensorflow.compat.v1.keras.backend import clear_session

config=tf.compat.v1.ConfigProto()
session = tf.compat.v1.Session(config=config)

num_x_points =64
num_y_points =16
num_freqs =1024
x = np.arange(0,num_x_points,1).tolist()
y = np.arange(0,num_y_points,1).tolist()
grid_y, grid_x, grid_freq = np.mgrid[ 0:num_y_points:1, 0:num_x_points:1, 0:num_freqs/2:1]

def normalize(in_content):
	in_content_abs = np.abs(in_content)
	in_content_norm = in_content_abs/in_content_abs.max()
	return in_content_norm

def prepareDataset(datapath,numdim,x_down_factor,y_down_factor,num_freqs):

	print('Preparing data')

	tensor_dir = os.path.abspath(datapath)
	tensor_data = pd.read_csv(tensor_dir,header=None)
	tensor_data = np.array(tensor_data)

	tensor = np.zeros((num_y_points,num_x_points,num_freqs))

	X_test = []
	Y_test = []

	if numdim==2:
		for idx in x:
			idx_row = idx*num_y_points
			tensor[:,idx,:] = np.flip(np.array(tensor_data[idx_row:idx_row+num_y_points,3:num_freqs+3]),0)

		target = normalize(tensor[13,:,:])
		#target = normalize(np.array(tensor[0:num_x_points,3:num_freqs+3]))
		inpt = np.zeros((num_x_points,num_freqs))

		sampled_list = sample(x,k=int(num_x_points*(1/x_down_factor)))
		sampled_list.sort()

		i=0
		for idx in x:
			if i==num_x_points*(1/x_down_factor):
				break
			elif idx==sampled_list[i]:
				inpt[idx,:]=target[idx,:]
				i=i+1

		X_test.append(inpt)
		Y_test.append(target)

	elif numdim==3:
		target = np.zeros((num_y_points,num_x_points,num_freqs))

		'''for idx in y:
			idx_row = idx*num_x_points
			target[idx,:,:] = np.array(tensor_data[idx_row:idx_row+num_x_points,3:num_freqs+3]) #3D matrix 16*64*1024'''

		for idx in x:
			idx_row = idx*num_y_points
			target[:,idx,:] = np.flip(np.array(tensor_data[idx_row:idx_row+num_y_points,3:num_freqs+3]),0)


		target = normalize(target)
		inpt = target.copy()

		x_sampled_list = sample(x,k=int(num_x_points*(1/x_down_factor)))
		x_sampled_list.sort()
		y_sampled_list = sample(y,k=int(num_y_points*(1/y_down_factor)))
		y_sampled_list.sort()
		sampled_list = x_sampled_list+y_sampled_list

		i=0
		for idx in x:
			if i==int(num_x_points/x_down_factor):
				inpt[:,idx:num_x_points,:] = np.zeros((num_y_points,num_x_points-idx,num_freqs))
				break
			elif idx!=sampled_list[i]:
				inpt[:,idx,:]=np.zeros((num_y_points,num_freqs))
			else:
				i=i+1
		for idy in y:
			if i==len(sampled_list):
				inpt[idy:num_y_points,:,:] = np.zeros((num_y_points-idy,num_x_points,num_freqs))
				break
			elif idy!=sampled_list[i]:
				inpt[idy,:,:]=np.zeros((num_x_points,num_freqs))
			else:
				i=i+1
		X_test.append(inpt[:,:,0:512])
		Y_test.append(target[:,:,0:512])
		X_test.append(inpt[:,:,512:num_freqs])
		Y_test.append(target[:,:,512:num_freqs])


	X_test = np.array(X_test)
	Y_test = np.array(Y_test)

	#print(sampled_list)
	'''fig,(ax1,ax2) = plt.subplots(1, 2)
	ax1.imshow(X_test[1][:,:,346], cmap='Reds', aspect='auto')
	ax1.set_xlabel('Frequency'), ax1.set_ylabel('X'), ax1.set_title('Input xf image')
	#ax1.grid(None)
	ax2.imshow(Y_test[1][:,:,346],cmap='Reds', aspect='auto')
	ax2.set_xlabel('Frequency'), ax2.set_ylabel('X'), ax2.set_title('Target xf image')
	#ax2.grid(None)
	plt.show()
	exit()'''

	return X_test,Y_test,sampled_list


def main():
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--datapath',type=str,required=False,default='./dataset/violin_curved.csv')
	parser.add_argument('--numdim',type=int,required=False,default=3)
	parser.add_argument('--x_down_factor',type=int,required=False,default=2)
	parser.add_argument('--y_down_factor',type=int,required=False,default=2)
	parser.add_argument('--outdir_plots',type=str,required=False,default='./Plots/2D/xf/64/Random/down4/Plot_xf_random_down4_violin')
	parser.add_argument('--lr',type=float,required=False,default=0.0004)

	args = parser.parse_args()

	X_test,Y_test,zero_lines_idxs = prepareDataset(args.datapath,args.numdim,args.x_down_factor,args.y_down_factor,num_freqs)

	print('')
	print('X_test dimensions: '+ str(np.shape(X_test)))
	print('')
	print('Y_test dimensions: '+ str(np.shape(Y_test)))
	print('')

	### TESTING
	print("Testing")

	opt = keras.optimizers.Adam(learning_rate=args.lr)

	if args.numdim==2:
		X_test = X_test.reshape(len(X_test),num_x_points,num_freqs,1)
		Y_test = Y_test.reshape(len(Y_test),num_x_points,num_freqs,1)

		uNet = load_model('./ModelCheckpoint/2D/xf/64/Random/down'+str(args.x_down_factor)+'/super_res_xf_random_down'+str(args.x_down_factor)+'.h5', 
			custom_objects = {'loss': mask_mse(batch_size=1,num_x_points=num_x_points),'NMSE': NMSE, 'ncc': ncc, 'ReflectionPadding2D':ReflectionPadding2D})
		uNet.summary()
		uNet.compile(loss=mask_mse(batch_size=1,num_x_points=num_x_points), optimizer=opt, metrics=[NMSE, ncc])

	elif args.numdim==3:
		X_test = X_test.reshape(len(X_test),num_y_points,num_x_points,512,1)
		Y_test = Y_test.reshape(len(Y_test),num_y_points,num_x_points,512,1)

		uNet = load_model('./ModelCheckpoint/3D/down_xy/Random/super_res_3D_random_down'+str(args.x_down_factor*args.y_down_factor)+'.h5', 
			custom_objects = {'loss': mask_mse_3D(batch_size=1, num_freqs=512),'NMSE': NMSE, 'ncc': ncc})
		uNet.summary()
		uNet.compile(loss=mask_mse_3D(batch_size=1, num_freqs=512), optimizer=opt, metrics=[NMSE, ncc])

	score = uNet.evaluate(X_test, Y_test, verbose=1, batch_size=1)
	probs = uNet.predict(X_test, verbose=1, batch_size=1)

	print("Calculating NMSE and NCC for predictions")

	list_plots = []
	x_ds = np.arange(0,num_x_points,args.x_down_factor)
	y_ds = np.arange(0,num_y_points,args.y_down_factor)

	for idx in range(len(Y_test)):
		if args.numdim==2:
			f = np.arange(0,num_freqs,1).tolist()
			down = X_test[idx][:,:,0]
			ground_truth = Y_test[idx][:,:,0]
			prediction = probs[idx][:,:,0]

			ds_img = np.zeros((int(num_x_points/args.x_down_factor),num_freqs))
			zero_row_idx=0
			count=0

			for i in x:
				if i==zero_lines_idxs[zero_row_idx]:
					ds_img[count,:] = ground_truth[i,:]
					count=count+1
					zero_row_idx=zero_row_idx+1
				if zero_row_idx==int(num_x_points/args.x_down_factor):
					break

			#interp = resample(ds_img,num_x_points,axis=0)
			interp = interp2d(f, x_ds, ds_img, kind='cubic')
			interp = interp(f,x)

			'''plt.figure()
			plt.subplot(141), plt.title('Target')
			plt.imshow(ground_truth, cmap='Reds', aspect='auto')
			plt.xlabel('Frequency [Hz]'), plt.ylabel('X [m]')
			#plt.grid(None)
			plt.subplot(142), plt.title('U-net input')
			plt.imshow(down, cmap='Reds', aspect='auto')
			plt.xlabel('Frequency [Hz]')
			#plt.grid(None)
			plt.subplot(143), plt.title('Interp input')
			plt.imshow(ds_img, cmap='Reds', aspect='auto')
			plt.xlabel('Frequency [Hz]')
			#plt.grid(None)
			plt.subplot(144), plt.title('Interp output')
			plt.imshow(interp, cmap='Reds', aspect='auto'), plt.colorbar()
			plt.xlabel('Frequency [Hz]')
			#plt.grid(None)
			plt.show()
			exit()'''

		elif args.numdim==3:

			f = np.arange(0,num_freqs/2,1).tolist()
			down = X_test[idx][:,:,:,0]
			ground_truth = Y_test[idx][:,:,:,0]
			prediction = probs[idx][:,:,:,0]

			x_ds_tens = np.zeros((num_y_points,int(num_x_points/args.x_down_factor),512))
			zero_row_idx=0
			count=0

			for j in x:
				if j==zero_lines_idxs[zero_row_idx]:
					x_ds_tens[:,count,:] = ground_truth[:,j,:]
					count=count+1
					zero_row_idx=zero_row_idx+1
				if zero_row_idx==int(num_x_points*(args.x_down_factor)):
					break

			count=0
			ds_tens = np.zeros((int(num_y_points/args.y_down_factor),int(num_x_points/args.x_down_factor),512))

			for i in y:
				if i==zero_lines_idxs[zero_row_idx]:
					ds_tens[count,:,:] = x_ds_tens[i,:,:]
					count=count+1
					zero_row_idx=zero_row_idx+1
				if zero_row_idx==len(zero_lines_idxs):
					break

			'''interp_y= resample(ds_tens,num_y_points,axis=0)
			interp= resample(interp_y,num_x_points,axis=1)'''


			num_points = len(x_ds)*len(y)*len(f)
			points = np.zeros((num_points,args.numdim))
			values = np.zeros((num_points))
			count = 0

			for i in range(len(y_ds)):
				for j in range (len(x_ds)):
					for z in range(len(f)):
						points[count,:] = (y_ds[i], x_ds[j], f[z])
						values[count] = ds_tens[i,j,z]
						count = count+1

			interp = griddata(points, values, (grid_y, grid_x, grid_freq), method='nearest')

			'''print(nmse(ground_truth,interp))

			plt.figure()
			plt.subplot(141), plt.title('Target')
			plt.imshow(ground_truth[5,:,:], cmap='Reds', aspect='auto')
			plt.xlabel('X [m]'), plt.ylabel('Y [m]')
			plt.grid(None)
			plt.subplot(142), plt.title('U-net input')
			plt.imshow(down[5,:,:], cmap='Reds', aspect='auto')
			plt.xlabel('X [m]'), plt.ylabel('Y [m]')
			plt.grid(None)
			plt.subplot(143), plt.title('Interp input')
			plt.imshow(ds_tens[5,:,:], cmap='Reds', aspect='auto')
			plt.xlabel('X [m]'), plt.ylabel('Y [m]')
			plt.grid(None)
			plt.subplot(144), plt.title('Interp output')
			plt.imshow(interp[5,:,:], cmap='Reds', aspect='auto'), plt.colorbar()
			plt.xlabel('X [m]'), plt.ylabel('Y [m]')
			plt.grid(None)
			plt.show()
			exit()'''


		nmse_Unet = nmse(ground_truth,prediction)
		ncc_Unet = NCC(ground_truth,prediction)
		nmse_interp = nmse(ground_truth,interp)
		ncc_interp = NCC(ground_truth,interp)

		print('================== NMSE ==================')
		print('')
		print('U-net : ' + str(nmse_Unet)+' dB')
		print('Interp : ' + str(nmse_interp)+' dB ')

		print('')

		print('================== NCC ==================')
		print('')
		print('U-net : ' + str(ncc_Unet))
		print('Interp : ' + str(ncc_interp))

		print('')

		list_plots.append((down,ground_truth,prediction,interp))
		
	with open(args.outdir_plots,'wb') as output:
		pickle.dump(list_plots,output)

if __name__=='__main__':
	main()