import numpy as np
from scipy.special import boxcox,inv_boxcox
import pickle
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import keras as keras
import keras.callbacks as cb
from keras import Model
from tensorflow.keras.models import load_model
from Model_Unet_3D import *
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from CustomMetricsLosses import *
from scipy.interpolate import interpn,griddata
import argparse
import os
from random import sample
from scipy.signal import resample

# usa gpu con più memoria libera
import GPUtil

#GPU = tf.config.experimental.list_physical_devices('GPU')
#GPU = str(GPUtil.getFirstAvailable(order='memory')[0])
GPU = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
print('GPU selected:', str(GPU))

# crea sessione tensorflow
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import clear_session

config=tf.compat.v1.ConfigProto()
session = tf.compat.v1.Session(config=config)

#inizializza variabili globali
num_x_points = 64
num_y_points = 16
num_dimensions = 3
x = np.arange(0,num_x_points,1).tolist() # x-axis
y = np.arange(0,num_y_points,1).tolist() # y-axis

# Reset Keras Session
def reset_keras(sess):
	clear_session()
	sess.close()

def bc_clip(tensor, lam):
	bc_tensor = boxcox(tensor, lam)
	bc_min = bc_tensor.min()
	bc_max = bc_tensor.max()
	bc_tensor = (bc_tensor - bc_min) / (bc_max - bc_min)  
	return (bc_tensor, bc_min, bc_max)

def inv_bc_clip(tensor, mi, mx, lam):
    bc_tensor = tensor * (mx - mi) + mi
    bc_tensor = inv_boxcox(bc_tensor, lam)
    return bc_tensor

def invertible_clipping(in_content, mi, mx, p):
	"""
    :param in_content: data to be processed
    :param mi: min value for sigmoid function
    :param mx: max value for sigmoid function
    :param p: exponent for the power function
    :return: normalized soft clipped image
    """
	image_clip = np.zeros(in_content.shape)
    #Forward Clipping
	i,j,z = np.where(in_content < mi)
	image_clip[i, j, z] = 1e-4 * in_content[i, j, z] + (1 - 1e-4) * mi
	i,j,z = np.where(in_content > mx)
	image_clip[i, j, z] = 1e-4 * in_content[i, j, z] + (1 - 1e-4) * mx
	i,j,z = np.where((in_content >= mi) & (in_content <= mx))
	image_clip[i, j, z] = in_content[i, j, z]

	#Forward Normalization
	#real_mx = 1e-4 *1 + (1 - 1e-4) * mx
	#image_clip_norm = image_clip/real_mx
	image_clip_norm = 2 * (image_clip - mi) / (mx - mi) - 1

	#Forward Power
	image_clip_pow = np.power(np.abs(image_clip_norm), p)
	return image_clip_pow

def normalize(in_content):
	in_content = np.abs(in_content)
	max_el = in_content.max()
	in_content_norm = in_content/max_el
	return in_content_norm

def prepareDataset(datapath,init,end,ds_axis,num_freqs,down_factor,snr_dB,downsampling,tensors_per_file):

	tensors = []
	zero_lines_idxs = []
	counter_array = np.arange(init,end+1,1)

	for count in counter_array:

		with open(datapath+str(count), 'rb') as data:
			dati = pickle.load(data)

		tens_init_freq = np.arange(0,1024,num_freqs)
		#tens_init_freq = np.arange(0,1024,int(num_freqs/2))

		print('')
		print('Preparing '+datapath+str(count))
		print('')

		list_tensors = sample(np.arange(0,len(dati),1).tolist(),k=tensors_per_file)
		#print(list_tensors)

		perc=10

		for step,tens_idx in enumerate(list_tensors):
		#for step in range(len(dati)):
			percentage= round((step/len(dati))*100,0)
			if percentage==perc:
				print('Percentage: '+str(perc)+'%')
				perc = perc+10
			if step==len(dati)-1:
				print('Percentage: 100%')

			'''print(' Tensor : '+str(step))
			print('')'''

			for i,init_freq in enumerate(tens_init_freq):

				if init_freq>1024-num_freqs:
					break

				end_freq = init_freq+num_freqs

				'''print('   init_freq = '+str(init_freq))
				print('   end_freq = '+str(end_freq))
				print('')'''

				freq = np.arange(init_freq,end_freq,1)

				target_tens = np.array(dati[tens_idx][5][:,:,init_freq:end_freq])
				input_tens = np.array(dati[tens_idx][5][:,:,init_freq:end_freq])

				if snr_dB>0:
					if step==0 and i==0:
						print('')
						print('Adding '+str(snr_dB)+' dB noise to input tensors')
						print('')
					power = np.mean(input_tens ** 2)
					var = power / (10 ** (snr_dB / 10))
					noise = np.random.normal(0, np.sqrt(var), np.shape(input_tens))
					input_tens = np.abs(input_tens+noise)

				if ds_axis=='x':
					if downsampling=='random':
						sampled_list = sample(x,k=int(num_x_points*(1-1/down_factor)))
						sampled_list.sort()
						i=0
						for idx in x:
							if i==len(sampled_list):
								break
							elif idx==sampled_list[i]:
								input_tens[:,idx,:]=np.zeros((num_y_points,num_freqs))
								i=i+1
					elif downsampling=='regular':
						sampled_list = x[1::down_factor]
						for idx in x:
							if idx%down_factor!=0:
								input_tens[:,idx,:]=np.zeros((num_y_points,num_freqs))

				elif ds_axis=='y':
					if downsampling=='random':
						sampled_list = sample(y,k=int(num_y_points*(1-1/down_factor)))
						sampled_list.sort()
						i=0
						for idy in y:
							if i==len(sampled_list):
								break
							elif idy==sampled_list[i]:
								input_tens[idy,:,:]=np.zeros((num_x_points,num_freqs))
								i=i+1
					elif downsampling=='regular':
						sampled_list = y[1::down_factor]
						for idy in y:
							if idy%down_factor!=0:
								input_tens[idy,:,:]=np.zeros((num_x_points,num_freqs))

				elif ds_axis=='xy':
					if downsampling=='random':
						x_sampled_list = sample(x,k=int(num_x_points*(1-1/down_factor)))
						x_sampled_list.sort()
						if down_factor==4:
							y_sampled_list = sample(y,k=int(num_y_points*(1-2/down_factor)))
						else:
							y_sampled_list = sample(y,k=int(num_y_points*(1-1/down_factor)))
						y_sampled_list.sort()
						sampled_list = x_sampled_list+y_sampled_list

						i=0
						for idx in x:
							if i==num_x_points*(1-1/down_factor):
								break
							elif idx==sampled_list[i]:
								input_tens[:,idx,:]=np.zeros((num_y_points,num_freqs))
								i=i+1
						for idy in y:
							if i==len(sampled_list):
								break
							elif idy==sampled_list[i]:
								input_tens[idy,:,:]=np.zeros((num_x_points,num_freqs))
								i=i+1
					elif downsampling=='regular':
						if down_factor==4:
							sampled_list = x[1::down_factor]+y[1::int(down_factor/2)]
						else:
							sampled_list = x[1::down_factor]+y[1::down_factor]

						for idx in x:
							if idx%down_factor!=0:
								input_tens[:,idx,:]=np.zeros((num_y_points,num_freqs))
						for idy in y:
							if down_factor==4:
								if idy%(down_factor/2)!=0:
									input_tens[idy,:,:]=np.zeros((num_x_points,num_freqs))
							else:
								if idy%down_factor!=0:
									input_tens[idy,:,:]=np.zeros((num_x_points,num_freqs))

				zero_lines_idxs.append(sampled_list)
				tensors.append((input_tens,target_tens))

				'''plt.figure(figsize=(12, 5))
				plt.subplot(121), plt.title('Original Histogram')
				b, bins, patches = plt.hist(target_tens.ravel(), 100)
				plt.subplot(122), plt.title('Histogram (pow)')
				plt.hist(target_tens_eq.ravel(), 100)
				plt.show()

				fig,(ax1,ax2) = plt.subplots(1, 2)
				fig1 = ax1.imshow(input_tens[7,:,:],extent=[init_freq,end_freq,num_x_points,0], cmap='bone', aspect='auto')
				fig.colorbar(fig1,ax=ax1)
				ax1.set_xlabel('Frequency [Hz]'), ax1.set_ylabel('X [m]'), ax1.set_title('Input xf image')
				ax1.grid(None)
				fig2 = ax2.imshow(target_tens[7,:,:], extent=[init_freq,end_freq,num_x_points,0],cmap='bone', aspect='auto')
				fig.colorbar(fig2,ax=ax2)
				ax2.set_xlabel('Frequency [Hz]'), ax2.set_ylabel('X [m]'), ax2.set_title('Target xf image')
				ax2.grid(None)
				plt.show()

				fig,(ax1,ax2) = plt.subplots(1, 2)
				fig1 = ax1.imshow(input_tens[:,33,:],extent=[init_freq,end_freq,num_y_points,0], cmap='bone', aspect='auto')
				fig.colorbar(fig1,ax=ax1)
				ax1.set_xlabel('Frequency [Hz]'), ax1.set_ylabel('Y [m]'), ax1.set_title('Input yf image')
				ax1.grid(None)
				fig2 = ax2.imshow(target_tens[:,33,:], extent=[init_freq,end_freq,num_y_points,0], cmap='bone', aspect='auto')
				fig.colorbar(fig2,ax=ax2)
				ax2.set_xlabel('Frequency [Hz]'), ax2.set_ylabel('Y [m]'), ax2.set_title('Target yf image')
				ax2.grid(None)
				plt.show()

				plt.subplot(121), plt.title('Input xy image')
				plt.imshow(input_tens[:,:,120], cmap='bone', aspect='auto'), plt.colorbar()
				plt.xlabel('X [m]'), plt.ylabel('Y [m]')
				plt.grid(None)
				plt.subplot(122), plt.title('Target xy image')
				plt.imshow(target_tens[:,:,120], cmap='bone', aspect='auto'), plt.colorbar()
				plt.xlabel('X [m]'), plt.ylabel('Y [m]')
				plt.grid(None)
				plt.show()

				plt.figure()
				plt.subplot(121), plt.title('Missing FRF')
				plt.plot(freq,input_tens[7,33,:])
				plt.xlabel('Frequency [Hz]')
				plt.subplot(122), plt.title('Target FRF')
				plt.plot(freq,target_tens[7,33,:])
				plt.xlabel('Frequency [Hz]')
				plt.show()'''
	print('')
	print('Dataset ready to be splitted --> '+str(len(tensors))+' tensors')
	print('')
	return tensors,zero_lines_idxs

def splitDataset(dataset,zero_lines_idxs,batch_size,num_freqs,pw):
	### DIVIDING THE DATASET INTO TRAIN, VALIDATION AND TEST SETS
	shuffler = np.random.permutation(len(dataset))

	dataset = np.array(dataset, dtype='float32')
	zero_lines_idxs = np.array(zero_lines_idxs)

	dataset = dataset[shuffler]
	zero_lines_idxs = zero_lines_idxs[shuffler]

	train, val, test = np.split(dataset,[int(.8 * len(dataset)),int(.9 * len(dataset))])
	train_zli, val_zli, test_zli = np.split(zero_lines_idxs,[int(.8 * len(zero_lines_idxs)),int(.9 * len(zero_lines_idxs))])

	X_train = []
	Y_train = []

	print("Preparing training set ")
	train_samples = len(train)-len(train)%batch_size
	for idx in range(train_samples):

		input_tens = normalize(train[idx][0])
		target_tens = normalize(train[idx][1])

		X_train.append(input_tens)
		Y_train.append(target_tens)	

	X_train = np.array(X_train)
	Y_train = np.array(Y_train)

	X_train = X_train.reshape(len(X_train),num_y_points,num_x_points,num_freqs,1)
	Y_train = Y_train.reshape(len(Y_train),num_y_points,num_x_points,num_freqs,1)

	print("Training set ready --> dimensions: "+str(np.shape(X_train)))
	print('')

	X_val = []
	Y_val = []

	print("Preparing validation set ")
	val_samples = len(val)-len(val)%batch_size
	for idx in range(val_samples):

		input_tens = normalize(val[idx][0])
		target_tens = normalize(val[idx][1])

		X_val.append(input_tens)
		Y_val.append(target_tens)

	X_val = np.array(X_val)
	Y_val = np.array(Y_val)

	X_val = X_val.reshape(len(X_val),num_y_points,num_x_points,num_freqs,1)
	Y_val = Y_val.reshape(len(Y_val),num_y_points,num_x_points,num_freqs,1)

	print("Validation set ready --> dimensions: "+str(np.shape(X_val)))
	print('')

	X_test = []
	Y_test = []
	low_dnmcs_tens_idxs = []

	print("Preparing test set ")
	test_samples = len(test)-len(test)%batch_size
	for idx in range(test_samples):

		input_tens = normalize(test[idx][0])
		target_tens = normalize(test[idx][1])

		X_test.append(input_tens)
		Y_test.append(target_tens)

		'''print(test_zli[idx])
		fig,(ax1,ax2) = plt.subplots(1, 2)
		fig1 = ax1.imshow(input_tens[7,:,:], cmap='bone', aspect='auto')
		fig.colorbar(fig1,ax=ax1)
		ax1.set_xlabel('Frequency [Hz]'), ax1.set_ylabel('X [m]'), ax1.set_title('Input xf image')
		ax1.grid(None)
		fig2 = ax2.imshow(target_tens[7,:,:], cmap='bone', aspect='auto')
		fig.colorbar(fig2,ax=ax2)
		ax2.set_xlabel('Frequency [Hz]'), ax2.set_ylabel('X [m]'), ax2.set_title('Target xf image')
		ax2.grid(None)
		plt.show()'''

	X_test = np.array(X_test)
	Y_test = np.array(Y_test)

	X_test = X_test.reshape(len(X_test),num_y_points,num_x_points,num_freqs,1)
	Y_test = Y_test.reshape(len(Y_test),num_y_points,num_x_points,num_freqs,1)

	print("Test set ready --> dimensions: "+str(np.shape(X_test)))
	print('')

	return X_train,Y_train,X_val,Y_val,X_test,Y_test,test_zli,low_dnmcs_tens_idxs

def main():

	parser = argparse.ArgumentParser()

	parser.add_argument('--datapath',type=str,required=False,
		default='./dataset/DatasetFiles/Dataset_xyf/Dataset_xyf_')
	parser.add_argument('--ds_axis',type=str,required=False,default='x')
	parser.add_argument('--num_freqs',type=int,required=False,default=512)
	parser.add_argument('--train_session',type=int,required=False,default=1)
	parser.add_argument('--init_dataset_idx',type=int,required=False,default=1)
	parser.add_argument('--final_dataset_idx',type=int,required=False,default=1)
	parser.add_argument('--outdir',type=str,required=False,
		default='./ModelCheckpoint/3D/super_res_3D.hdf5')
	parser.add_argument('--outdir_trainhistory',type=str,required=False,
		default='./ModelCheckpoint/3D/th_3D')
	parser.add_argument('--outdir_metrics',type=str,required=False,
		default='./Metrics/3D/Metrics_behaviour_3D')
	parser.add_argument('--outdir_plots',type=str,required=False,
		default='./Plots/3D/Plot_3D')
	parser.add_argument('--epochs',type=int,required=False,default=1)
	parser.add_argument('--lr',type=float,required=False,default=0.0004)
	parser.add_argument('--batch_size',type=int,required=False,default=1)
	parser.add_argument('--snr',type=int,required=False,default=0)
	parser.add_argument('--downsampling',type=str,required=False,
		default='regular')
	parser.add_argument('--down_factor',type=int,required=False,default=2)
	parser.add_argument('--pow',type=float,required=False,default=.7)
	parser.add_argument('--tensors_per_file',type=int,required=False,default=30)

	args = parser.parse_args()

	dataset,zero_lines_idxs = prepareDataset(args.datapath,args.init_dataset_idx,args.final_dataset_idx,args.ds_axis,
		args.num_freqs,args.down_factor,args.snr,args.downsampling,args.tensors_per_file)

	X_train,Y_train,X_val,Y_val,X_test,Y_test,test_zli,low_dnmcs_tens_idxs = splitDataset(dataset,zero_lines_idxs,args.batch_size,args.num_freqs,args.pow)

	print("Compiling model")

	uNet = uNet3(args.num_freqs)
	uNet.summary()

	opt = keras.optimizers.Adam(learning_rate=args.lr)

	uNet.compile(loss=mask_mse_3D(args.batch_size, args.num_freqs), optimizer=opt, metrics=[NMSE, ncc])

	callback = [ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.2),
				ModelCheckpoint(
                	filepath=args.outdir,
                	monitor='val_loss', verbose=1, save_best_only=True)]

	print("Model compiled. Training model")

	### TRAINING THE U-net

	history = uNet.fit(X_train, Y_train, epochs=args.epochs, verbose=1, callbacks=callback, validation_data=(X_val, Y_val), batch_size=args.batch_size)

	with open(args.outdir_trainhistory, 'wb') as file_pi:
		pickle.dump(history.history, file_pi)

	### TESTING THE U-net

	print("Testing")

	score = uNet.evaluate(X_test, Y_test, verbose=1, batch_size=args.batch_size)
	probs = uNet.predict(X_test, verbose=1, batch_size=args.batch_size)


	print("Custom metrics and plot results")
	### CALCULATE THE CUSTOM METRICS AND SAVE THEM USING PICKLE

	x_ds = np.arange(0,num_x_points,args.down_factor) # down-sampled x-axis vector
	y_ds = np.arange(0,num_y_points,args.down_factor) # down-sampled y-axis vector
	freq = np.arange(0,args.num_freqs,1) # frequency axis

	if args.ds_axis=='xy' and args.down_factor==4:
		y_ds = np.arange(0,num_y_points,int(args.down_factor/2))

	list_metrics = []
	list_plots = []

	for idx in range(len(Y_test)):
		'''print('Tensor n°: '+str(idx))
		print('')'''

		down =  X_test[idx][:,:,:,0]
		ground_truth = Y_test[idx][:,:,:,0]
		prediction = probs[idx][:,:,:,0]

		'''fig,(ax1,ax2,ax3,ax4) = plt.subplots(1, 4)
		fig1 = ax1.imshow(down[7,:,:], cmap='bone', aspect='auto')
		fig.colorbar(fig1,ax=ax1)
		ax1.set_xlabel('Frequency [Hz]'), ax1.set_ylabel('X [m]'), ax1.set_title('Input xf image')
		ax1.grid(None)
		fig2 = ax2.imshow(ground_truth[7,:,:], cmap='bone', aspect='auto')
		fig.colorbar(fig2,ax=ax2)
		ax2.set_xlabel('Frequency [Hz]'), ax2.set_ylabel('X [m]'), ax2.set_title('Target xf image')
		ax2.grid(None)
		fig3 = ax3.plot(down[7,7,:]), ax3.set_xlabel('Frequency [Hz]')
		fig4 = ax4.plot(ground_truth[7,7,:]), ax4.set_xlabel('Frequency [Hz]')

		fig,(ax1,ax2) = plt.subplots(1, 2)
		fig1 = ax1.imshow(down[:,:,100], cmap='bone', aspect='auto')
		fig.colorbar(fig1,ax=ax1)
		ax1.set_xlabel('Frequency [Hz]'), ax1.set_ylabel('X [m]'), ax1.set_title('Input xy image')
		ax1.grid(None)
		fig2 = ax2.imshow(ground_truth[:,:,100], cmap='bone', aspect='auto')
		fig.colorbar(fig2,ax=ax2)
		ax2.set_xlabel('Frequency [Hz]'), ax2.set_ylabel('X [m]'), ax2.set_title('Target xy image')
		ax2.grid(None)
		plt.show()'''

		nmse1 = nmse(ground_truth,prediction)
		ncc1 = NCC(ground_truth,prediction)

		zero_row_idxs = test_zli[idx]

		grid_y, grid_x, grid_freq = np.mgrid[ 0:num_y_points:1, 0:num_x_points:1, 0:args.num_freqs:1]

		if args.ds_axis=='x':
			if args.downsampling=='random':
				ds_tensor = np.zeros((num_y_points,int(num_x_points/args.down_factor),args.num_freqs))

				zero_row_idx=0
				count=0

				for i in x:
					if i!=zero_row_idxs[zero_row_idx]:
						ds_tensor[:,count,:] = down[:,i,:]
						count=count+1
					else:
						zero_row_idx=zero_row_idx+1
						if zero_row_idx==len(zero_row_idxs):
							break
			
			elif args.downsampling=='regular':
				ds_tensor = ground_truth[:,::args.down_factor,:]

			'''num_points = len(x_ds)*len(y)*len(freq)
			points = np.zeros((num_points,num_dimensions))
			values = np.zeros((num_points))
			count = 0

			for z in range(len(y)):
				for j in range (len(x_ds)):
					for i in range(len(freq)):
						points[count,:] = (y[z], x_ds[j], freq[i])
						values[count] = ds_tensor[z,j,i]
						count = count+1

			interp_tensor = griddata(points, values, (grid_y, grid_x, grid_freq), method='nearest')'''

			interp_tensor= resample(ds_tensor,num_x_points,axis=1)

			'''plt.subplot(141), plt.title('Target')
			plt.imshow(ground_truth[:,:,120], cmap='bone', aspect='auto')
			plt.xlabel('X [m]'), plt.ylabel('Y [m]')
			plt.grid(None)
			plt.subplot(142), plt.title('U-net input')
			plt.imshow(down[:,:,120], cmap='bone', aspect='auto')
			plt.xlabel('X [m]'), plt.ylabel('Y [m]')
			plt.grid(None)
			plt.subplot(143), plt.title('Interp input')
			plt.imshow(ds_tensor[:,:,120], cmap='bone', aspect='auto')
			plt.xlabel('X [m]'), plt.ylabel('Y [m]')
			plt.grid(None)
			plt.subplot(144), plt.title('Interp input')
			plt.imshow(interp_tensor[:,:,120], cmap='bone', aspect='auto'), plt.colorbar()
			plt.xlabel('X [m]'), plt.ylabel('Y [m]')
			plt.grid(None)
			plt.show()'''

		elif args.ds_axis=='xy':
			if args.downsampling=='random':
				x_ds_tensor = np.zeros((num_y_points,int(num_x_points/args.down_factor),args.num_freqs))

				zero_row_idx=0
				count=0

				for j in x:
					if j!=zero_row_idxs[zero_row_idx]:
						x_ds_tensor[:,count,:] = down[:,j,:]
						count=count+1
					else:
						zero_row_idx=zero_row_idx+1
						if zero_row_idx==num_x_points*(1-1/args.down_factor):
							break
				count=0

				if args.down_factor==4:
					ds_tensor = np.zeros((int(2*num_y_points/args.down_factor),int(num_x_points/args.down_factor),args.num_freqs))
				else:
					ds_tensor = np.zeros((int(num_y_points/args.down_factor),int(num_x_points/args.down_factor),args.num_freqs))

				for i in y:
					if i!=zero_row_idxs[zero_row_idx]:
						ds_tensor[count,:,:] = x_ds_tensor[i,:,:]
						count=count+1
					else:
						zero_row_idx=zero_row_idx+1
						if zero_row_idx==len(zero_row_idxs):
							break
			elif args.downsampling=='regular':
				if args.down_factor==4:
					ds_tensor = ground_truth[::int(args.down_factor/2),::args.down_factor,:]
				else:
					ds_tensor = ground_truth[::args.down_factor,::args.down_factor,:]

			'''num_points = len(x_ds)*len(y_ds)*len(freq)
			points = np.zeros((num_points,num_dimensions))
			values = np.zeros((num_points))
			count = 0

			for z in range(len(y_ds)):
				for j in range (len(x_ds)):
					for i in range(len(freq)):
						points[count,:] = (y_ds[z], x_ds[j], freq[i])
						values[count] = ds_tensor[z,j,i]
						count = count+1

			interp_tensor = griddata(points, values, (grid_y, grid_x, grid_freq), method='nearest')'''

			interp_y = resample(ds_tensor,num_y_points,axis=0)
			interp_tensor = resample(interp_y,num_x_points,axis=1)

			'''plt.subplot(141), plt.title('Target')
			plt.imshow(ground_truth[:,:,120], cmap='bone', aspect='auto')
			plt.xlabel('X [m]'), plt.ylabel('Y [m]')
			plt.grid(None)
			plt.subplot(142), plt.title('U-net input')
			plt.imshow(down[:,:,120], cmap='bone', aspect='auto')
			plt.xlabel('X [m]'), plt.ylabel('Y [m]')
			plt.grid(None)
			plt.subplot(143), plt.title('Interp input')
			plt.imshow(ds_tensor[:,:,120], cmap='bone', aspect='auto')
			plt.xlabel('X [m]'), plt.ylabel('Y [m]')
			plt.grid(None)
			plt.subplot(144), plt.title('Interp output')
			plt.imshow(interp_tensor[:,:,120], cmap='bone', aspect='auto'), plt.colorbar()
			plt.xlabel('X [m]'), plt.ylabel('Y [m]')
			plt.grid(None)
			plt.show()'''

		nmse2 = nmse(ground_truth,interp_tensor)
		ncc2 = NCC(ground_truth,interp_tensor)
		
		if args.snr>0:
			list_metrics.append((nmse1,ncc1))
			list_plots.append((down,ground_truth,prediction))
		else:
			list_metrics.append((nmse1,nmse2,ncc1,ncc2))
			if idx<50:
				list_plots.append((down,ground_truth,prediction,interp_tensor))

	with open(args.outdir_metrics,'wb') as output:
		pickle.dump(list_metrics,output)

	with open(args.outdir_plots,'wb') as output:
		pickle.dump(list_plots,output)

	metrics = np.array(list_metrics)
	metrics = metrics.transpose()

	mean_nmse_net = round(np.mean(metrics[0]),2)
	mean_nmse_interp = round(np.mean(metrics[1]),2)

	mean_ncc_net = round(np.mean(metrics[2]),2)
	mean_ncc_interp = round(np.mean(metrics[3]),2)

	std_nmse_net = round(np.std(metrics[0]),2)
	std_nmse_interp = round(np.std(metrics[1]),2)

	std_ncc_net = round(np.std(metrics[2]),2)
	std_ncc_interp = round(np.std(metrics[3]),2)

	print('================== NMSE ==================')
	print('')
	print('U-net :  Mean = ' + str(mean_nmse_net)+' dB || Std = '+str(std_nmse_net)+' dB')
	print('Interp : Mean = ' + str(mean_nmse_interp)+' dB || Std = '+str(std_nmse_interp)+' dB')

	print('')

	print('================== NCC ==================')
	print('')
	print('U-net :  Mean = ' + str(mean_ncc_net)+' || Std = '+str(std_ncc_net))
	print('Interp : Mean = ' + str(mean_ncc_interp)+' || Std = '+str(std_ncc_interp))

	print('')

	reset_keras(session)

if __name__=='__main__':
	main()