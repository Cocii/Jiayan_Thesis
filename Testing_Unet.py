"""

1) Test the U-net on a completely new and unseen dataset, different from that one used for training 
   and validate. This separate code allows to test without performing anytime training before. 
   Each 2D image is normalised by the maximum before testing.

2) Compute and save the costum metrics as: nmse, psnr
   Both for couples:
   - (ground truth,learned image)
   - (ground truth, interpolated image)
   To compare performances of the U-net with respect to a classic interpolator

"""

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
from scipy.interpolate import interp2d
from Model_Unet import *
from scipy.signal import resample

def normalize(in_content):
    in_content_abs = np.abs(in_content)
    in_content_norm = in_content_abs/in_content_abs.max()
    return in_content_norm

def prepareTestSet(datapath,snr_dB,img_type,down_factor,numdim,ds_axis,images,downsampling,num_x_points,num_y_points,num_freqs):
    X_test = []
    Y_test = []

    x = np.arange(0,num_x_points,1).tolist()
    y = np.arange(0,num_y_points,1).tolist()

    if images=='synth':
        with open(datapath, 'rb') as data:
            dati = pickle.load(data)
        dati = np.array(dati)

        ### NORMALIZING ALL THE 2D IMAGES IN THE DATASET

        print('Preparing '+datapath)
        for idx in range(len(dati)):

            inpt = np.array(dati[idx][0])
            target = np.array(dati[idx][1])

            if snr_dB>0:

                    inpt = np.array(dati[idx][1])
                    if idx==0:
                        print('')
                        print('Adding '+str(snr_dB)+' dB noise to input images')
                        print('')

                    power = np.mean(inpt ** 2)
                    var = power / (10 ** (snr_dB / 10))
                    noise = np.random.normal(0, np.sqrt(var), np.shape(inpt))
                    inpt = inpt+noise

                    if numdim==2:
                        if img_type=='xy':
                            if downsampling=='random':
                                if down_factor==2:
                                    x_sampled_list = sample(x,k=int(num_x_points*(1/down_factor)))
                                    
                                else:
                                    if down_factor==4:
                                        x_sampled_list = sample(x,k=int(num_x_points*(2/down_factor)))
                                        y_sampled_list = sample(y,k=int(num_y_points*(2/down_factor)))

                                    else:
                                        x_sampled_list = sample(x,k=int(num_x_points*(2/down_factor)))
                                        y_sampled_list = sample(y,k=int(num_y_points*(4/down_factor)))

                                    y_sampled_list.sort()

                                x_sampled_list.sort()

                                if down_factor==2:
                                    sampled_list = x_sampled_list
                                    i=0
                                    for idx in x:
                                        if i==num_x_points*(1/down_factor):
                                            break
                                        elif idx!=sampled_list[i]:
                                            input_img[:,idx]=np.zeros(num_y_points)
                                        else:
                                            i=i+1
                                else:
                                    sampled_list = x_sampled_list+y_sampled_list
                                    x_down_factor=down_factor/2

                                    i=0

                                    for idx in x:
                                        if i==int(num_x_points*(1/x_down_factor)):
                                            break
                                        elif idx!=sampled_list[i]:
                                            input_img[:,idx]=np.zeros(num_y_points)
                                        else:
                                            i=i+1
                                    for idy in y:
                                        if i==len(sampled_list):
                                            break
                                        elif idy!=sampled_list[i]:
                                            input_img[idy,:]=np.zeros(num_x_points)
                                        else:
                                            i=i+1

                            elif downsampling=='regular':

                                if down_factor==2:
                                    sampled_list = x[::down_factor]

                                    for idx in x:
                                        if idx%down_factor!=0:
                                            input_img[:,idx]=np.zeros(num_y_points)

                                else:
                                    x_down_factor = int(down_factor/2)
                                    if down_factor==4:
                                        y_down_factor = x_down_factor                               
                                        sampled_list = x[::x_down_factor]+y[::y_down_factor]
                                    else:
                                        y_down_factor = int(x_down_factor/2)
                                        sampled_list = x[::x_down_factor]+y[::y_down_factor]

                                    for idx in x:
                                        if idx%x_down_factor!=0:
                                            input_img[:,idx]=np.zeros(num_y_points)

                                    for idy in y:
                                        if idy%y_down_factor==0:
                                            input_img[idy,:]=np.zeros(num_x_points)
                            
                            #print(sampled_list)
                        
                        elif img_type=='yf':
                            if downsampling=='random':
                                sampled_list = sample(y,k=int(num_y_points*(1-1/down_factor)))
                                sampled_list.sort()
                                i=0
                                for idx in y:
                                    if i==len(sampled_list):
                                        break
                                    elif idx==sampled_list[i]:
                                        input_img[idx,:]=np.zeros(num_freqs)
                                        i=i+1
                            elif downsampling=='regular':
                                for idy in y:
                                    if idy%down_factor!=0:
                                        input_img[idy,:]=np.zeros(num_freqs)

                        elif img_type=='xf':
                            if downsampling=='random':
                                sampled_list = sample(x,k=int(num_x_points*(1/down_factor)))
                                sampled_list.sort()

                                i=0
                                for idx in x:
                                    if i==num_x_points*(1/down_factor):
                                        break
                                    elif idx!=sampled_list[i]:
                                        input_img[idx,:]=np.zeros(num_freqs)
                                    else:
                                        i=i+1

                            elif downsampling=='regular':
                                sampled_list = x[::down_factor]

                                for idx in x:
                                    if idx%down_factor!=0:
                                        input_img[idx,:]=np.zeros(num_freqs)

                    elif numdim==3:
                        if ds_axis=='x':
                            if downsampling=='random':
                                sampled_list = sample(x,k=int(num_x_points*(1-1/down_factor)))
                                sampled_list.sort()
                                i=0
                                for idx in x:
                                    if i==len(sampled_list):
                                        break
                                    elif idx==sampled_list[i]:
                                        inpt[:,idx,:]=np.zeros((num_y_points,num_freqs))
                                        i=i+1
                            elif downsampling=='regular':
                                sampled_list = x[1::down_factor]
                                for idx in x:
                                    if idx%down_factor!=0:
                                        inpt[:,idx,:]=np.zeros((num_y_points,num_freqs))
                                        
                        if ds_axis=='xy':
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
                                        inpt[:,idx,:]=np.zeros((num_y_points,num_freqs))
                                        i=i+1
                                for idy in y:
                                    if i==len(sampled_list):
                                        break
                                    elif idy==sampled_list[i]:
                                        inpt[idy,:,:]=np.zeros((num_x_points,num_freqs))
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
                            elif ds_axis=='x':
                                for idx in x:
                                    if idx%down_factor!=0:
                                        inpt[:,idx,:]=np.zeros((num_y_points,num_freqs))
                            elif ds_axis=='y':
                                for idy in y:
                                    if idy%down_factor!=0:
                                        inpt[idy,:,:]=np.zeros((num_x_points,num_freqs))

            X_test.append(normalize(inpt))
            Y_test.append(normalize(target))

    elif images=='real':
        dataset = loadmat(datapath)
        image = np.array(dataset.get('image')).transpose()
        target = np.abs(image[5:37,:])
        inpt = np.zeros((np.shape(target)))

        if downsampling=='random':
            sampled_list = sample(x,k=int(num_x_points*(1/down_factor)))
            sampled_list.sort()

            i=0
            for idx in x:
                if i==num_x_points*(1/down_factor):
                    break
                elif idx==sampled_list[i]:
                    inpt[idx,:]=target[idx,:]
                    i=i+1

        elif downsampling=='regular':
            sampled_list = x[::down_factor]

            for idx in x:
                if idx%down_factor!=0:
                    inpt[idx,:]=np.zeros(num_freqs)

        X_test.append(normalize(inpt))
        Y_test.append(normalize(target))

    '''plt.subplot(121)
    plt.imshow(normalize(inpt), cmap='bone', aspect='auto'), plt.colorbar()
    #plt.grid(None)
    plt.subplot(122)
    plt.imshow(normalize(target), cmap='bone', aspect='auto'), plt.colorbar()
    #plt.grid(None)
    plt.show()'''

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    if numdim==3:
        X_test = X_test.reshape(len(X_test),num_y_points,num_x_points,num_freqs,1)
        Y_test = Y_test.reshape(len(Y_test),num_y_points,num_x_points,num_freqs,1)

    elif images=='real':
        X_test = X_test.reshape(len(X_test),num_x_points,num_freqs,1)
        Y_test = Y_test.reshape(len(Y_test),num_x_points,num_freqs,1)

    return X_test,Y_test, sampled_list

def main():
    parser = argparse.ArgumentParser()

    #dataset parameters
    parser.add_argument('--img_type',type=str,required=False,default='xf')
    parser.add_argument('--batch_size',type=int,required=False,default=1)
    parser.add_argument('--images',type=str,required=False,default='real')
    parser.add_argument('--ds_axis',type=str,required=False,default='x')
    parser.add_argument('--numdim',type=int,required=False,default=2)
    parser.add_argument('--num_x_points',type=int,required=False,default=32)
    parser.add_argument('--num_y_points',type=int,required=False,default=16)
    parser.add_argument('--num_freqs',type=int,required=False,default=1024)
    parser.add_argument('--down_factor',type=int,required=False,default=4)
    parser.add_argument('--downsampling',type=str,required=False,default='random')
    parser.add_argument('--datapath',type=str,required=False,default='./dataset/RealImages/real_frf_1024x39.mat')
    parser.add_argument('--outdir_metrics',type=str,required=False,default='./Metrics/2D/xf/32/Random/Down4/Metrics_behaviour_real_image_random_down4')
    parser.add_argument('--outdir_plots',type=str,required=False,default='./Plots/2D/xf/32/Random/Down4/Plot_real_image_random_down4')
    parser.add_argument('--modeldir',type=str,required=False,default='./ModelCheckpoint/2D/xf/32/Random/Down4/super_res_xf_random_down4.h5')
    parser.add_argument('--snr',type=int,required=False,default=40)
    parser.add_argument('--lr',type=float,required=False,default=0.0004)

    args = parser.parse_args()

    X_test,Y_test,zero_row_idxs = prepareTestSet(args.datapath,args.snr,args.img_type,args.down_factor,args.numdim,args.ds_axis,
        args.images,args.downsampling, args.num_x_points, args.num_y_points, args.num_freqs)

    print('')
    print('X_test dimensions: '+ str(np.shape(X_test)))
    print('')
    print('Y_test dimensions: '+ str(np.shape(Y_test)))
    print('')

    ### TESTING
    print("Testing")

    opt = keras.optimizers.Adam(learning_rate=args.lr)

    if args.numdim==2:
        uNet = load_model(args.modeldir, custom_objects = {'loss': mask_mse(args.batch_size,args.num_x_points),
            'NMSE': NMSE, 'ncc': ncc, 'ReflectionPadding2D':ReflectionPadding2D})
        uNet.compile(loss=mask_mse(args.batch_size,args.num_x_points), optimizer=opt, metrics=[NMSE, ncc])

    elif args.numdim==3:
        uNet = load_model(args.modeldir, custom_objects = {'loss': mask_mse_3D(args.batch_size, args.num_freqs), 'NMSE': NMSE, 'ncc': ncc})
        uNet.compile(loss=mask_mse_3D(args.batch_size, args.num_freqs), optimizer=opt, metrics=[NMSE, ncc])

    score = uNet.evaluate(X_test, Y_test, verbose=1, batch_size=args.batch_size)
    probs = uNet.predict(X_test, verbose=1, batch_size=args.batch_size)

    print('')
    print("Custom metrics and plot results")
    print('')
    ### CALCULATE THE CUSTOM METRICS AND SAVE THEM USING PICKLE

    list_metrics = []
    list_plots = []

    num_x_points = args.num_x_points
    num_y_points = args.num_x_points
    num_freqs = args.num_freqs
    x = np.arange(0,num_x_points,1).tolist() # x-axis
    y = np.arange(0,num_y_points,1).tolist() # y-axis
    freq = np.arange(0,num_freqs,1).tolist()  #frequency axis

    x_ds = np.arange(0,num_x_points,args.down_factor)

    for idx in range(len(Y_test)):
        if args.numdim==2:
            down = X_test[idx][:,:,0]
            ground_truth = Y_test[idx][:,:,0]
            prediction = probs[idx][:,:,0]
        elif args.numdim==3:
            down = X_test[idx]
            ground_truth = Y_test[idx]
            prediction = probs[idx]

        nmse1 = nmse(ground_truth,prediction)
        ncc1 = NCC(ground_truth,prediction)

        if args.img_type=='xf':

            if args.downsampling=='random':

                ds_image = np.zeros((int(num_x_points/args.down_factor),num_freqs))

                zero_row_idx=0
                count=0

                for j in x:
                    if j==zero_row_idxs[zero_row_idx]:
                        ds_image[count,:] = down[j,:]
                        count=count+1
                        zero_row_idx=zero_row_idx+1
                    if zero_row_idx==num_x_points*(1/args.down_factor):
                        break

            elif args.downsampling=='regular':
                    ds_image = ground_truth[::args.down_factor,:]

            '''interp = interp2d(freq, x_ds, ds_image, kind='cubic')
            interp_image = interp(freq,x)'''
            interp_image = resample(ds_image,num_x_points,axis=0)

        nmse2 = nmse(ground_truth,interp_image)
        ncc2 = NCC(ground_truth,interp_image)

        '''plt.subplot(141), plt.title('Target')
        plt.imshow(ground_truth, cmap='bone', aspect='auto')
        plt.xlabel('Freq [Hz]'), plt.ylabel('X [m]')
        #plt.grid(None)
        plt.subplot(142), plt.title('U-net input')
        plt.imshow(down, cmap='bone', aspect='auto')
        plt.xlabel('Freq [Hz]')
        #plt.grid(None)
        plt.subplot(143), plt.title('Interp input')
        plt.imshow(ds_image, cmap='bone', aspect='auto')
        plt.xlabel('Freq [Hz]')
        #plt.grid(None)
        plt.subplot(144), plt.title('Interp output')
        plt.imshow(interp_image, cmap='bone', aspect='auto')
        plt.xlabel('Freq [Hz]')
        #plt.grid(None)
        plt.show()'''

        list_metrics.append((nmse1,nmse2,ncc1,ncc2))
        list_plots.append((down,ground_truth,prediction,interp_image))

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

if __name__=='__main__':
    main()
