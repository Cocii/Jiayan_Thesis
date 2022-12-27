import numpy as np
from math import log10,sqrt
import tensorflow as tf
import keras.backend as k

def log10(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def l1dist(target,output):
    """
    Sparsity of the error
    """
    return k.mean(k.abs(target-output))

def l2norm(tensor):
    """
    Energy of the image
    """
    return k.mean(tensor ** 2)

def l2dist(target, output):
    """
    Energy of the error
    """
    return k.mean((target - output)** 2)

def NMSE(target, output): 
    """
    Using Keras Backend
    """
    """
    :param target: groundtruth
    :param output: prediction
    :return: normalized mean squared error
    """
    nmse = l2dist(target, output) / l2norm(target)
    nmse = 20*log10(nmse)
    return nmse

def nmse(target, output):
    """
    Using Numpy
    """
    """
    :param target: groundtruth
    :param output: prediction
    :return: normalized mean squared error
    """
    nmse = np.mean((target - output) ** 2) / (np.mean(target ** 2))
    nmse = 20*np.log10(nmse)
    return nmse

def nmse_complex(target, output):
    """
    Using Numpy
    """
    """
    :param target: groundtruth
    :param output: prediction
    :return: normalized mean squared error
    """
    if isinstance(target[:,:,0], np.complex):
        errors = np.sqrt((target[:,:,0] - output[:,:,0])**2 + (target[:,:,1] - output[:,:,1])**2)
        mean_error = np.mean(errors)
        complex_target = target[:,:,0] + target[:,:,1] * 1j

        mean_true = np.mean(np.abs(complex_target))
        nmse = mean_error**2 / mean_true**2 
    else:
        nmse = np.mean((target - output) ** 2) / (np.mean(target ** 2))
    nmse = 20*np.log10(nmse)
    return nmse

def psnr(target, output):
    """
    :param target: groundtruth
    :param output: prediction
    :return: peack signal to noise ratio in dB
    """
    mse = np.mean((target - output) ** 2)
    if mse == 0:
        return 100
    max = 1.0
    psnr = 10*log10(max**2/mse)
    return psnr

# def psnr_complex(target, output):
#     """
#     :param target: groundtruth
#     :param output: prediction
#     :return: peack signal to noise ratio in dB
#     """
#     complex_target = target[:,:,0] + target[:,:,1] * 1j
#     complex_output = output[:,:,0] + output[:,:,1] * 1j
#     mse = np.mean((complex_target - complex_output) ** 2)
#     if mse == 0:
#         return 100
#     max = 1.0
#     psnr = 10*log10(max**2/mse)
#     return psnr

def ncc(target,output):
    """
    :param target: groundtruth
    :param output: prediction
    :return: normalized cross correlation 
    """
    ncc = k.sum(target*output)/(k.sum(target**2)**0.5 * k.sum(output**2)**0.5)
    return ncc

def NCC(target,output):
    """
    :param target: groundtruth
    :param output: prediction
    :return: normalized cross correlation 
    """
    ncc = np.sum(target*output)/(np.sum(target**2)**0.5 * np.sum(output**2)**0.5)
    return ncc

def mse(target,output):
    mse = tf.keras.losses.MeanSquaredError()
    mse = mse(target,output)
    return mse

def mask_mse(batch_size,num_x_points):
    def loss(target, output):
        """
        Computes the loss function as:
            masked_mse + 1e-6*l1dist
        The mask is composed by mask1*mask2:
            - mask1 --> target (to better reconstruct the peaks)
            - mask2 --> composed by 0.5s only on the first and last raws of the target images (--> spatial edges)
        """
        mask1 = target

        mask2 = np.zeros((batch_size,num_x_points,1024,1))
        first = 0
        last = np.shape(mask2)[1]-1
        mask2[:,first,:,:] = np.ones((np.shape(mask2)[0],np.shape(mask2)[2],1))*0.5
        mask2[:,last,:,:] = mask2[:,first,:,:]
        mask2 = tf.stack(mask2)
        mask2 = tf.cast(mask2,k.floatx())
        
        mask3 = k.max(target)-target

        mask = 0.7*mask1+mask2+0.3*mask3

        return l2dist(mask*target,mask*output)
    return loss

def mask_mse_3D(batch_size, num_freqs):
    def loss(target,output):

        mask1 = np.ones((batch_size,16,64,num_freqs,1))
        mask2 = np.zeros((np.shape(mask1)))
        mask3 = target

        first = 0
        lasty = np.shape(mask2)[1]-1
        lastx = np.shape(mask2)[2]-1

        mask2[:,:,first,:,:] = np.ones((np.shape(mask2)[0],np.shape(mask2)[1],np.shape(mask2)[3],1))*0.5
        mask2[:,:,lastx,:,:] = mask2[:,:,first,:,:]
        mask2[:,first,:,:,:] = np.ones((np.shape(mask2)[0],np.shape(mask2)[2],np.shape(mask2)[3],1))*0.5
        mask2[:,lasty,:,:,:] = mask2[:,first,:,:,:]

        mask2 = tf.stack(mask2)
        mask2 = tf.cast(mask2,k.floatx())

        mask = mask1+mask2+mask3
        return l2dist(mask*target, mask*output)
    return loss

'''def mask_mse_plus_L1(target,output):
    mask1 = target
    mask2 = np.zeros((1,64,1024,1))
    first = 0
    last = np.shape(mask2)[1]-1
    mask2[:,first,:,:] = np.ones((np.shape(mask2)[0],np.shape(mask2)[2],1))*0.5
    mask2[:,last,:,:] = mask2[:,first,:,:]
    mask2 = tf.stack(mask2)
    mask2 = tf.cast(mask2,k.floatx())
    mask3 = k.max(target)-target
    mask = 0.7*mask1+mask2+0.3*mask3
    return l2dist(mask*target,mask*output)

def mask_mse_3D(target, output):

    mask1 = np.ones((1,16,64,512,1))
    mask2 = np.zeros((np.shape(mask1)))
    mask3 = target

    first = 0
    lasty = np.shape(mask2)[1]-1
    lastx = np.shape(mask2)[2]-1

    mask2[:,:,first,:,:] = np.ones((1,np.shape(mask2)[1],np.shape(mask2)[3],1))*0.5
    mask2[:,:,lastx,:,:] = mask2[:,:,first,:,:]
    mask2[:,first,:,:,:] = np.ones((np.shape(mask2)[0],np.shape(mask2)[2],np.shape(mask2)[3],1))*0.5
    mask2[:,lasty,:,:,:] = mask2[:,first,:,:,:]

    mask = mask1+mask2+mask3
    return k.mean((mask*target - mask*output)** 2)'''


'''A = np.array([[4, 3], [2, 1]])
B = np.array([[-4, -3], [-2, -1]])

A = np.random.rand(1,16,64,512,1)
A = np.abs(A)/np.max(A)
B = np.random.rand(1,16,64,512,1)
B = np.abs(B)/np.max(B)

target = k.variable(value=A, dtype='float64',name='A')
output = k.variable(value=B, dtype='float64',name='B')

print('l2dist= '+str(l2dist(target,output)))
print('l1dist= '+str(l1dist(target,output)))'''