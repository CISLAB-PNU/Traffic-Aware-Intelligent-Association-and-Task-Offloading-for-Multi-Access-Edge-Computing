import tensorflow as tf
from tensorflow.keras import backend as K


def mape(y_true, y_pred):
    """
    Returns the mean absolute percentage error.
    For examples on losses see:
    https://github.com/keras-team/keras/blob/master/keras/losses.py
    """
    return (K.abs(y_true - y_pred) / K.abs(y_pred)) * 100
    #diff = K.abs(y_true - y_pred) / K.abs(y_true)
    #return 100. * K.mean(diff)#, axis=-1)

def smape(y_true, y_pred):
    """
    Returns the Symmetric mean absolute percentage error.
    For examples on losses see:
    https://github.com/keras-team/keras/blob/master/keras/losses.py
    """
    return 100*K.mean(K.abs(y_pred - y_true) / ((K.abs(y_true) + K.abs(y_pred))), axis=-1)
    #Symmetric mean absolute percentage error
    #return 100 * K.mean(K.abs(y_pred - y_true) / (K.abs(y_pred) + K.abs(y_true)))#, axis=-1)

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def mae(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def mase(y_true, y_pred):
    sust = K.mean(K.abs(y_true[:,1:] - y_true[:,:-1]))
    diff = K.mean(K.abs(y_pred - y_true))

    return diff/sust

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )