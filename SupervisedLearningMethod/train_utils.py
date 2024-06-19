# @Author: Daniel Vallejo Aldana
# @Institution: Ecole Polytechnique Fédérale de Lausanne / LTS5 laboratory
# Implementación de la red neuronal en tensorflow para entrenamiento en batch

import tensorflow as tf
import numpy as np
import random
import math
from joblib import Parallel, delayed
import multiprocessing as mp
import os
import time
import argparse
import json
from keras.callbacks import History
#### Funciones utilitarias ####

# Agregamos ruido riciano a la señal de resonancia magnética #

def addRicianNoise(signal,bound=(30,60)):
    Snr=random.uniform(bound[0],bound[1])
    r1=[random.gauss(0,1./Snr) for _ in range(signal.shape[0])]
    r2=[random.gauss(0,1./Snr) for _ in range(signal.shape[0])]
    rsig=[math.sqrt((x+y)**2+z**2) for (x,y,z) in zip(signal,r1,r2)]
    return np.array(rsig),Snr
# Función que ensucia en paralelo la señal de entrada #

def addRicianNoiseParallel(signals:np.ndarray,bound=(30,60)):
    for work in range(mp.cpu_count()):
        random.seed((os.getpid() * int(time.time())) % 123456789)
    r=Parallel(n_jobs=mp.cpu_count())(delayed(addRicianNoise)(signals[i],bound) for i in range(signals.shape[0]))
    nsignal,snr=zip(*r)
    return np.array(nsignal) #Solo necesitamos la señal ensuciada

arr=np.arange(1,61,1)
# batch_size=512
arr=np.tile(arr,(2000, 1))
arr_tf=tf.constant(arr.astype('float32'), dtype=tf.float32)


#Implementation of the Wasserstein Distance
def wasserstein_distance(y_actual,y_pred):
    abs_cdf_difference=tf.math.abs(tf.math.cumsum(y_actual-y_pred,axis=1))
    return tf.reduce_mean(0.5*tf.reduce_sum(tf.math.multiply(-arr_tf[:,:-1]+arr_tf[:,1:],abs_cdf_difference[:,:-1]+abs_cdf_difference[:,1:]),axis=1))

#Combination loss function used in MIML
def MSE_wasserstein_combo(y_actual,y_pred):
    wass_loss=wasserstein_distance(y_actual,y_pred)
    MSE= tf.math.reduce_mean(tf.reduce_mean(tf.math.squared_difference(y_pred, y_actual),axis=1))
    return wass_loss+1000.*MSE


def create_model(indim:int,out_dim:int,use_softmax=True):
    inputs = tf.keras.Input(shape=(indim,))
    x = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(inputs)
    x = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(x)
    x=tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(x)
    x=tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(x)
    x=tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(x)
    x=tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(x)
    outputs=tf.keras.layers.Dense(out_dim, activation=tf.keras.activations.softmax if use_softmax else tf.keras.activations.relu, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_custom_model(indim:int,outdim:int,use_softmax:bool=True,hidden_layers:int=4,hidden_units:int=256):
    inputs=tf.keras.Input(shape=(indim,))
    x=tf.keras.layers.Dense(hidden_units,activation=tf.nn.leaky_relu,kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(inputs)
    for _ in range(hidden_layers):
        x=tf.keras.layers.Dense(hidden_units,activation=tf.nn.leaky_relu,kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(x)
    outputs=tf.keras.layers.Dense(outdim,activation=tf.keras.activations.softmax if use_softmax else tf.keras.activations.relu,kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(x)
    model=tf.keras.Model(inputs=inputs,outputs=outputs)
    return model