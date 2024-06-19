#   Código de generación de señales sintéticas
#   @Author Daniel Vallejo Aldana
#   @Institution CIMAT / Ecole Polytechnique Federale de Lausanne (EPFL)
#   Bug fixing - Adjusting the T2 generation signals to a certain SNR


#Importación de librerías

"""
Los datos que necesitamos generar en este caso son los ángulos de refocusing, las señales de resonancia magnética así como
el Signal to Noise Ratio de cada una de las señales y usar dichas representaciones para poder estimad de forma más precisa
la distribución de T2 
"""

from __future__ import division
import argparse
from scipy.stats import rv_continuous,norm
from typing import Union,Optional
import time
import random
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import functools
import math
from tqdm import tqdm
from scipy.stats import truncnorm
import os
import math
import logging
from multiprocessing import Process, Lock,Value
from joblib import Parallel,delayed
import json

#En este caso solo tenemos dos tipos de tejido, podemos cargar una nueva configuración desde un archivo de tipo JSON

#Defining T2 Intervals for the generated data including mean and standar deviation of the different tissues

#Double check to this interval

# Definición de las funciones auxiliares
class Log:
    '''
    Definición de la clase logaritmo, al no tener numpy una clase logaritmo para cada una de las poibles bases
    se crea una clase logaritmo para crear la distribución de los bines de la reducción de dimensión de las
    señales de resonancia magnética
    '''
    def __init__(self,base):
        self._base=base
    def __call__(self,x):
        return np.log(x)/np.log(self._base)

#Creación de la matriz de transformación
def BuildTransformMatrix(dist_range:tuple=(1,2000),first_resolution:int=20000,second_resolution:int=30,scale:Optional[Log]=None):
    '''
    Transformation matrix to downsample a high resolution matrix to a low resolution matrix
    the dist range is set to 1 to 2000 milliseconds
    first_resolution corresponds to the hr resolution and is set to 20000
    the low resolution is set to 60
    we use different scales depending on the data (logarithmic for humans and linear for rats at 7T)
    '''
    original=np.linspace(dist_range[0],dist_range[1],num=first_resolution)
    points=np.linspace(dist_range[0],dist_range[1],num=second_resolution).tolist()
    if scale is not None:
        powers=np.linspace(scale(dist_range[0]),scale(dist_range[1]),num=second_resolution).tolist()
        points=[scale._base**p for p in powers]
    Transform=np.zeros((second_resolution,first_resolution))
    assert(second_resolution>=2)
    La,Lb=points[0],points[1]
    row=0
    for i in range(original.shape[0]-1):
        if original[i]>=Lb:
            row+=1
            La=Lb
            Lb=points[row+1]
        Transform[row][i]=1-(abs(original[i]-La)/abs(Lb-La))
        Transform[row+1][i]=1-(abs(original[i]-Lb)/abs(Lb-La))
    Transform[-1][-1]=1.0
    return Transform

def GaussianMixture(x:any,submodels:list,alphas:np.ndarray):
    '''
    Generador de un modelo de mezcla de Gaussianas, este modelo sirve para generar las distribuciones
    iniciales de alta resulución usadas para generar la señal de resonancia magnética
    '''
    assert(len(alphas)==len(submodels))
    assert(type(x)==np.ndarray and type(alphas)==np.ndarray)
    Pdf=alphas[0]*submodels[0].pdf(x) #These models are initializated outside
    for i,submodel in enumerate(submodels[1:]):
        Pdf+=alphas[i+1]*submodel.pdf(x)
    return Pdf

def generateT2Sequence(start:int,end:int,resolution:int=20000,scale:Optional[Log]=None):
    '''
    This function generates the data corresponding to the discretization of the T2 scale
    li represents a linear scale and log a logarithmic scale
    '''
    if scale is None:
        return np.linspace(start,end,num=resolution)
    else:
        begin,stop=scale(start),scale(end)
        powers=np.linspace(begin,stop,num=resolution).tolist()
        return np.array([scale._base**p for p in powers])


def generateSingleHRdist(parameters:tuple,sequence:Union[list,np.ndarray],resolution:int=20000,myelinRange=(0,0.4)):
    '''
    Here is where the things become dirty, we want to change the seed just one time
    if the function has not been called before
    '''
    if type(sequence)!=np.ndarray:
        sequence=np.array(sequence) #We need a list
    Means=[]
    parameter=random.choice(list(parameters.keys()))
    spec=parameters[parameter]
    Nmeans=len(spec)//4
    for i in range(Nmeans):
        mean=round(random.uniform(spec[4*i],spec[4*i+1]),2)
        sd=round(random.uniform(spec[4*i+2],spec[4*i+3]),2)
        Means.append(norm(mean,sd))
    indicator=spec[-1]
    alphas=None
    if indicator:
        alphas=np.random.uniform(myelinRange[0],myelinRange[1],size=1)
        weights=np.random.uniform(0,1,size=Nmeans-1)
        weights/=np.sum(weights)
        weights=(weights*(1-alphas[0]))
        alphas=np.concatenate((alphas,weights),axis=0)
    else:
        alphas=np.random.uniform(0,1,size=Nmeans)
        alphas/=np.sum(alphas)
    dist=GaussianMixture(sequence,Means,alphas)
    return sequence,dist/sum(dist)

def generateLRsignal(HRSignal,TransformMatrix):
    dist=np.dot(TransformMatrix,HRSignal)
    return dist/np.sum(dist)

#Parte del formalismo de EPG
'''
@Author Erik Canales
'''
# ------------------------------------------------------------------------------
# Functions to generate the Dictionary of multi-echo T2 signals using the exponential model
def create_met2_design_matrix(TEs, T2s):
    '''
    Creates the Multi Echo T2 (spectrum) design matrix.
    Given a grid of echo times (numpy vector TEs) and a grid of T2 times
    (numpy vector T2s), it returns the deign matrix to perform the inversion.
    '''
    M = len(TEs)
    N = len(T2s)
    design_matrix = np.zeros((M,N))
    for row in range(M):
        for col in range(N):
            exponent = -(TEs[row] / T2s[col])
            design_matrix[row,col] = np.exp(exponent)
        # end for col
    # end for row
    return design_matrix
#end fun

# Functions to generate the Dictionary of multi-echo T2 signals using the EPG model
def create_met2_design_matrix_epg(Npc, T2s, T1s, nEchoes, tau, flip_angle, TR):
    '''
    Creates the Multi Echo T2 (spectrum) design matrix.
    Given a grid of echo times (numpy vector TEs) and a grid of T2 times
    (numpy vector T2s), it returns the deign matrix to perform the inversion.
    *** Here we use the epg model to simulate signal artifacts
    '''
    design_matrix = np.zeros((nEchoes, Npc))
    rad           = np.pi/180.0  # constant to convert degrees to radians
    for cols in range(Npc):
        signal = (1.0 - np.exp(-TR/T1s[cols])) * epg_signal(nEchoes, tau, np.array([1.0/T1s[cols]]), np.array([1.0/T2s[cols]]), flip_angle * rad, flip_angle/2.0 * rad)
        #signal = (1.0 - np.exp(-TR/T1s[cols])) * epg_signal(nEchoes, tau, np.array([1.0/T1s[cols]]), np.array([1.0/T2s[cols]]), flip_angle * rad, 90.0 * rad)
        design_matrix[:, cols] = signal.flatten()
        # end for row
    return design_matrix
#end fun

def epg_signal(n, tau, R1vec, R2vec, alpha, alpha_exc):
    nRates = R2vec.shape[0]
    tau = tau/2.0

    # defining signal matrix
    H = np.zeros((n, nRates))

    # RF mixing matrix
    T = fill_T(n, alpha)

    # Selection matrix to move all traverse states up one coherence level
    S = fill_S(n)

    for iRate in range(nRates):
        # Relaxation matrix
        R2 = R2vec[iRate]
        R1 = R1vec[iRate]

        R0      = np.zeros((3,3))
        R0[0,0] = np.exp(-tau*R2)
        R0[1,1] = np.exp(-tau*R2)
        R0[2,2] = np.exp(-tau*R1)
        #print(R1,R2)

        R = fill_R(n, tau, R0, R2)
        # Precession and relaxation matrix
        P = np.dot(R,S)
        #print(P)
        # Matrix representing the inter-echo duration
        E = np.dot(np.dot(P,T),P)
        #print(E)
        H = fill_H(R, n, E, H, iRate, alpha_exc)
        # end
    return H
#end fun
def fill_S(n):
    the_size = 3*n + 1
    #print(n)
    S = np.zeros((the_size,the_size))
    S[0,2]=1.0
    S[1,0]=1.0
    S[2,5]=1.0
    S[3,3]=1.0
    for o in range(2,n+1):
        offset1=( (o-1) - 1)*3 + 2
        offset2=( (o+1) - 1)*3 + 3
        if offset1<=(3*n+1):
            S[3*o-2,offset1-1] = 1.0  # F_k <- F_{k-1}
        # end
        if offset2<=(3*n+1):
            S[3*o-1,offset2-1] = 1.0  # F_-k <- F_{-k-1}
        # end
        S[3*o,3*o] = 1.0              # Z_order
    # end for
    return S
#end fun
def fill_T(n, alpha):
    T0      = np.zeros((3,3))
    T0[0,:] = [math.cos(alpha/2.0)**2, math.sin(alpha/2.0)**2,  math.sin(alpha)]
    T0[1,:] = [math.sin(alpha/2.0)**2, math.cos(alpha/2.0)**2, -math.sin(alpha)]
    T0[2,:] = [-0.5*math.sin(alpha),   0.5*math.sin(alpha),     math.cos(alpha)]

    T = np.zeros((3*n + 1, 3*n + 1))
    T[0,0] = 1.0
    T[1:3+1, 1:3+1] = T0
    for itn in range(n-1):
        T[(itn+1)*3+1:(itn+2)*3+1,(itn+1)*3+1:(itn+2)*3+1] = T0
    # end
    return T
#end fun
def fill_R(n, tau, R0, R2):
    R  = np.zeros((3*n + 1, 3*n + 1))
    R[0,0] = np.exp(-tau*R2)
    R[1:3+1, 1:3+1] = R0
    for itn in range(n-1):
        R[(itn+1)*3+1:(itn+2)*3+1,(itn+1)*3+1:(itn+2)*3+1] = R0
    # end
    return R
#end fun

def fill_H(R, n, E, H, iRate, alpha_exc):
    numero=int(R.shape[0])
    x= np.zeros((numero,1),dtype=np.float64)
    x[0] = math.sin(alpha_exc)
    x[1] = 0.0
    x[2] = math.cos(alpha_exc)
    for iEcho in range(n):
        x = np.dot(E,x)
        H[iEcho, iRate] = x[0][0]
    #end for IEcho
    return H
#end fun
def create_Dic_3D(Npc, T2s, T1s, nEchoes, tau, alpha_values, TR):
    dim3   = len(alpha_values)
    Dic_3D = np.zeros((nEchoes, Npc, dim3))
    for iter in range(dim3):
        Dic_3D[:,:,iter] = create_met2_design_matrix_epg(Npc, T2s, T1s, nEchoes, tau, alpha_values[iter], TR)
    #end for
    return Dic_3D
#end fun

def generateMatrix(refoc_angle:int,resolution:int,t2s:any,nechos:int,timeinit:int,reptime:int,t1:int):
    t1s=np.tile(t1,resolution)
    sequence=create_met2_design_matrix_epg(resolution,t2s,t1s,nechos,timeinit,refoc_angle,reptime)
    return sequence

def generateCuboid(refoc_angle_start:int,refoc_angle_end:int,resolution:int,t2s:any,nechos:int,timeinit:int,reptime:int,t1:int):
    Cuboid=np.zeros(((refoc_angle_end-refoc_angle_start)+1,nechos,resolution))
    # Generamos la matriz de EPG en paralelo #
    logging.info("Using {} cores to generate EPG matrix".format(mp.cpu_count()))
    Cuboid=Parallel(n_jobs=mp.cpu_count())(delayed(generateMatrix)(refoc_angle_start+i, resolution, t2s, nechos, timeinit, reptime, t1) for i in range(refoc_angle_end-refoc_angle_start+1))
    return Cuboid

def addRicianNoise(signal,bound=(30,60)):
    """
    Regresamos el SNR de la señal de resonancia
    Necesitamos agregar el SNR de la señal de forma controlada
    """
    #Snr=random.gauss(0.5*(bound[0]+bound[1]),np.sqrt((bound[1]-bound[0])))
    Snr=random.uniform(bound[0],bound[1])
    r1=[random.gauss(0,1./Snr) for _ in range(signal.shape[0])]
    r2=[random.gauss(0,1./Snr) for _ in range(signal.shape[0])]
    rsig=[math.sqrt((x+y)**2+z**2) for (x,y,z) in zip(signal,r1,r2)]
    return np.array(rsig),Snr
def generateDistAndSignal(parameters:any,sequence:any,Cuboid:any,TransformMat:any,rangle:int,rab:tuple=(90,180),snrb:tuple=(10,100),scale:Optional[Log]=None,myelinrange:tuple=(0,0.4)):
    sequence,distribution=generateSingleHRdist(parameters,sequence,myelinRange=myelinrange)
    lrDist=generateLRsignal(distribution,TransformMat)
    signal=np.dot(Cuboid[rangle-rab[0]],lrDist)
    signal=signal/signal[0]
    signal,noise=addRicianNoise(signal,bound=snrb)
    return signal,lrDist/sum(lrDist),rangle,noise

#Generación de ángulos de refocusing con diferente distribución
def generate_angles(start:float,end:float,number:int=10000,rab:tuple=(90,180),use_uniform:bool=False):
    if not use_uniform:
        samples=truncnorm(start,end,loc=0).rvs(number)
    else:
        samples=np.random.uniform(rab[0],rab[1],size=number).tolist()
    angles=[]
    for s in samples:
        if not use_uniform:
            angles.append(rab[0]+np.round((start-s)/(end-start)*rab[1]))
        else:
            angles.append(np.round(s))
    angles=np.array(angles,dtype=int)
    assert(np.min(angles)==rab[0] and np.max(angles)==rab[1])
    return list(angles)

def GenNSignalsandDists(number:int,parameters:any,sequence:any,Cuboid:any,TransformMat:any,snr_bounds:tuple,scale:Optional[Log]=None,rab:tuple=(90,180),closure:tuple=(0.0,2.4),num_workers:int=-1,use_uniform:bool=False,myelinrange:tuple=(0,0.4)):
    #Parameters es un diccionario de parámetros con los que contienen tuplas para generar señales y distribuciones de resonancia magnética
    angles=generate_angles(start=closure[0],end=closure[1],number=number,rab=rab,use_uniform=use_uniform)
    for work in range(num_workers):
        random.seed((os.getpid() * int(time.time())) % 123456789)
    r=Parallel(n_jobs=num_workers)(delayed(generateDistAndSignal)(parameters=parameters,sequence=sequence,Cuboid=Cuboid,TransformMat=TransformMat,rangle=angles[i],snrb=snr_bounds,scale=scale,rab=rab,myelinrange=myelinrange) for i in range(number))
    signals,dist,angles,noise=zip(*r)
    return signals,dist,angles,noise