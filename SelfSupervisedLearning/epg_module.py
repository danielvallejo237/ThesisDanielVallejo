# Módulo modificado de EPG para la estimación paramétrica de la distribución de T2 tipo voxelwise
# @Author ejcanalesr-danielvallejo237 / Centro de Investigación en Matemáticas A.C. - Ecole Polytechnique Fédérale de Lausanne (EPFL)

from __future__ import division
import torch 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F
import argparse
from typing import Union, Optional
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
from multiprocessing import Process, Lock, Value
from joblib import Parallel, delayed
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger("torchepg")


def create_met2_design_matrix_epg(Npc, T2s, T1s, nEchoes, tau, flip_angle, TR, dtype=torch.float64, device="cpu"):
    design_matrix = torch.zeros((nEchoes, Npc), dtype=dtype, device=device)
    TR = torch.tensor(TR, dtype=dtype, device=device)
    rad = torch.tensor(math.pi / 180.0, dtype=dtype, device=device)  # constant to convert degrees to radians
    for cols in range(Npc):
        T1s_col = torch.tensor([1.0 / T1s[cols]], dtype=dtype, device=device)
        T2s_col = torch.tensor([1.0 / T2s[cols]], dtype=dtype, device=device)
        flip_angle_rad = flip_angle * rad
        flip_angle_half_rad = flip_angle / 2.0 * rad
        epg_s = epg_signal(nEchoes, tau, T1s_col, T2s_col, flip_angle_rad, flip_angle_half_rad)
        #Put all tensors in the same device
        epg_s=epg_s.to(device=device,dtype=dtype)
        TR = TR.to(device=device,dtype=dtype)
        T1s_col = T1s_col.to(device=device,dtype=dtype)
        signal = (1.0 - torch.exp(-TR / T1s_col)) * epg_s
        design_matrix[:, cols] = signal.flatten()
    return design_matrix

def epg_signal(n, tau, R1vec, R2vec, alpha, alpha_exc, dtype=torch.float64, device="cpu"):
    nRates = R2vec.shape[0]
    tau = tau / 2.0

    # defining signal matrix
    H = torch.zeros((n, nRates), dtype=dtype, device=device)

    # RF mixing matrix
    T = fill_T(n, alpha, dtype=dtype, device=device)

    # Selection matrix to move all traverse states up one coherence level
    S = fill_S(n, dtype=dtype, device=device)

    for iRate in range(nRates):
        # Relaxation matrix
        R2 = R2vec[iRate]
        R1 = R1vec[iRate]

        R0 = torch.zeros((3, 3), dtype=dtype, device=device)
        R0[0, 0] = torch.exp(-tau * R2)
        R0[1, 1] = torch.exp(-tau * R2)
        R0[2, 2] = torch.exp(-tau * R1)

        R = fill_R(n, tau, R0, R2, dtype=dtype, device=device)
        # Precession and relaxation matrix
        P = torch.matmul(R, S)
        # Matrix representing the inter-echo duration
        E = torch.matmul(torch.matmul(P, T), P)
        H = fill_H(R, n, E, H, iRate, alpha_exc, dtype=dtype, device=device)

    return H

def fill_S(n, dtype=torch.float64, device="cpu"):
    the_size = 3 * n + 1
    S = torch.zeros((the_size, the_size), dtype=dtype, device=device)
    S[0, 2] = 1.0
    S[1, 0] = 1.0
    S[2, 5] = 1.0
    S[3, 3] = 1.0
    for o in range(2, n + 1):
        offset1 = ((o - 1) - 1) * 3 + 2
        offset2 = ((o + 1) - 1) * 3 + 3
        if offset1 <= (3 * n + 1):
            S[3 * o - 2, offset1 - 1] = 1.0  # F_k <- F_{k-1}
        if offset2 <= (3 * n + 1):
            S[3 * o - 1, offset2 - 1] = 1.0  # F_-k <- F_{-k-1}
        S[3 * o, 3 * o] = 1.0  # Z_order
    return S

def fill_T(n, alpha, dtype=torch.float64, device="cpu"):
    T0 = torch.zeros((3, 3), dtype=dtype, device=device)
    T0[0, :] = torch.tensor([torch.cos(alpha / 2.0)**2, torch.sin(alpha / 2.0)**2, torch.sin(alpha)], dtype=dtype, device=device)
    T0[1, :] = torch.tensor([torch.sin(alpha / 2.0)**2, torch.cos(alpha / 2.0)**2, -torch.sin(alpha)], dtype=dtype, device=device)
    T0[2, :] = torch.tensor([-0.5 * torch.sin(alpha), 0.5 * torch.sin(alpha), torch.cos(alpha)], dtype=dtype, device=device)

    T = torch.zeros((3 * n + 1, 3 * n + 1), dtype=dtype, device=device)
    T[0, 0] = 1.0
    T[1:3+1, 1:3+1] = T0
    for itn in range(n-1):
        T[(itn+1)*3+1:(itn+2)*3+1, (itn+1)*3+1:(itn+2)*3+1] = T0
    return T

def fill_R(n, tau, R0, R2, dtype=torch.float64, device="cpu"):
    R = torch.zeros((3 * n + 1, 3 * n + 1), dtype=dtype, device=device)
    R[0, 0] = torch.exp(-tau * R2)
    R[1:3+1, 1:3+1] = R0
    for itn in range(n-1):
        R[(itn+1)*3+1:(itn+2)*3+1, (itn+1)*3+1:(itn+2)*3+1] = R0
    return R

def fill_H(R, n, E, H, iRate, alpha_exc, dtype=torch.float64, device="cpu"):
    numero = int(R.shape[0])
    x = torch.zeros((numero, 1), dtype=dtype, device=device)
    x[0] = torch.sin(alpha_exc)
    x[1] = 0.0
    x[2] = torch.cos(alpha_exc)
    for iEcho in range(n):
        x = torch.matmul(E, x)
        H[iEcho, iRate] = x[0, 0]
    return H

# ------------------------------------------------------------ #

# ------------------------------------------------------------ #
'''
    Esta es la función que hay que evaluar acerca de su buen o mal funcionamiento
'''

 # Decorador que indica que no es requerido el cáclulo de gradientes
def return_tensors(refoc_angle: torch.Tensor, t2s: torch.Tensor, weights: torch.Tensor, config:any,device:Optional[torch.device]=None,dtype:Optional[torch.dtype]=None) -> torch.Tensor:
    # creación de las matriz de diseño pero en paralelo
    in_device = device if device is not None else config.device
    in_dtype = dtype if dtype is not None else config.dtype
    t1s= config.epg_parameters["t1"] * torch.ones_like(t2s, dtype=in_dtype,device=in_device)
    with torch.no_grad():
        matrices_paralelo= Parallel(n_jobs=-1)(delayed(create_met2_design_matrix_epg)(config.compartments, t2s[i], t1s[i], config.epg_parameters["nechos"], config.epg_parameters["timeinit"], refoc_angle[i], config.epg_parameters["reptime"], dtype=in_dtype, device=in_device) for i in range(refoc_angle.shape[0]))
    matrices=torch.stack(matrices_paralelo, dim=0)
    signals=torch.bmm(matrices,weights.unsqueeze(-1)).squeeze(-1)
    signals=signals/signals[:,0].unsqueeze(1)
    return signals

def return_tensors_no_parallel(refoc_angle: torch.Tensor, t2s: torch.Tensor, weights: torch.Tensor, config:any,device:Optional[torch.device]=None,dtype:Optional[torch.dtype]=None) -> torch.Tensor:
    in_device = device if device is not None else config.device
    in_dtype = dtype if dtype is not None else config.dtype
    t1s= config.epg_parameters["t1"] * torch.ones_like(t2s, dtype=in_dtype,device=in_device)
    matrices=torch.zeros((refoc_angle.shape[0],config.epg_parameters["nechos"],config.compartments),dtype=in_dtype,device=in_device)
    for i in range(refoc_angle.shape[0]):
        matrices[i]=create_met2_design_matrix_epg(config.compartments, t2s[i], t1s[i], config.epg_parameters["nechos"], config.epg_parameters["timeinit"], refoc_angle[i], config.epg_parameters["reptime"], dtype=in_dtype, device=in_device)
    signals=torch.bmm(matrices,weights.unsqueeze(-1)).squeeze(-1)
    signals=signals/signals[:,0].unsqueeze(1)
    return signals

class EPGModule(nn.Module):
    def __init__(self,config:any):
        super(EPGModule, self).__init__()
        self.config=config
    def forward(self,refoc_angle:torch.Tensor,t2s:torch.Tensor,weights:torch.Tensor)->torch.Tensor:
        t1s= self.config.epg_parameters["t1"] * torch.ones_like(t2s, dtype=self.config.dtype,device=self.config.device)
        # for each element in batch process the matrices one by one
        signals=torch.zeros((refoc_angle.shape[0],self.config.epg_parameters["nechos"]),dtype=self.config.dtype,device=self.config.device)
        for i in range(refoc_angle.shape[0]):
            matrices=create_met2_design_matrix_epg(self.config.compartments, t2s[i], t1s[i], self.config.epg_parameters["nechos"], self.config.epg_parameters["timeinit"], refoc_angle[i], self.config.epg_parameters["reptime"], dtype=self.config.dtype, device=self.config.device)
            signal=torch.matmul(matrices,weights[i])
            signal=signal/signal[0]
            signals[i]=signal
        return signals
# ------------------------------------------------------------ #


def _return_tensors(refoc_angle: torch.Tensor, t2s: torch.Tensor, weights: torch.Tensor,parameters:dict,device:Optional[torch.device]=None,dtype:Optional[torch.dtype]=None) -> torch.Tensor:
    #Parameter semplaza a config epg_parameters de tal forma que no necesitemos la configuración completa del archivo de redes neuronales
    in_device = device
    in_dtype = dtype
    t1s= parameters["t1"] * torch.ones_like(t2s, dtype=in_dtype,device=in_device)
    matrices=torch.zeros((refoc_angle.shape[0],parameters["nechos"],parameters["n"]),dtype=in_dtype,device=in_device)
    with torch.no_grad():
        matrices_paralelo= Parallel(n_jobs=-1)(delayed(create_met2_design_matrix_epg)(parameters['n'], t2s[i], t1s[i], parameters["nechos"], parameters["timeinit"], refoc_angle[i], parameters["reptime"], dtype=in_dtype, device=in_device) for i in range(refoc_angle.shape[0]))
    matrices= torch.stack(matrices_paralelo, dim=0)
    signals=torch.bmm(matrices,weights.unsqueeze(-1)).squeeze(-1)
    signals=signals/signals[:,0].unsqueeze(1)
    return signals

# Add rician noise to tensors
def rician_noise(signal:torch.Tensor,snr_range:tuple,device:Optional[torch.device]=None,dtype:Optional[torch.dtype]=None)->torch.Tensor:
    in_device = device if device is not None else signal.device
    in_dtype = dtype if dtype is not None else signal.dtype
    snr = torch.rand((signal.shape[0], 1)) * (snr_range[1] - snr_range[0]) + snr_range[0]
    # use the mean intensity of the first echo as the reference
    scale_factor = torch.mean(signal, dim=0)[0]
    # calculate variance (https://www.statisticshowto.com/rayleigh-distribution/)
    variance = scale_factor * (1 / (snr * np.sqrt(np.pi / 2)))
    # noise_real = tf.random.normal(tf.shape(signal), 0, variance)
    noise_real = torch.randn_like(signal,dtype=in_dtype, device=in_device)
    # noise_img = tf.random.normal(tf.shape(signal), 0, variance)
    noise_img = torch.randn_like(signal, dtype=in_dtype, device=in_device)
    # shift variance of both distributions to variance
    noise_real = noise_real * variance
    noise_img = noise_img * variance
    # noisy_signal = ((noise_real+signal)**2 + noise_img**2)**0.5
    noisy_signal = torch.sqrt((noise_real + signal)**2 + noise_img**2)
    return noisy_signal
