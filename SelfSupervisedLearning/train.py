from parameters import *
from utils import Config,TrainingConfig
from self_supervised_network import JointNetworkV2,train_JointNetworkV2,predict
from epg_module import rician_noise
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os 
import argparse

config=Config(
    indim=N_ECHOS,
    compartments=NUM_PARMETERS,
    t2_myelin=T2_MYELIN_TRAINING,
    t2_ie=T2_IE_TRAINING,
    t2_csf=T2_CSF_TRAINING,
    angle_boundaries=ANGLE_BOUNDARIES_TRAINING,
    epg_parameters={
                "nechos":N_ECHOS,
                "timeinit":TIME_INIT, #Tiempo de inicio de la secuencia en milisegundos
                "reptime":REP_TIME, #Tiempo de repetición de la secuencia en milisegundos
                "t1":T1
            },
    snr_range = SNRRANGE
    )

tconfig=TrainingConfig(epochs=EPOCHS,
                       lr=LR)

# Inicializamos la semilla para que los experimentos sean reproducibles
torch.manual_seed(0)

# Inicializamos el modelo
model=JointNetworkV2(config)


def get_files_names(inpath:str):
    file=inpath.split("/")[-1]
    weights=os.path.join(inpath,file+"_weights.npy")
    t2s=os.path.join(inpath,file+"_t2means.npy")
    signals=os.path.join(inpath,file+"_signals.npy")
    return weights,t2s,signals

def load_data(inpath:str):
    weights,t2s,signals=get_files_names(inpath)
    weights=np.load(weights)
    t2s=np.load(t2s)
    signals=np.load(signals)
    if SNRRANGE is not None:
        signals=rician_noise(signals,SNRRANGE)
    return weights,t2s,signals

def create_dloader(signals:np.ndarray):
    dataset=torch.utils.data.TensorDataset(torch.tensor(signals,dtype=config.dtype),torch.tensor(np.zeros(signals.shape[0])+180.0,dtype=config.dtype,device=config.device))
    dloader=torch.utils.data.DataLoader(dataset,batch_size=tconfig.batch_size,shuffle=True)
    return dloader

parser=argparse.ArgumentParser(description='Entrenamiento de la red neuronal')
parser.add_argument('--inpath',type=str,help='Ruta de los archivos de entrada')

if __name__=='__main__':
    args=parser.parse_args()
    weights,t2s,signals=load_data(args.inpath)
    dloader=create_dloader(signals)
    model.eval()
    train_JointNetworkV2(model,tconfig,dloader)