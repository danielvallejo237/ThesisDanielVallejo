import numpy as np
import torch
import torch.nn as nn
from typing import Union,Optional,Dict,Any,Tuple,List


class Config:
    def __init__(self,
            device: str = "cpu",
            dtype: torch.dtype = torch.float32,
            indim=32,
            hidden_layers=3,
            hidden_dim=16,
            compartments=3,
            t2_myelin=(10,25.0),
            t2_ie=(40,80),
            t2_csf=(500,1000),
            angle_boundaries=(90,180),
            epg_parameters={
                "nechos":32,
                "timeinit":5.5, #Tiempo de inicio de la secuencia en milisegundos
                "reptime":2000, #Tiempo de repetición de la secuencia en milisegundos
                "t1":1000
            },
            snr_range: Optional[Tuple[float, float]] = None
    ):
        self.device = device
        self.dtype = dtype
        self.indim = indim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.compartments = compartments
        self.t2_myelin = t2_myelin
        self.t2_ie = t2_ie
        self.t2_csf = t2_csf
        self.angle_boundaries = angle_boundaries
        self.epg_parameters = epg_parameters
        self.snr_range = snr_range

class TrainingConfig:
    def __init__(self,
                lr:float=0.001,
                batch_size=32,
                epochs=60,
                loss=nn.MSELoss,
                dtype=torch.float32,
                device=torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")):
        self.epochs=epochs
        self.lr=lr
        self.batch_size=batch_size
        self.loss=loss
        self.optimizer=torch.optim.SGD #Optimizador a utilizar 
        self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau #Scheduler a utilizar
        self.scheduler_kwargs={'mode':'min','factor':0.1}
        self.optimizer_kwargs={'lr':self.lr,'momentum':0.9}
        self.dtype=dtype
        self.device=device