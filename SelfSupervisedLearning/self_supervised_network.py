import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union,Optional,Dict,Any,Tuple,List
from epg_module import return_tensors,rician_noise,EPGModule,return_tensors_no_parallel
from utils import Config,TrainingConfig
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
import logging
from parameters import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
loger=logging.getLogger("torchepg")
# Definimos el nombre de la red como JointNetwork

class JointNetworkSingleHead(nn.Module):
    '''
        JointNetworkSingleHead: Red neuronal que toma como entrada un vector de características y devuelve un escalar de salida entre 0 y 1
        Este escalar de salida se utiliza para calcular algún valor del tipo T2 de parámetros de caída
    '''
    def __init__(self,indim:int,boundaries:tuple,*args,**kwargs):
        super(JointNetworkSingleHead, self).__init__()
        self.head=nn.Linear(in_features=indim,out_features=1,*args,**kwargs)
        self.boundaries=boundaries
        self.scale=(self.boundaries[1]-self.boundaries[0])/2.0
    def forward(self,x):
        multiplier=F.tanh(self.head(x)) + 1.0 # Lo anterior nos deja con un valor entre 0 y 2
        return self.boundaries[0]+ self.scale*multiplier

class JointNetworkNoConstraints(nn.Module):
    def __init__(self,indim:int,boundaries:tuple,*args,**kwargs):
        super(JointNetworkNoConstraints, self).__init__()
        self.head=nn.Linear(in_features=indim,out_features=1,*args,**kwargs)
        self.epsilon=1e-6
    def forward(self,x):
        salida=F.relu(self.head(x))+self.epsilon # Solo pedimos que sea mayor que 0 y que sea un valor positivo
        print(salida)
        return salida

# Un módulo que solamente regrese un tensor de tamaño del batch con valores fijos

class JointNetworkFixedValue(nn.Module):
    def __init__(self,value:float):
        super(JointNetworkFixedValue, self).__init__()
        self.value=torch.tensor(value)
    def forward(self,x):
        value=self.value.repeat(x.shape[0]) #regresamos un tensor a un valor fijo
        value=value.view(-1,1)
        return value

class JointNetworkMultiHead(nn.Module):
    '''
        JointNetworkMultiHead: Red neuronal que toma como entrada un vector de características y devuelve un vector de salida
        Este vector de salida se utiliza para calcular algún valor del tipo T2 de parámetros de caída
    '''
    def __init__(self,indim:int,outdim:int,*args,**kwargs):
        super(JointNetworkMultiHead, self).__init__()
        self.head=nn.Sequential(
            nn.Linear(indim,outdim,*args,**kwargs),
        )
    def forward(self,x):
        x=self.head(x)
        return F.softmax(x,dim=-1) # Softmax para obtener una distribución de probabilidad de cada uno de los diferentes compartimentos
    
class JointNetworkLayer(nn.Module):
    def __init__(self,indim:int,hidden_dim:int,*args,**kwargs):
        super(JointNetworkLayer, self).__init__()
        self.layer=nn.Sequential(
            nn.Linear(indim,hidden_dim,*args,**kwargs),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim), #Una normalización de las capas en cada módulo
        )
    def forward(self,x):
        return self.layer(x)

class JointNetworkBackBone(nn.Module):
    '''
        JointNetwork: Red neuronal que es el backbone de la red de detección de caídas
    '''
    def __init__(self,indim:int,hidden_dim:Union[int,List],num_layers:int,*args,**kwargs):
        super(JointNetworkBackBone, self).__init__()
        self.num_layers=num_layers if isinstance(num_layers,int) else len(num_layers)
        if isinstance(hidden_dim,List):
            self.layers=[]
            for i in range(self.num_layers):
                self.layers.append(JointNetworkLayer(indim,hidden_dim[i],*args,**kwargs))
                indim=hidden_dim[i]
            self.layers=nn.Sequential(*self.layers)
        else:
            self.layers=[]
            for i in range(self.num_layers):
                self.layers.append(JointNetworkLayer(indim,hidden_dim,*args,**kwargs))
                indim=hidden_dim
            self.layers=nn.Sequential(*self.layers) # Creamos un módulo secuencial con todas las capas que acabamos de definir
    def forward(self,x):
        return self.layers(x)

# Creación de la red neuronal en donde podemos fijar diferentes valores de T2 para cada uno de los compartimentos o pueden ser estimados por la red neuronal

class JointNetwork(nn.Module):
    '''
        JointNetwork: Red neuronal que toma como entrada un vector de características y devuelve un vector de salida
        Este vector de salida se utiliza para calcular algún valor del tipo T2 de parámetros de caída
    '''
    def __init__(self,indim:int,hidden_dim:Union[int,List],num_layers:int,num_compartments:int,isfixed:List,boundaries:List,*args,**kwargs):
        super(JointNetwork, self).__init__()
        self.backbone=JointNetworkBackBone(indim,hidden_dim,num_layers,*args,**kwargs)
        self.t2_heads=[]
        for i in range(num_compartments):
            if isinstance(hidden_dim,List):
                if isfixed[i] is None:
                    self.t2_heads.append(JointNetworkSingleHead(hidden_dim[-1],boundaries=boundaries[i],*args,**kwargs))
                else:
                    self.t2_heads.append(JointNetworkFixedValue(isfixed[i]))
            else:
                if isfixed[i] is None:
                    self.t2_heads.append(JointNetworkSingleHead(hidden_dim,boundaries=boundaries[i],*args,**kwargs))
                else:
                    self.t2_heads.append(JointNetworkFixedValue(isfixed[i]))
        self.t2_heads=nn.ModuleList(self.t2_heads)
        if isinstance(hidden_dim,List):
            self.weigths_heads=JointNetworkMultiHead(hidden_dim[-1],num_compartments,*args,**kwargs)
        else:
            self.weigths_heads=JointNetworkMultiHead(hidden_dim,num_compartments,*args,**kwargs)
    def forward(self,x):
        x=self.backbone(x)
        t2s=[head(x) for head in self.t2_heads]
        t2s=torch.stack(t2s,dim=-1).squeeze(1)
        weights=self.weigths_heads(x)
        return t2s,weights
    
############################################################################################################
class JointNetworkV2(nn.Module):
    def __init__(self,config):
        super(JointNetworkV2, self).__init__()
        self.config=config
        list_is_fixed=[None if isinstance(config.t2_myelin,tuple) else config.t2_myelin, None if isinstance(config.t2_ie,tuple) else config.t2_ie, None if isinstance(config.t2_csf,tuple) else config.t2_csf]
        self.network=JointNetwork(indim=config.indim,hidden_dim=config.hidden_dim,num_layers=config.hidden_layers,num_compartments=config.compartments,boundaries=[config.t2_myelin,config.t2_ie,config.t2_csf][:config.compartments],isfixed=list_is_fixed[:config.compartments])
        self.snr_range=config.snr_range
        self.epg=ExponentialModule(config)
    def forward(self,x,y):
        """
        x: corresponde a la señal de resonancia magnética 
        y: corresponde al ángulo de refocamiento, este se manda directamente a la función de retorno de tensores para no interferir en la función de reconstrucción de señal
        """
        t2s,weights=self.network(x)
        signals=self.epg(t2s,weights)
        if self.snr_range is not None:
            signals=rician_noise(signals,self.snr_range)
        return signals,weights

# Creamos un modelo donde la salida sean solamente los pesos y los t2 de los compartimentos y lo demás se maneje dentro de la función de pérdida 
class JointNetworkV2NoSignal(nn.Module):
    def __init__(self,config):
        super(JointNetworkV2NoSignal, self).__init__()
        self.config=config
        list_is_fixed=[None if isinstance(config.t2_myelin,tuple) else config.t2_myelin, None if isinstance(config.t2_ie,tuple) else config.t2_ie, None if isinstance(config.t2_csf,tuple) else config.t2_csf]
        self.network=JointNetwork(indim=config.indim,hidden_dim=config.hidden_dim,num_layers=config.hidden_layers,num_compartments=config.compartments,boundaries=[config.t2_myelin,config.t2_ie,config.t2_csf][:config.compartments],isfixed=list_is_fixed[:config.compartments])
    def forward(self,x,y):
        t2s,weights=self.network(x)
        return t2s,weights

class EPGLoss(_Loss):
    def __init__(self,config:Config,reduction:str='mean'):
        super(EPGLoss, self).__init__(reduction=reduction)
        self.config=config
        self.epg_function=return_tensors_no_parallel
    def forward(self,flip_angle,t2s,weight,targets):
        flip_angle=flip_angle.to(self.config.device)
        t2s=t2s.to(self.config.device)
        weight=weight.to(self.config.device)
        targets=targets.to(self.config.device)
        signals=self.epg_function(flip_angle,t2s,weight,self.config)
        return F.mse_loss(signals,targets,reduction=self.reduction)

class ExponentialModule(nn.Module):
    def __init__(self,config:Config):
        super(ExponentialModule,self).__init__()
        self.te=torch.tensor(np.array([TIME_INIT*i for i in range(1,N_ECHOS+1)])).view(1,-1)
        self.te=self.te.to(config.device,config.dtype)
    def forward(self,t2s,weights):
        # input de los t2s y de los pesos
        # t2s: Tensor de tamaño (batch_size,compartments)
        # weights: Tensor de tamaño (batch_size,compartments)
        # output: Tensor de tamaño (batch_size,32)
        t2s=t2s.squeeze(0)
        weights=weights.squeeze(0)
        te=self.te.repeat(t2s.shape[0],t2s.shape[-1],1).to(t2s.device)
        #t2s and weights are of size (batch_size,compartments) and must be the same shape as te
        t2s=t2s.unsqueeze(-1)
        weights=weights.unsqueeze(-1)
        t2s=t2s.repeat(1,1,te.shape[-1])
        weights=weights.repeat(1,1,te.shape[-1])
        te=torch.exp(-te/t2s)
        te=te*weights
        te=torch.sum(te,dim=1)
        # dividir entre el primer elemento para normalizar
        te=te/te[:,0].unsqueeze(-1)
        return te
    

# Crear un regularizador de los pesos 
class RegularizationLoss(_Loss):
    def __init__(self,lambda_l2=0.01,reduction:str='mean'):
        super(RegularizationLoss, self).__init__(reduction=reduction)
        self.lambda_l2 = lambda_l2
    def forward(self,outputs):
        return self.lambda_l2*F.mse_loss(outputs,torch.zeros_like(outputs),reduction=self.reduction)

def train_JointNetworkV2(model:nn.Module,config:TrainingConfig,train_dl:DataLoader,run_obj:Optional[object]=None):
    # Definimos un acelerador del modelo
    model.to(config.device)
    model.train()
    optimizer=config.optimizer(model.parameters(),**config.optimizer_kwargs)
    #scheduler=config.scheduler(optimizer,**config.scheduler_kwargs)
    loss_fn=config.loss()
    for epoch in range(config.epochs):
        running_loss=0.0
        for i,data in enumerate(tqdm(train_dl)):
            signal,angle=data
            signal=signal.to(config.device,dtype=config.dtype)
            angle=angle.to(config.device,dtype=config.dtype).view(-1,1)
            optimizer.zero_grad()
            outputs,_=model(signal,angle)
            loss=10000.*loss_fn(outputs,signal) #Hacemos una regularización de los pesos
            loss.backward()
            optimizer.step()
            #torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0,norm_type=2) # Clippeamos el gradiente para evitar que se dispare
            running_loss+=loss.item()
            #Evaluación del modelo para corroborar que la pérdida corresponde con la pérdida del entrenamiendo
        loger.info(f"Epoch {epoch+1} Training Loss: {running_loss/len(train_dl)}") #Calculamos las funciones de evaluación para cada uno de los conjuntos y ver si hay consistencia con los resultados obtenidos vs los resultados reportados
        if run_obj is not None:
            run_obj.log({"Loss Signal Reconstruction ":running_loss/len(train_dl)})
        #scheduler.step(running_loss/len(train_dl)) #Las metricas para reducir la tasa de entrenamiento 
    loger.info("Finished training")


def train_JointNetworkV2NoSignal(model:nn.Module,config:TrainingConfig,parameter_config:Config,train_dl:DataLoader,run_obj:Optional[object]=None):
    # Definimos un acelerador del modelo
    model.to(config.device)
    model.train()
    optimizer=config.optimizer(model.parameters(),**config.optimizer_kwargs)
    #scheduler=config.scheduler(optimizer,**config.scheduler_kwargs)
    loss_fn=EPGLoss(config=parameter_config)
    for epoch in range(config.epochs):
        running_loss=0.0
        for i,data in enumerate(tqdm(train_dl)):
            signal,angle=data
            signal=signal.to(config.device,dtype=config.dtype)
            angle=angle.to(config.device,dtype=config.dtype).view(-1,1)
            optimizer.zero_grad()
            t2s,weights=model(signal,angle)
            loss=10000.*loss_fn(angle,t2s,weights,signal) #Hacemos una regularización de los pesos
            #print(loss.item())
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0,norm_type=2) # Clippeamos el gradiente para evitar que se dispare
            running_loss+=loss.item()
            #Evaluación del modelo para corroborar que la pérdida corresponde con la pérdida del entrenamiendo
        loger.info(f"Epoch {epoch+1} Training Loss: {running_loss/len(train_dl)}") #Calculamos las funciones de evaluación para cada uno de los conjuntos y ver si hay consistencia con los resultados obtenidos vs los resultados reportados
        if run_obj is not None:
            run_obj.log({"Loss Signal Reconstruction ":running_loss/len(train_dl)})
        #scheduler.step(running_loss/len(train_dl)) #Las metricas para reducir la tasa de entrenamiento 
    loger.info("Finished training")

def predict_no_signal(model:nn.Module,config:Config,dloader:DataLoader):
    model.eval()
    with torch.no_grad():
        predicted_t2s=[]
        predicted_weights=[]
        for i,data in enumerate(tqdm(dloader)):
            signal,angle=data
            signal=signal.to(config.device,dtype=config.dtype)
            angle=angle.to(config.device,dtype=config.dtype).view(-1,1)
            t2,weights=model.network(signal)
            predicted_t2s.append(t2.cpu().numpy())
            predicted_weights.append(weights.cpu().numpy())
        predicted_t2s=np.concatenate(predicted_t2s,axis=0)
        predicted_weights=np.concatenate(predicted_weights,axis=0)
    signals=return_tensors_no_parallel(angle,predicted_t2s,predicted_weights,config)
    return signals,predicted_t2s,predicted_weights


def predict(model:nn.Module,config:Config,dloader:DataLoader):
    model.eval()
    with torch.no_grad():
        predicted_signals=[]
        predicted_t2s=[]
        predicted_weights=[]
        for i,data in enumerate(tqdm(dloader)):
            signal,angle=data
            signal=signal.to(config.device,dtype=config.dtype)
            angle=angle.to(config.device,dtype=config.dtype).view(-1,1)
            outputs,_=model(signal,angle)
            t2,weights=model.network(signal)
            predicted_signals.append(outputs.cpu().numpy())
            predicted_t2s.append(t2.cpu().numpy())
            predicted_weights.append(weights.cpu().numpy())
        predicted_signals=np.concatenate(predicted_signals,axis=0)
        predicted_t2s=np.concatenate(predicted_t2s,axis=0)
        predicted_weights=np.concatenate(predicted_weights,axis=0)
    return predicted_signals,predicted_t2s,predicted_weights