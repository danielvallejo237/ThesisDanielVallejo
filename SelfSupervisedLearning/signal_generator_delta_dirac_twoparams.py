# Generador de señales basado en el decoder del modelo de redes neuronales
# @Author: Daniel Vallejo // Ecole Polytechnique Fédérale de Lausanne (EPFL) - Centro de Investigación en Matemáticas A.C.

from epg_module import _return_tensors # Regresamos los tensores de las diferentes señales de resonancia magnética usando el decoder del modelo de redes neuronales previamente codificado
import torch
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import argparse
import os 
import nibabel as nib

# Modificamos las funciones anteriores para que regresen solamente dos compartimentos, el de la mielina y el del IntraExtra
def delta_dirac_parameters_generator_twoparams(
     t2_myelin:tuple,
        t2_ie:tuple,
        angle_boundaries:tuple,
        t1:float,
        myelin_range_constrain:tuple, # Rango de valores de la señal de mielina que en este caso no debe de sobrepasar el 0.4 
):
    t2_myelin=np.random.uniform(t2_myelin[0],t2_myelin[1])
    t2_ie=np.random.uniform(t2_ie[0],t2_ie[1])
    t2s=torch.tensor([t2_myelin,t2_ie])
    refoc_angle=torch.tensor([np.random.uniform(angle_boundaries[0],angle_boundaries[1])])
    myelin_weigth=np.random.uniform(myelin_range_constrain[0],myelin_range_constrain[1])
    ie_weight=1.0-myelin_weigth
    weights=torch.tensor([myelin_weigth,ie_weight]) #Removemos uno de los compartimentos, en este caso es el del CSF
    return refoc_angle,t2s,weights # Regresamos los tensores de los parámetros de la señal de resonancia magnética   

def Parallel_dirac_signal_parameters_generator_twoparams(
        N:int, #Numero de señales a ser generadas
        t2_myelin:tuple,
        t2_ie:tuple,
        angle_boundaries:tuple,
        t1:float,
        myelin_range_constrain:tuple, # R
):
    parameters=Parallel(n_jobs=-1)(delayed(delta_dirac_parameters_generator_twoparams)(t2_myelin,t2_ie,angle_boundaries,t1,myelin_range_constrain) for i in range(N))
    # stacking all the parameters into a tensor
    refoc_angles=torch.stack([i[0] for i in parameters],dim=0)
    t2s=torch.stack([i[1] for i in parameters],dim=0)
    weights=torch.stack([i[2] for i in parameters],dim=0)
    return refoc_angles,t2s,weights

parser=argparse.ArgumentParser(description='Generador de señales de resonancia magnética')
parser.add_argument('--N',type=int,default=100,help='Número de señales a ser generadas por lado')
parser.add_argument('--t2_myelin',nargs=2,type=float,default=(10,25),help='Rango de valores de T2 para la mielina')
parser.add_argument('--t2_ie',nargs=2,type=float,default=(30,70),help='Rango de valores de T2 para el ie')
parser.add_argument('--angle_boundaries',nargs=2,type=float,default=(179.9999,180),help='Rango de valores de los ángulos de refocamiento')
parser.add_argument('--myelin_range_constrain',nargs=2,type=float,default=(0.01,0.4),help='Rango de valores de la señal de mielina')
parser.add_argument('--timeinit',type=float,default=5.5,help='Tiempo de inicio de la secuencia en milisegundos')
parser.add_argument('--reptime',type=float,default=2000,help='Tiempo de repetición de la secuencia en milisegundos')
parser.add_argument('--nechos',type=int,default=32,help='Número de ecos en la secuencia')
parser.add_argument('--t1',type=float,default=1000,help='Valor de T1')
parser.add_argument('--prefix',type=str,default="signal",help='Prefijo para el nombre de los archivos de salida')
if __name__=="__main__":
    with torch.no_grad():
        arguments=parser.parse_args()
        refoc_angles,t2s,weights=Parallel_dirac_signal_parameters_generator_twoparams(
            arguments.N*arguments.N,
            arguments.t2_myelin,
            arguments.t2_ie,
            arguments.angle_boundaries,
            arguments.t1,
            arguments.myelin_range_constrain
        )
        parameters={
            'n':2,
            "nechos":arguments.nechos,
            "timeinit":arguments.timeinit,
            "reptime":arguments.reptime,
            "t1":arguments.t1
        }
        # Imprimir la shape de cada uno de los tensores generados
        signals=_return_tensors(refoc_angles,t2s,weights,parameters=parameters,device=torch.device('cpu'),dtype=torch.float32)
        if not os.path.exists(arguments.prefix):
            os.makedirs(arguments.prefix)
        signals_name=os.path.join(arguments.prefix,arguments.prefix+"_signals.npy")
        angles_name=os.path.join(arguments.prefix,arguments.prefix+"_angles.npy")
        weights_name=os.path.join(arguments.prefix,arguments.prefix+"_weights.npy")
        means_name=os.path.join(arguments.prefix,arguments.prefix+"_t2means.npy")
        np.save(signals_name,signals.numpy())
        np.save(angles_name,refoc_angles.numpy())
        np.save(weights_name,weights.numpy())
        np.save(means_name,t2s.numpy())
        # Hacer un reshape de las señales para poder guardarlas en archivo nifti de forma (x,y,z,n)
        signals=signals.reshape(arguments.N,arguments.N,1,arguments.nechos)
        signals_name_nifti=os.path.join(arguments.prefix,arguments.prefix+"_signals.nii")
        nib.save(nib.Nifti1Image(signals.numpy(),np.eye(4)),signals_name_nifti)
        # Crear una máscara para los datos que son putos unos
        mask=np.ones((arguments.N,arguments.N,1))
        mask_name=os.path.join(arguments.prefix,arguments.prefix+"_mask.nii")
        nib.save(nib.Nifti1Image(mask,np.eye(4)),mask_name)
        t2_my, t2_ie = t2s[:,0].numpy(), t2s[:,1].numpy()
        t2_myelin=t2_my.reshape(arguments.N,arguments.N,1)
        t2_ie=t2_ie.reshape(arguments.N,arguments.N,1)
        t2_myelin_name=os.path.join(arguments.prefix,arguments.prefix+"_t2myelin.nii")
        t2_ie_name=os.path.join(arguments.prefix,arguments.prefix+"_t2ie.nii")
        nib.save(nib.Nifti1Image(t2_myelin,np.eye(4)),t2_myelin_name)
        nib.save(nib.Nifti1Image(t2_ie,np.eye(4)),t2_ie_name)
        anngulos=refoc_angles.numpy()
        anngulos=anngulos.reshape(arguments.N,arguments.N,1)
        angles_name_nifti=os.path.join(arguments.prefix,arguments.prefix+"_angles.nii")
        nib.save(nib.Nifti1Image(anngulos,np.eye(4)),angles_name_nifti)
        print("Señales generadas y guardadas en el directorio {}".format(arguments.prefix))