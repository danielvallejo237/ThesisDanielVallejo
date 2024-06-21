# Archivo de registro de los datos para concatenar todas las repeticiones
# @Author Daniel Vallejo Aldana - Erick Canales - Alonso Ramírez (CIMAT,EPFL,CIMAT) 2023
# Contact: daniel.vallejo@cimat.mx
"""
This registration software receives the path containing files of various echoes from the same pre-clinical data,
merges them into different acquisitions for the same patient, and then applies the MP-PCA denoising algorithm. 
Subsequently, it registers the means of these different acquisitions to correct for movement between echoes. 
Using the computed registration matrices, it applies the same transformation to all acquisitions and saves the averaged acquisition. 
To prevent a specific order in the averaging process, the software randomly selects a specified number of acquisitions.
"""

import nibabel as nib
import numpy as np
import os
import ants
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import preprocessing as pre
from dipy.denoise.localpca import mppca
import time
from tqdm import tqdm
import argparse 

parser=argparse.ArgumentParser(description='Script para concatenar las repeticiones de un mismo paciente (Ratón)')
parser.add_argument('--path',dest='path',type=str,help='Path donde se encuentran los datos a concatenar')
parser.add_argument('--n_rep',dest='n_rep',type=int,help='Número de repeticiones a concatenar') #Esto incrementa o decrementa la resolución de la estimación de las imágenes
parser.add_argument('--max_iteraciones',dest='max_iteraciones',type=int,help='Número máximo de iteraciones para el registro',default=200)
parser.add_argument('--save_path',dest='save_path',type=str,help='Path donde se van a guardar los datos')
if __name__=='__main__':
    path, n_rep,max_iter = parser.parse_args().path, parser.parse_args().n_rep,parser.parse_args().max_iteraciones
    # Módulo 1: Lectura de los diferentes tiempos echo y juntarlos dentro de un conjunto de repeticiones
    save_path=parser.parse_args().save_path
    all_repetitions,affines,headers=[],[],[]
    files=os.listdir(path)
    #Abrimos alguno de los archivos para obtener la información de la cabecera
    tmp=nib.load(os.path.join(path,files[0]))
    #Obtenemos la última dimensión de los datos
    last_dim=tmp.shape[-1]
    print("Number of volumes in path: {}".format(last_dim))
    #Obtenemos el numero de elementos que terminen con .nii.gz
    files=[i for i in files if i.endswith(".nii.gz")]
    for desired_volume in tqdm(range(last_dim)):
        fdatas=[]
        affine=None
        header=None
        for i in range(1,33):
            if i<10:
                name=os.path.join(path,path.split("/")[-1]+"_echo_"+'0'+str(i)+".nii.gz")
            else:
                name=os.path.join(path,path.split("/")[-1]+"_echo_"+str(i)+".nii.gz")
            #open_volumes
            vol=nib.load(name)
            if i==1:
                affine=vol.affine
                affines.append(affine)
                header=vol.header
                headers.append(header)
            fdatas.append(vol.get_fdata()[:,:,:,desired_volume])
        recovered_volume=np.stack(fdatas,axis=-1)
        #Aplicamos el algoritmo de denoising de mppca de dipy
        recovered_volume=mppca(recovered_volume, patch_radius=2)
        all_repetitions.append(recovered_volume)
        # Módulo 1.2: Escogemos aleatoriamente n_repeticiones que son las que vamos a concatenar
    all_repetitions=np.array(all_repetitions)
    # Escogemos aleatoriamente n_repeticiones
    all_repetitions=all_repetitions[np.random.choice(all_repetitions.shape[0],n_rep,replace=False)]
    # regresamos a lista
    all_repetitions=list(all_repetitions)
    # Módulo 2: Concatenación de las repeticiones, registo de las imágenes y guardado de las mismas
    # Vamos a minimizar el error de concatenación, por lo tanto es necesario fijarlo a infinito
    error=float('inf')
    # Para usar el software de registro, es necesario crear antsimages
    ants_images=[ants.from_numpy(rep) for rep in all_repetitions]
    means=[ants.from_numpy(a.mean(axis=-1)) for a in ants_images]
    # Vamos a usar el primer volumen como referencia
    iterations=0
    with tqdm(total=max_iter) as pbar:
        pbar.set_description("Iterations")
        for _ in range(max_iter):
            registers=[]
            transformations=[]
            for i in range(1,len(means)):
                info=ants.registration(means[0],means[i],type_of_transform='BOLDRigid',
                                    verbose=False,
                                    grad_step=5e-4,
                                    aff_random_sampling_rate=0.7,
                                    aff_iterations=(3300, 2300, 2300, 10),
                                    multivariate_extras=("MeanSquares",means[0],means[i],0.5,0)
                                    )
                registers.append(info['warpedmovout'])
                transformations.append(info['fwdtransforms'])
            # Prueba de control de calidad
            medias=[means[0]]+registers
            medias=[m.numpy() for m in medias]
            medias=np.stack(medias,axis=-1)
            std=np.std(medias,axis=-1)
            error=np.mean(std.reshape(-1))
            if error<0.07:
                break
            iterations+=1
            pbar.update(1)
    print("Minimum Error: {}".format(error))
    # Módulo 3: Aplicar la transformación a todas las imágenes y guardarlas
    # Ahora que tenemos las matrices de transformación se le aplica a cada uno de los diferentes volúmenes
    corrected_images=[]
    corrected_images.append(ants_images[0])
    for i in range(1,len(ants_images)):
        corrected_image=[]
        for k in range(corrected_images[0].shape[-1]):
            transformated=ants.apply_transforms(fixed=ants.from_numpy(medias[...,0]),moving=ants.from_numpy(ants_images[i].numpy()[...,k]),transformlist=transformations[i-1])
            corrected_image.append(transformated.numpy())
        corrected_image=np.stack(corrected_image,axis=-1)
        corrected_images.append(ants.from_numpy(corrected_image))
    # Módulo 4: Promediado de las imágenes y guardado de las mismas en el nombre del archivo que se desee
    corrected_images=[c.numpy() for c in corrected_images]
    corrected_images=np.stack(corrected_images,axis=-1)
    corrected_images=np.mean(corrected_images,axis=-1)
    nib.save(nib.Nifti1Image(corrected_images,affine,header),os.path.join(save_path,"{}_averaged_{}_reps.nii.gz".format(path.split("/")[-1],n_rep)))