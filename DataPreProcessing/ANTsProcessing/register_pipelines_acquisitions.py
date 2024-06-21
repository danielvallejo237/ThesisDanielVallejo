# @Authors: Daniel Vallejo Aldana- Erick Canales - Alonso Ramírez (CIMAT,EPFL,CIMAT) 2024
# Contact: daniel.vallejo@cimat.mx
"""
This registration software recieves the different acquisitions of the same pre clinical data,
computes the mean of the acquisitions and registers them to correct for movement between echoes.
Using the computed registration matrices, it applies the same transformation to all acquisitions and saves the averaged acquisition.
To prevent a specific order in the averaging process, the software randomly selects a specified number of acquisitions.
"""

import nibabel as nib
import numpy as np
import os
import ants
from ants import from_numpy
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
    path,n_rep,max_iter=parser.parse_args().path,parser.parse_args().n_rep,parser.parse_args().max_iteraciones
    #lectura de los datos desde el folder que lo contiene
    files=[f for f in os.listdir(path) if f.endswith(".nii.gz")]
    loaded_files=[nib.load(os.path.join(path,f)) for f in files]
    denoised_files=[]
    print("Denoising files...")
    for i in tqdm(range(len(loaded_files))):
        denoised_files.append(mppca(loaded_files[i].get_fdata(),patch_radius=2))
    denoised_files=np.array(denoised_files)
    if n_rep>1:
        print("Selecting {} repetitions".format(n_rep))
        selected_files=denoised_files[np.random.choice(denoised_files.shape[0],n_rep,replace=False)]
    else:
        selected_files=denoised_files
    
    shape=loaded_files[0].get_fdata().shape
    error=float('inf')
    # Para usar el software de registro, es necesario crear antsimages
    ants_images=[ants.from_numpy(rep) for rep in selected_files]
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
            nib.save(nib.Nifti1Image(medias,loaded_files[0].affine),os.path.join(parser.parse_args().save_path,"test.nii.gz"))
            std=np.std(medias,axis=-1)
            error=std.mean()
            if error<0.11:
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
    print(corrected_images.shape)
    nib.save(nib.Nifti1Image(corrected_images,loaded_files[0].affine),os.path.join(parser.parse_args().save_path,"corrected_image.nii.gz"))
