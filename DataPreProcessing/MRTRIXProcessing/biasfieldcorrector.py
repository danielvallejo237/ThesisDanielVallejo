#Archivo de corrección de biasfield
#@Author: Daniel Vallejo Aldana - CIMAT
#Contact: daniel.vallejo@cimat.mx

import numpy as np
import nibabel as nib
import argparse


parser = argparse.ArgumentParser(description='Datos de entrada para el procesamiento de los datos')
parser.add_argument('--path', type=str, help='Path de la imagen')
parser.add_argument('--bffile', type=str, help='Path del archivo de biasfield') #Archivo de biasfield

def load_files(path:str,bffile:str):
    '''
    Cargamos la imagen y el archivo de biasfield
    '''
    img = nib.load(path) 
    bf = nib.load(bffile)
    return img,bf

def process_biasfield(path:str,bffile:str):
    '''
    Procesamos el biasfield para poder corregir la imagen
    '''
    img,bf=load_files(path,bffile)
    img_data=img.get_fdata()
    bf_data=bf.get_fdata()
    for i in range(img_data.shape[3]):
        img_data[:,:,:,i]=np.multiply(img_data[:,:,:,i],1./bf_data)
    return nib.Nifti1Image(img_data,img.affine,img.header)

if __name__=='__main__':
    args = parser.parse_args()
    img=process_biasfield(args.path,args.bffile)
    nib.save(img,args.path)