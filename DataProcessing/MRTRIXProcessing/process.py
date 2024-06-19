#Toolbox para poder procesar los datos de ratón 
#@Author: Daniel Vallejo Aldana - CIMAT
#Contact: daniel.vallejo@cimat.mx

import nibabel as nib
import nibabel.orientations as nio 
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Datos de entrada para el procesamiento de los datos')
parser.add_argument('--path', type=str, help='Path de la imagen')
parser.add_argument('--echos', type=int, help='Número de echos')
parser.add_argument('--repetitions', type=int, help='Número de repeticiones')
def load_and_transform(path:str):
    '''
    Cargamos la imagen y la estandarizamos para poderla procesar posteriormente
    '''
    img = nib.load(path) 
    return img

def split_image(path:str,echos:int,repetitions:int):
    '''
    Dividimos la imagen en los diferentes echos y repeticiones
    '''
    img = load_and_transform(path)
    current_path=os.getcwd()
    if not os.path.exists(os.path.join(current_path,path.split('.')[0])):
        os.makedirs(os.path.join(current_path,path.split('.')[0]))
    pathname=os.path.join(current_path,path.split('.')[0])
    images=[] #Creamos una lista que contiene las imágenes de cada echo 
    for i in range(echos):
        if (i+1)>=10:
            name=path.split('.')[0]+'_echo_'+str(i+1)+'.nii.gz'
        else:
            name=path.split('.')[0]+'_echo_0'+str(i+1)+'.nii.gz'
        images.append(img.dataobj[:,:,:,i*repetitions:(i+1)*repetitions])
        nib.save(nib.Nifti1Image(images[i],img.affine,img.header),os.path.join(pathname,name))

if __name__=='__main__':
    args = parser.parse_args()
    split_image(args.path,args.echos,args.repetitions)

