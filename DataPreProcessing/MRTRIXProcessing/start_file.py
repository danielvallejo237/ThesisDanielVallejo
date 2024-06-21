#@Author: Daniel Vallejo Aldana - CIMAT
#Contact: daniel.vallejo@cimat.mx

import nibabel as nib
import argparse
import os
import subprocess

parser=argparse.ArgumentParser(description="Dato de entrada para la obtención del header")
parser.add_argument("--file",type=str,help="Archivo de entrada que será usado para procesar")
parser.add_argument("--dim",type=str,help="Dimensión de la imagen de entrada")

def get_voxel_information(file:str):
    img=nib.load(file+".nii.gz")
    header=img.header['pixdim'][[3,2,1]]
    d1,d2,d3=header
    #exporting variable names to bash
    return str(d1),str(d2),str(d3)

if __name__=='__main__':
    args=parser.parse_args()
    env=os.environ.copy()
    d1,d2,d3=get_voxel_information(args.file)
    subprocess.run("bash run.sh {} {} {} {} {}".format(args.file,args.dim,d1,d2,d3),shell=True,env=env)
    
    
