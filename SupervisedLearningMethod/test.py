# Inference file using a single model and the flip angle model 
# @Author: Daniel Vallejo Aldana - CIMAT
# @Contact: daniel.vallejo@cimat.mx

from parameters import *
from train_utils import *
import tensorflow as tf
import numpy as np
import nibabel as nib
from typing import Optional
import argparse

def lock_and_load(volume:str,mask:Optional[str]=None):
    vol=nib.load(volume)
    if mask != '':
        mask=nib.load(mask)
        return vol,mask
    return vol,None

def process_volume(vol:np.ndarray,mask:Optional[np.ndarray]=None):
    if mask is not None:
        for i in range(vol.shape[-1]):
            vol[...,i]=np.multiply(vol[...,i],mask)
    return vol

def return_sequence():
    return np.logspace(np.log10(DISTRANGE[0]),np.log10(DISTRANGE[1]),num=LRBINS,base=SCALE)

def find_indices(sequence:np.ndarray):
    pars=RANGES
    indices={}
    indices['MY']=np.where(np.logical_and(sequence>=pars['MY'][0],sequence<=pars['MY'][1]))[0]
    indices['IE']=np.where(np.logical_and(sequence>=pars['IE'][0],sequence<=pars['IE'][1]))[0]
    indices['PTH']=np.where(np.logical_and(sequence>=pars['PTH'][0],sequence<=pars['PTH'][1]))[0]
    indices['CSF']=np.where(np.logical_and(sequence>=pars['CSF'][0],sequence<=pars['CSF'][1]))[0]
    return indices

def compute_volume_fraction(distribution:np.ndarray,indices:dict):
    volume_fraction={}
    volume_fraction['MY']=np.sum(distribution[:,indices['MY']],axis=-1)
    volume_fraction['IE']=np.sum(distribution[:,indices['IE']],axis=-1)
    volume_fraction['PTH']=np.sum(distribution[:,indices['PTH']],axis=-1)
    volume_fraction['CSF']=np.sum(distribution[:,indices['CSF']],axis=-1)
    return volume_fraction

def main(input_volume:str,mask:Optional[str]=None):
    vol,mask=lock_and_load(input_volume,mask)
    vol=process_volume(vol,mask)
    model=tf.keras.models.load_model('models/model.keras',custom_objects={'wasserstein_distance':wasserstein_distance,'MSE_wasserstein_combo':MSE_wasserstein_combo})
    angle_model=tf.keras.models.load_model('models/angle_model.keras')
    signals=vol.get_fdata()
    shape=signals.shape
    signals=signals.reshape(-1,signals.shape[-1])
    distributions=model.predict(signals)
    angles=angle_model.predict(signals)
    sequence=return_sequence()
    indices=find_indices(sequence)
    volume_fractions=compute_volume_fraction(distributions,indices)
    mwf_volume=volume_fractions['MY'].reshape(shape[:-1]+(1,))
    ie_volume=volume_fractions['IE'].reshape(shape[:-1]+(1,))
    pth_volume=volume_fractions['PTH'].reshape(shape[:-1]+(1,))
    csf_volume=volume_fractions['CSF'].reshape(shape[:-1]+(1,))
    all_volumes=np.concatenate((mwf_volume,ie_volume,pth_volume,csf_volume),axis=-1)
    name=input_volume.split('/')[-1]
    nib.save(nib.Nifti1Image(all_volumes,vol.affine),input_volume.replace('.nii.gz','_predicted_volumes.nii.gz'))
    angle_volume=angles.reshape(shape[:-1]+(1,))
    nib.save(nib.Nifti1Image(angle_volume,vol.affine),input_volume.replace('.nii.gz','_predicted_angles.nii.gz'))

parser=argparse.ArgumentParser()
parser.add_argument('--input_volume',type=str,help='The input volume to process')
parser.add_argument('--mask',type=str,default="",help='The mask to apply to the volume')

if __name__=='__main__':
    args=parser.parse_args()
    main(args.input_volume,args.mask)