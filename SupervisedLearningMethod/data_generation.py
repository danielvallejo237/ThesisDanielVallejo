# Generation of the synthetic signals to train the model of the thesis
# @Author: Daniel Vallejo Aldana - CIMAT
# Contact: daniel.vallejo@cimat.mx
from __future__ import division 
import numpy as np
import os
from parameters import *
from data_generation_utils import *

import logging

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

def generate_data(tissue,hrSequence,EPGLR,Tmat):
    # The tissue parameters, the hr sequence and the low resolution EPG matrix as well as the transformation matrix are required to generate the synthetic data
    parameters=tissue['TISSUE']
    dbsize=tissue['DBSIZE']
    myelinpercentage=tissue['MYELINPERCENTAGE']
    signals,distributions,angles,noise=GenNSignalsandDists(dbsize,
                                                           parameters,
                                                           hrSequence,
                                                           EPGLR,
                                                           Tmat,
                                                           snr_bounds=SNRBOUNDS,
                                                           num_workers=WORKERS,
                                                           scale=Log(SCALE)
                                                           ,rab=(EPGPARAMETERS['RefocAnfleStart'],EPGPARAMETERS['RefocAnfleEnd'])
                                                           ,closure=(0.0,2.4),
                                                           use_uniform=True,
                                                           myelinrange=myelinpercentage)
    signals=np.stack(signals,axis=0)
    distributions=np.stack(distributions,axis=0)
    return signals,distributions,angles

def main():
    logger.info("Strating the generation of the synthetic data")
    hrSequence=generateT2Sequence(DISTRANGE[0],DISTRANGE[1],resolution=HRBINS,scale=None)
    lrSequence=generateT2Sequence(DISTRANGE[0],DISTRANGE[1],resolution=LRBINS,scale=Log(SCALE))
    logger.info("Creating EPG High Resolution Matrix")
    EPG=generateCuboid(EPGPARAMETERS['RefocAnfleStart'],EPGPARAMETERS['RefocAnfleEnd'],HRBINS,hrSequence,EPGPARAMETERS['Echos'],EPGPARAMETERS['Tinit'],EPGPARAMETERS['Reptime'],EPGPARAMETERS['T1'])
    logger.info("Creating EPG Low Resolution Matrix")
    EPGLR=generateCuboid(EPGPARAMETERS['RefocAnfleStart'],EPGPARAMETERS['RefocAnfleEnd'],LRBINS,lrSequence,EPGPARAMETERS['Echos'],EPGPARAMETERS['Tinit'],EPGPARAMETERS['Reptime'],EPGPARAMETERS['T1'])
    logger.info("Creating Transformation Matrix")
    Tmat=BuildTransformMatrix(dist_range=DISTRANGE,second_resolution=LRBINS,scale=Log(SCALE))
    logger.info("Generating the synthetic data")
    logger.info("Generating Pure White Matter")
    signals_1,distributions_1,angles_1=generate_data(TISSUES['PureWhiteMatter'],hrSequence,EPGLR,Tmat)
    logger.info("Generating White Matter")
    signals_2,distributions_2,angles_2=generate_data(TISSUES['WhiteMatter'],hrSequence,EPGLR,Tmat)
    logger.info("Generating Gray Matter")
    signals_3,distributions_3,angles_3=generate_data(TISSUES['GrayMatter'],hrSequence,EPGLR,Tmat)
    signals=np.concatenate([signals_1,signals_2,signals_3],axis=0)
    distributions=np.concatenate([distributions_1,distributions_2,distributions_3],axis=0)
    angles=np.concatenate([angles_1,angles_2,angles_3],axis=0)
    logger.info("Saving the synthetic data")
    if not os.path.exists("data"):
        os.makedirs("data")
    np.save("data/signals.npy",signals)
    np.save("data/distributions.npy",distributions)
    np.save("data/angles.npy",angles)

if __name__ == "__main__":
    main()