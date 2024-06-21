import os 
from parameters import *

def data_generation(parameters:int):
    if parameters==2:
        os.system("python signal_generator_delta_dirac_twoparams.py --N {} --t2_myelin {} {} --t2_ie {} {} --angle_boundaries {} {} --myelin_range_constrain {} {} --timeinit {} --reptime {} --nechos {} --t1 {} --prefix {}".format(NUM_SIGNALS,T2_MYELIN[0],T2_MYELIN[1],T2_IE[0],T2_IE[1],ANGLE_BOUNDARIES[0],ANGLE_BOUNDARIES[1],MYELIN_RANGE_CONSTRAIN[0],MYELIN_RANGE_CONSTRAIN[1],TIME_INIT,REP_TIME,N_ECHOS,T1,PREFIX))
    elif parameters==3:
        os.system("python signal_generator_delta_dirac.py --N {} --t2_myelin {} {} --t2_ie {} {} --t2_csf {} {} --angle_boundaries {} {} --myelin_range_constrain {} {} --timeinit {} --reptime {} --nechos {} --t1 {} --prefix {}".format(NUM_SIGNALS,T2_MYELIN[0],T2_MYELIN[1],T2_IE[0],T2_IE[1],T2_CSF[0],T2_CSF[1],ANGLE_BOUNDARIES[0],ANGLE_BOUNDARIES[1],MYELIN_RANGE_CONSTRAIN[0],MYELIN_RANGE_CONSTRAIN[1],TIME_INIT,REP_TIME,N_ECHOS,T1,PREFIX))
    else:
        raise ValueError("The number of parameters is not valid")
    
if __name__=="__main__":
    data_generation(NUM_PARMETERS)