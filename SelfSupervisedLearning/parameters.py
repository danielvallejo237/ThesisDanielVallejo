# Parameters of the training and signal generation process for the self-supervised learning approach
# Description of the synthetic signals generation parameters for the signal generator
# Number of signals to generate
NUM_SIGNALS = 100
#Number of parameters to generate
NUM_PARMETERS = 2
T2_MYELIN=(10,25)
T2_IE=(30,70)
T2_CSF=(750,750.00001) # The T2 of the CSF is fixed
ANGLE_BOUNDARIES=(179.9999,180)
T1=1000
MYELIN_RANGE_CONSTRAIN=(0.01,0.4)
#Parameters of the acquisition sequence
TIME_INIT=5.5
REP_TIME=2000
N_ECHOS=32
#Prefix for the output files
PREFIX="signal"

#Training parameters
EPOCHS=10
LR=1e-5
T2_MYELIN_TRAINING=(10,25)
T2_IE_TRAINING=(30,70)
T2_CSF_TRAINING=750 #The value of the t2 compartment of the CSF is fixed to 750
ANGLE_BOUNDARIES_TRAINING=(179.9999,180)

#This value may be changed to None if the SNR is not going to be considered
SNRRANGE=None 