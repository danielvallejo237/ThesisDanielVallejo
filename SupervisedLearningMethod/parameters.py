# Parameters for signal generation, training and testing of the supervised learning model
# @Author: Daniel Vallejo Aldana - CIMAT
# @Contact: daniel.vallejo@cimat.mx

# Synthetic data generation
# The different parameters used to generate the synthetic data
TISSUES={
    'PureWhiteMatter':{
        'TISSUE':{
            "MyelinWM": [10, 20, 1.0, 3.0, 30, 60, 3.0, 8.0, True], 
            "MyelinWMCSF": [10, 20, 1.0, 3.0, 30, 60, 3.0, 8.0, 500, 1000, 1.0, 8.0, True]},
        'DBSIZE':20000,
        'MYELINPERCENTAGE':[0.05,0.4]
    },
    'WhiteMatter':{
        'TISSUE':{"MyelinWM": [10, 20, 1.0, 3.0, 30, 60, 3.0, 8.0, True], 
                  "MyelinWMCSF": [10, 20, 1.0, 3.0, 30, 60, 3.0, 8.0, 500, 1000, 1.0, 8.0, True], 
                  "PV": [80, 450, 1.0, 8.0, False], 
                  "CSF": [500, 1000, 1.0, 8.0, False]},
        'DBSIZE':20000,
        'MYELINPERCENTAGE':[0.0,0.4]
    },
    'GrayMatter':{
        'TISSUE':{"MyelinGM": [10, 20, 1.0, 3.0, 35, 80, 3.0, 8.0, True],
                  "MyelinGMCSF": [10, 20, 1.0, 3.0, 35, 80, 3.0, 8.0, 500, 1000, 1.0, 8.0, True], 
                  "MyelinWMGM": [10, 20, 1.0, 3.0, 30, 60, 3.0, 8.0, 35, 80, 3.0, 8.0, True]},
        'DBSIZE':20000,
        'MYELINPERCENTAGE':[0.0,0.05]
    }
}
# The logarithmic scale of the signal
SCALE=10 #base 10 logarithm 
# The number of bins of the high resolution matrix
HRBINS=20000
# The number of bins of the low resolution matrix
LRBINS=60
SNRBOUNDS=[100000,1000001] 
# The range of the distribution of the T2 values
DISTRANGE=[10,1000]
# The number of threads where the program will run, this will speed up computations while loweing the CPU performance for other activities so be carefull when setting this parameter
WORKERS=-1 #We use all workers available in the CPU
#The EPG parameters according to real data
EPGPARAMETERS={
    'Tinit':5.677,
    'Spacing':5.677,
    'Echos':32,
    'Reptime':2000,
    'T1':1000,
    'RefocAnfleStart':90,
    'RefocAnfleEnd':180
}

# Training process configuration

#The number of ecpochs to train the model
EPOCHS=100
# The number of layers of the model
LAYERS=6
#The number of hidden units of the model
HIDDEN=256
#verbose
VERBOSE=1
#batch size
BATCHSIZE=2000

# Testing process configuration

RANGES={"MY": [10, 25],
        "IE": [26, 120], 
        "PTH": [120.1, 500], 
        "CSF": [500, 1000]}

