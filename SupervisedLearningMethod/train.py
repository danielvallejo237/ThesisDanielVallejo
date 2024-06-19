import numpy as np
import tensorflow as tf
from train_utils import *
from parameters import *

def main(signals_file:str,distributions_file:str,angles_file:str):
    signals,distributions,angles=np.load(signals_file,allow_pickle=True),np.load(distributions_file,allow_pickle=True),np.load(angles_file,allow_pickle=True)
    signals=addRicianNoiseParallel(signals,SNRBOUNDS)
    #create the custom model
    model=create_custom_model(indim=EPGPARAMETERS['Echos'],
                              outdim=LRBINS,
                              use_softmax=True,
                              hidden_layers=LAYERS,
                              hidden_units=HIDDEN)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss=MSE_wasserstein_combo,metrics=['mse',wasserstein_distance])
    start=time.time()
    model.fit(signals,distributions,epochs=EPOCHS,verbose=VERBOSE,validation_split=0.1,batch_size=BATCHSIZE)
    end=time.time()
    print('TRAINING TIME: {} seconds'.format(end-start))
    if not os.path.exists('models'):
        os.mkdir('models')
    model.save('models/model.keras')

if __name__=='__main__':
    main('data/signals.npy','data/distributions.npy','data/angles.npy')