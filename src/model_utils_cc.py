
from keras.callbacks import EarlyStopping,Callback,ModelCheckpoint,ReduceLROnPlateau
from keras.models import model_from_json
from keras.callbacks import TensorBoard
"""this file contains code for handling  model utilities"""

""" Notes:
# all models are palced and stored in output directory
# model checkpoint are saved in weight directory
"""

# python modules
from keras.callbacks import (EarlyStopping,
                             Callback,
                             ModelCheckpoint,
                             ReduceLROnPlateau)

import numpy
import os

# project modules
from .. import config
#print("saiful")

############################################################
#          BP
##########################################################
#model name

MODEL_NAME="cc_model1.json"
MODEL_WEIGHT="weight_cc_model1_mcc.h5"





def save_model(model,model_name,weight_name):

    """
    save model to output directory
    """
    
    #serialize the model to json
    model_json=model.to_json()
    #write
    with open(os.path.join(config.output_path(),model_name), "w") as json_file:
        json_file.write(model_json)
        
    print("model saved")

    #save the weight
    #serialize weights to HDF5

    model.save_weights(os.path.join(config.output_path(),weight_name))
    print("weight saved")
    
def load_model_only(model_name):
    """
    load a model from output directory by name
    """
    json_file=open(os.path.join(config.output_path(),model_name))
    loaded_json_file=json_file.read()
    json_file.close()
    loaded_model=model_from_json(loaded_json_file)
    
    return loaded_model

##def load_model(model_name,weight_path):
##
##    model.save_weights(os.path.join(config.output_path(), MAIN_MODEL_WEIGHT))
##    print("weight saved")


def load_model(model_name,weight_name):

    """
    load a model from output directory by name
    """
    json_file=open(os.path.join(config.output_path(),model_name))
    loaded_json_file=json_file.read()
    json_file.close()
    loaded_model=model_from_json(loaded_json_file)
    print("model loaded")
    loaded_model.load_weights(os.path.join(config.weight_path(),weight_name))
    
    print("weight loaded")
    
    return loaded_model


def save_model_only(model,string):
    model_json=model.to_json()
    with open(os.path.join(config.output_path(),string),"w") as json_file:
        json_file.write(model_json)
        print("model saved")

#**************** Callbacks**************
class LossHistory(Callback):
    def on_train_begin(self,logs={}):
        self.losses=[]
        self.val_losses=[]
        
    def on_epoch_end(self,batch,logs={}):
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))



#for early stopping if no improvement within patience batches occured
def set_early_stopping():
    return EarlyStopping(monitor="val_acc",
                         patience = 5,
                         mode = "auto",
                         verbose = 2)



def set_model_checkpoint():

    return ModelCheckpoint(os.path.join(config.weight_path(),MODEL_WEIGHT),

                monitor = 'val_f1_keras',
                verbose = 2,
                save_best_only = True,
                save_weights_only = True,
                mode = 'max',
                period = 1)


def set_tensorboard(path):
    return TensorBoard(log_dir=config.log_path(), histogram_freq=0,  write_graph=True, write_images=False)


def set_reduce_lr():
    return ReduceLROnPlateau(monitor='val_f1_keras',
                             factor = 0.5,
                             patience = 4,
                             min_lr = 1e-5)









        



















        
    
