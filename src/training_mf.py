import os
from keras.utils import Sequence
import numpy as np
from random import shuffle
from math import ceil
from keras.optimizers import Adam,SGD

# project modules
from . import data_preparation_mf as data_preparations 
from . import model_utils_mf as model_utils
from . import model_mf as model
from . import losses
from .. import config
from . import custom_metrics

# constant and path variables
epochs = 500
BATCH_SIZE = 100

#this sequence object will be prototype of generator that will deliver
#all training sample and labels for  one batch
class BPSequence(Sequence):
    #x is the list of training sequence index from processed_bp.csv
    def __init__(self,index_list,batch_size):
        self.index_list=index_list
        self.batch_size=batch_size

    def __len__(self):
        return int(np.floor(len(self.index_list) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_index = self.index_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        
        X,Y = data_preparations.prepare_batch(batch_index)

        return X,Y



##########################################test



   
#get all recods number
all_records = data_preparations.get_all_index()
#print("number of the protein sequence:", all_records)

#prepare list
all_index=[i for i in range(all_records)]


No = len(all_index)

#shuffle
shuffle(all_index)




#for training
no_train = int(np.ceil(No * 0.9))
print('number of training samples',no_train)


index_train = all_index[0:no_train]


#for validation
remaining = No-no_train
no_validation = remaining
print('number of validation samples',no_validation)

index_validation = all_index[no_train:(no_train + no_validation)]



#build model and compile



train_gen=BPSequence(index_train, BATCH_SIZE)
val_gen=BPSequence(index_validation, BATCH_SIZE)
steps_train=int(len(index_train) // BATCH_SIZE)
#print(steps_train)
steps_val=int(len(index_validation) // BATCH_SIZE)



model=model.build_model()
#model.load_weights(os.path.join(config.weight_path(),model_utils.MODEL_WEIGHT))


optimizer = Adam(lr=1e-5)
loss=losses.loss
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy', custom_metrics.f1_keras,custom_metrics.matthews_correlation])


print("complete model compiling")
#CALBACKS
history = model_utils.LossHistory()
#early_stopping = model_utils.set_early_stopping()
model_cp = model_utils.set_model_checkpoint()
reduce_lr = model_utils.set_reduce_lr()





#train
print("training starts....")
history = model.fit_generator(train_gen,
                            epochs = epochs,
                            steps_per_epoch = steps_train,
                            callbacks=[history,model_cp,reduce_lr],
                            validation_data = val_gen,
                            validation_steps=steps_val,
                            verbose = 2)




#saving
model_utils.save_model(model,model_utils.MODEL_NAME,model_utils.MODEL_WEIGHT)
