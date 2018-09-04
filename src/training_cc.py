import os
from keras.utils import Sequence
import numpy as np
from random import shuffle
from math import ceil
from keras.optimizers import Adam,SGD

# project modules
from . import data_preparation_cc as data_preparations 
from . import model_cc as model
from . import losses
from . import model_utils_cc as model_utils
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
        #target_index = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        X,Y = data_preparations.prepare_batch(batch_index)

        return X,Y



##########################################test

#X,Y=data_preparations.get_all_sample()

   
#get all recods number
all_records = data_preparations.get_all_index()
#print("number of the protein sequence:", all_records)

#prepare list
all_index=[i for i in range(all_records)]
#print(all_index)

No = len(all_index)

#shuffle
shuffle(all_index)


####### pilot batch 100 train ,100 test
#batch_train = all_index[0:100]
#batch_validation=all_index[100:200]

#for training
no_train = int(np.ceil(No * 0.9))
print('number of training samples',no_train)


index_train = all_index[0:no_train]
##print(len(index_train))
#index_train=math.ceil(
#print(len(index_train))

#for validation
remaining = No-no_train
no_validation = remaining
print('number of validation samples',no_validation)

index_validation = all_index[no_train:(no_train + no_validation)]

'''
#print(len(index_validation))



#for test
#no_test=remaining-no_validation
#index_test=all_index[(no_train+no_validation):No]
#print(len(index_test))
##X,Y=data_preparation.prepare_batch(index_train)
##
##print(X.shape)
##print(Y.shape)


##print(len(index_validation))
##print(len(index_test))


#train_gen=BPSequence(index_train, BATCH_SIZE)
#val_gen=BPSequence(index_validation, BATCH_SIZE)
##x,y=train_gen. __getitem__(1)
##print(x.shape)
##print(y.shape)
#print(train_gen.__len__())
'''

'''
for i in range(10):
    x, y = train_gen.__getitem__(i+1)  
    print("train shape:", x.shape)
    print("label shape: ", y.shape)
'''

#build model and compile



train_gen=BPSequence(index_train, BATCH_SIZE)
val_gen=BPSequence(index_validation, BATCH_SIZE)
steps_train=int(len(index_train) // BATCH_SIZE)
#print(steps_train)
steps_val=int(len(index_validation) // BATCH_SIZE)


"""

###print(steps_val)
for i in range(steps_train):
   x,y=train_gen.__getitem__(i)
    
   print(x.shape)
   print(y.shape)



##print("training sample generation complete")
###print(steps_val)
for i in range(steps_val):
    
    x,y=val_gen.__getitem__(i)
    print(x.shape)    
    print(y.shape)



"""
model=model.build_model()
#model.load_weights(os.path.join(config.weight_path(),model_utils.MODEL_WEIGHT))

#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
optimizer = Adam(lr=1e-5)
loss=losses.loss
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy', custom_metrics.f1_keras,custom_metrics.matthews_correlation])

#model.summary()
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

'''
history=model.fit(x=X,
                  y=Y,
                  batch_size=BATCH_SIZE,
                  epochs=epochs,
                  validation_split=0.2,
                  callbacks=[history,model_cp,reduce_lr])
'''
#saving
model_utils.save_model(model,model_utils.MODEL_NAME,model_utils.MODEL_WEIGHT)
