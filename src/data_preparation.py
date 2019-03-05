import pandas as pd
import os
import numpy as np
#import csv
from keras.utils import to_categorical
from keras.backend import int_shape
from .. import config


#CUT POINT

CUT_POINT=1000

#BP
TERM_PATH_BP=os.path.join(config.data_path(),"terms","bp_term_3rd.csv")
bb_terms_df=pd.read_csv(TERM_PATH_BP)
bp_terms=bb_terms_df['term_id'].tolist()


bp_terms_dict={}
bp_terms_dict={term:i for  i,term in enumerate(set(bp_terms))} 

NO_CLASSES=len(bp_terms_dict)

amino_path=os.path.join(config.data_path(),"a_code.csv")
amino_df=pd.read_csv(amino_path)
a_code=amino_df['code'].tolist()

#dic to convert letter to int
int_frm_letter={key:(value+1) for value,key in enumerate(a_code)}

letter_frm_int={(value+1):key for value,key in enumerate(a_code)}


def find_intersection(term_list):
    terms=[term.strip() for term in term_list.split(sep=",")]
    #print(terms)
    intersec=set(terms).intersection(set(bp_terms))
    return list(intersec)

def prepare_target_vector(existin_labels,all_target_dict):
#make a binary based target vector
#existing_labels=labels associated with current protien:list
#all_target_dict=all possible targets in a dictionary form:dictionary
    length_targets=len(all_target_dict)

    target_vector=[0]*length_targets
    
    label_list=find_intersection(existin_labels)

    for label in label_list:

        target_vector[all_target_dict.get(label)]=1

            
    return target_vector 


    
    
def prepare_one_sample(one_row):
    #prepare a sample for training
    #one_row=row containing sequence,go labels,length of protein
    #target_dict=dictionary containing all the target go id
        
    # unpack 
     labels=one_row['labels']#type str
     seq=one_row['seq']#type string
     #print(seq)

     target_vector=prepare_target_vector(labels,bp_terms_dict)



    #convert seq into index
     seq_index=[int(int_frm_letter.get(c)) for c in seq]
     leng=len(seq_index)
     #print(type(seq_index[1]))

     #PADDING OR TRIMMING
     if leng<CUT_POINT:
         zero_to_be_added=CUT_POINT-leng
         if zero_to_be_added==1:
             seq_one_hot=to_categorical(seq_index,num_classes=26)
             seq_one_hot=np.insert(seq_one_hot,0,[[0.0]*26]*1,axis=0)
         else:
             front_zero=zero_to_be_added//2
             end_zero=zero_to_be_added-front_zero

             seq_one_hot=to_categorical(seq_index,num_classes=26)
             
             #padding zero at first
             seq_one_hot=np.insert(seq_one_hot,0,[[0.0]*26]*front_zero,axis=0)
             
             #padding zero at last
             seq_one_hot=np.insert(seq_one_hot,seq_one_hot.shape[0],[[0.0]*26]*end_zero,axis=0)
             
     elif leng>CUT_POINT:
         seq_index=seq_index[0:(CUT_POINT)]
         seq_one_hot=to_categorical(seq_index,num_classes=26)
         
     else:
         seq_one_hot=to_categorical(seq_index,num_classes=26)
     
     

     return seq_one_hot,np.array(target_vector)


def prepare_batch(index_list):
    #this function returns numpy vector transformed seq,and associated labels
    #use prepare_one_sample

    #return np array of protein seq and np array of associated labels
    
    seq_arrays=[]
    target_arrays=[]
    for i in index_list:
        #print("now processing "+str(i) +"indexed sample")
        one_row=bp_db.loc[i,['labels','seq']]
        
        #prepare_one_sample(one_row)
        s,t=prepare_one_sample(one_row)
        seq_arrays.append(s)
        target_arrays.append(t)
    
    return np.array(seq_arrays),np.array(target_arrays)



def get_all_index():
#this function will return total number of record in processed_bp.csv file
#will be used for split whole data set into test an train
    BP_DATA_PATH=os.path.join(config.data_path(),"BP","processed_third_bp.csv")

    bps_db=pd.read_csv(BP_DATA_PATH)
    length=len(bps_db)
    
    del bps_db
    
    return length

def remove_obs_term_from_ancestor(ancestor):
    #remove the obsolete term s from ancestor list
    ancestor_list=[term.strip() for term in  ancestor.split(sep=",")]
    to_be_included=[term for term in ancestor_list if term not in obsolete_tirms]
    g=",".join(to_be_included)
    
    return g
    

#BP CATEGORY
BP_DATA_PATH=os.path.join(config.data_path(),"BP","processed_third_bp.csv")

bp_db=pd.read_csv(BP_DATA_PATH)


####################### TEST#####

one_row=bp_db.loc[100525,['labels','seq']]
print(one_row['seq'])
print(one_row['labels'])

s,t=prepare_one_sample(one_row)
print(s)
print(t.shape)
print(t)
print(np.where(t==1))
#testing

##all_index=[i for i in range(len(bp_db))]                     
##seq,tar=prepare_batch([88062])
##print(seq.shape)
##                  
##print(tar.shape)

##for i in range(len(bp_db)):
##    ancestor=bp_db.loc[i,'labels']
##    bp_db.loc[i,'l']=remove_obs_term_from_ancestor(ancestor)
###save
##BP_DATA_N_PATH=os.path.join(config.data_path(),"BP","processed_new_v1_bp.csv")
##bp_db.to_csv("a.csv", encoding='utf-8', index=False)
#print(obsolete_tirms)

##BP_OBS_PATH=os.path.join(config.data_path(),"BP","obsolete_go_terms.csv")
##resultFyle = open(BP_OBS_PATH,'wb')
##
### Create Writer Object
##wr = csv.writer(resultFyle, dialect='excel')
##
### Write Data to File
##for item in obsolete_tirms:
##    wr.writerow(item)
##print(a[2,100,:])
##print(b.shape)

#print(len(bp_db))

#print(get_all_index())


