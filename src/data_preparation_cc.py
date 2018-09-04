import pandas as pd
import os
import numpy as np
#import csv
from keras.utils import to_categorical
from keras.backend import int_shape
from .. import config
import json
################# CONSTANTS##################
#CUT POINT

CUT_POINT = 1000
SAMPLE_CSV_FILE = "processed_cc.csv"

TARGET_CSV_FILE = "cc_first_level_ancestors.csv"
TARGET_DICT= "cc_first_level.txt"
TERMS_DICT = {}

AMINO_ENCODE_CSV_FILE = "a_code.csv"
AMINO_ACID_CHR_TO_INT_FILE='amino_acid_chr_to_int.txt'
AMINO_ACID_CHR_TO_INT_DICT={}


GO_TYPE = "CC"


############### TARGET AND AMINO ACID ENCODING PREPARATION##########

#TARGET

TERM_CSV_PATH = os.path.join(config.data_path(),"terms",TARGET_CSV_FILE)




terms_df = pd.read_csv(TERM_CSV_PATH)


terms_ls = terms_df['term_id'].tolist()
#print(terms_ls)
#print(len(terms_ls))



#Write the dictionary containing element  in term_ls

#TERMS_DICT={term:i for  i,term in enumerate(set(terms_ls))}
#print(TERMS_DICT)

'''
#exDict = {'exDict': TERMS_DICT}
with open((os.path.join(config.data_path(),"terms",TARGET_DICT)), 'w') as file:
    file.write(json.dumps(TERMS_DICT))
'''

#load target_dictionary
with open((os.path.join(config.data_path(),"terms",TARGET_DICT)), 'r') as file:
    data=json.load(file)
    #TERMS_DICT=data['exDict']
    TERMS_DICT=data
        
#print(TERMS_DICT) 


NO_CLASSES=len(TERMS_DICT)
#print(NO_CLASSES)


####AMINO ACID 



#LOAD amino acid dictionary

with open((os.path.join(config.data_path(),"terms",AMINO_ACID_CHR_TO_INT_FILE)), 'r') as file:
    data=json.load(file)
    #TERMS_DICT=data['exDict']
    AMINO_ACID_CHR_TO_INT_DICT = data




########################################################## UTILITY FUNCTIONS########
def find_intersection(term_list):

    terms=[term.strip() for term in term_list.split(sep=",")]
    #print(terms)
    intersec=set(terms).intersection(set(terms_ls))
    return list(intersec)



def remove_obs_term_from_ancestor(ancestor):
    #remove the obsolete term s from ancestor list
    ancestor_list=[term.strip() for term in  ancestor.split(sep=",")]
    to_be_included=[term for term in ancestor_list if term not in obsolete_tirms]
    g=",".join(to_be_included)
    
    return g

########################### TRAINING SAMPLE PREPARATION######################

def prepare_target_vector(existin_labels,all_target_dict):
#make a binary based target vector
#existing_labels=labels associated with current protien:list
#all_target_dict=all possible targets in a dictionary form:dictionary
    length_targets=len(all_target_dict)

    target_vector=[0]*length_targets
    
    label_list=find_intersection(existin_labels)
    #print("ancestors are ")
    #print(label_list)
    for label in label_list:

        target_vector[all_target_dict.get(label)]=1
##        
####        try:
####            target_vector[all_target_dict.get(label)]=1
####        except IndexError:
####            #obsolete_tirms.append(label)
####            pass
####        except TypeError:
####            #obsolete_tirms.append(label)
####            pass
            
    return target_vector 


    
    
def prepare_one_sample(one_row):
    #prepare a sample for training
    #one_row=row containing sequence,go labels,length of protein
    #target_dict=dictionary containing all the target go id
        
    # unpack 
     labels=one_row['labels']#type str
     seq=one_row['seq']#type string
     #print(seq)

     target_vector=prepare_target_vector(labels,TERMS_DICT)



    #convert seq into index
     seq_index=[int(AMINO_ACID_CHR_TO_INT_DICT.get(c)) for c in seq]
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
    #sample_list=[DATA.loc[i,['labels','seq']] for i in index_list]
    seq_arrays=[]
    target_arrays=[]
    for i in index_list:
        #print("now processing "+str(i) +"indexed sample")
        try:
            one_row=DATA.loc[i,['labels','seq']]
        
            #prepare_one_sample(one_row)
            s,t=prepare_one_sample(one_row)
            seq_arrays.append(s)
            target_arrays.append(t)
        except KeyError:
            pass
        
    #seq_arrays,target_arrays=[prepare_one_sample(one_row) for one_row in sample_list]
    return np.array(seq_arrays),np.array(target_arrays)






    
######################### INITIALIZATION
# LOAD CSV FILE CONTAINING DATA

DATA_PATH=os.path.join(config.data_path(),GO_TYPE,SAMPLE_CSV_FILE)

DATA=pd.read_csv(DATA_PATH)

def get_all_index():
#this function will return total number of record in processed_bp.csv file
#will be used for split whole data set into test an train
    #BP_DATA_PATH=os.path.join(config.data_path(),GO_TYPE,SAMPLE_CSV_FILE)

    #bps_db=pd.read_csv(BP_DATA_PATH)
    length=len(DATA.index)
    
    #del bps_db
    
    return length


def get_all_sample():
    total=get_all_index()
    seq_arrays=[]
    target_arrays=[]
    for i in range(total):
        #print("now processing "+str(i) +"indexed sample")
        one_row=DATA.loc[i,['labels','seq']]
        
        #prepare_one_sample(one_row)
        s,t=prepare_one_sample(one_row)
        seq_arrays.append(s)
        target_arrays.append(t)
    #seq_arrays,target_arrays=[prepare_one_sample(one_row) for one_row in sample_list]
    return np.array(seq_arrays),np.array(target_arrays)    


def combine_all_ancestor_for_one_protein(protein_id):
    protein_df = DATA.loc[DATA['ID']==protein_id]
    go_label_ls = protein_df['labels'].tolist()
    first_level_ls=[]
    for label in go_label_ls:
        first_level_ls.append(find_intersection(label))
    #flatten the 2d list
    flatten_ancestor_ls=[j for sub in first_level_ls for j in sub]
    return ",".join(list(set(flatten_ancestor_ls)))
      
def preprocess_data_set(old_df):
    protein_id_ls = list(set(old_df['ID'].tolist()))
    new_data_ls=[]
    for protien in protein_id_ls:
        one_sample_ls=[]
        first_level_ancestor_ls = combine_all_ancestor_for_one_protein(protien)

        protien_rows_df = old_df.loc[old_df['ID']==protien]
        seq=old_df.loc[protien_rows_df.index[0],'seq']
        
        one_sample_ls.append(protien)
        one_sample_ls.append(seq)
        one_sample_ls.append(first_level_ancestor_ls)

        new_data_ls.append(one_sample_ls)

    names=["protien_id","seq","labels"]

    return pd.DataFrame(new_data_ls,columns=names)
        

#print(get_all_index())

'''
#Aggregate Data
DATA= preprocess_data_set(DATA)

DATA.to_csv(os.path.join(config.data_path(),GO_TYPE,SAMPLE_CSV_FILE),sep=",")
'''


"""
#print("data_preparation_bp_leve1 is executed")
#print(DATA.columns)
####################### TEST#####


# exploring dataset

#how many columns
#print(DATA.columns)


#Unique IDS
#uniques_ids=bp_d

#what are the rows corresponding to same ID
#combine_all_ancestor_for_one_protein('O95497')
#print('old dataframe length',len(DATA))
#print('new data frame length',preprocess_data_set(DATA))
#samples_ID=DATA.loc[DATA['ID']=='Q9Y2P4']
#print(DATA.loc[samples_ID.index[0],'seq'])

#print(type(samples_ID))
#print(len(samples_ID))
#print(samples_ID)

#new_df = preprocess_data_set(DATA)
#print("new data set length",len(new_df))

#grouped=DATA.groupby(DATA['ID'])
#print(grouped[1,:])
#this function aggregate all go_terms associated with a protein
##def aggregate_samples(data_frame,group_by):
##    ls_new_datafram=[]
##    for name, group in db_p.groupby(data_frame[group_by]):

        




"""
'''
one_row=DATA.loc[0,['labels','seq']]
#print(one_row['go_term'])
#print(one_row['seq'])
print(one_row['labels'])
print(type(one_row['labels']))
#term_list=one_row['labels']
#find_intersection(term_list)
s,t=prepare_one_sample(one_row)
'''
'''
index_list=[540,2,12321]


s,t=prepare_batch(index_list)
print(s.shape)
print(t.shape)
print(t)
print(np.where(t==1))
'''
#testing
"""
##all_index=[i for i in range(len(DATA))]                     
##seq,tar=prepare_batch([88062])
##print(seq.shape)
##                  
##print(tar.shape)

##for i in range(len(DATA)):
##    ancestor=DATA.loc[i,'labels']
##    DATA.loc[i,'l']=remove_obs_term_from_ancestor(ancestor)
###save
##BP_DATA_N_PATH=os.path.join(config.data_path(),GO_TYPE,"processed_new_v1_bp.csv")
##DATA.to_csv("a.csv", encoding='utf-8', index=False)
#print(obsolete_tirms)

##BP_OBS_PATH=os.path.join(config.data_path(),GO_TYPE,"obsolete_go_terms.csv")
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

#print(len(DATA))

#print(get_all_index())

#print(len(DATA.index))
'''
"""