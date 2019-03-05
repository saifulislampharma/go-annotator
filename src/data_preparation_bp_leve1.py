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

CUT_POINT=1000
SAMPLE_CSV_FILE="processed_third_bp.csv"
TARGET_CSV_FILE="bp_first_level.csv"
AMINO_ENCODE_CSV_FILE="a_code.csv"
GO_TYPE="BP"

############### TARGET AND AMINO ACID ENCODING PREPARATION##########

#TARGET

TERM_PATH_BP=os.path.join(config.data_path(),"terms",TARGET_CSV_FILE)
bb_terms_df=pd.read_csv(TERM_PATH_BP)
bp_terms=bb_terms_df['term_id'].tolist()


'''
exDict = {'exDict': bp_terms_dict}
with open((os.path.join(config.data_path(),"terms",'bp_first_level.txt')), 'w') as file:
    json.dumps(bp_terms_dict)
'''

#load target_dictionary
with open((os.path.join(config.data_path(),"terms",'bp_first_level.txt')), 'r') as file:
    data=json.load(file)
    bp_terms_dict=data['exDict']
        
              

NO_CLASSES=len(bp_terms_dict)



#AMINO ACID 
amino_path=os.path.join(config.data_path(),AMINO_ENCODE_CSV_FILE)
amino_df=pd.read_csv(amino_path)
a_code=amino_df['code'].tolist()

#dic to convert letter to int
int_frm_letter={key:(value+1) for value,key in enumerate(a_code)}

letter_frm_int={(value+1):key for value,key in enumerate(a_code)}

########################################################## UTILITY FUNCTIONS########
def find_intersection(term_list):
    terms=[term.strip() for term in term_list.split(sep=",")]
    #print(terms)
    intersec=set(terms).intersection(set(bp_terms))
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
    #sample_list=[bp_db.loc[i,['labels','seq']] for i in index_list]
    seq_arrays=[]
    target_arrays=[]
    for i in index_list:
        #print("now processing "+str(i) +"indexed sample")
        try:
            one_row=bp_db.loc[i,['labels','seq']]
        
            #prepare_one_sample(one_row)
            s,t=prepare_one_sample(one_row)
            seq_arrays.append(s)
            target_arrays.append(t)
        except KeyError:
            pass
        
   
    return np.array(seq_arrays),np.array(target_arrays)






    
######################### INITIALIZATION
BP_DATA_PATH=os.path.join(config.data_path(),GO_TYPE,SAMPLE_CSV_FILE)

bp_db=pd.read_csv(BP_DATA_PATH)




def get_all_index():
#this function will return total number of record in processed_bp.csv file
#will be used for split whole data set into test an train
    #BP_DATA_PATH=os.path.join(config.data_path(),GO_TYPE,SAMPLE_CSV_FILE)

    #bps_db=pd.read_csv(BP_DATA_PATH)
    length=len(bp_db.index)
    
    #del bps_db
    
    return length


def get_all_sample():
    total=get_all_index()
    seq_arrays=[]
    target_arrays=[]
    for i in range(total):
        #print("now processing "+str(i) +"indexed sample")
        one_row=bp_db.loc[i,['labels','seq']]
        
        #prepare_one_sample(one_row)
        s,t=prepare_one_sample(one_row)
        seq_arrays.append(s)
        target_arrays.append(t)
    #seq_arrays,target_arrays=[prepare_one_sample(one_row) for one_row in sample_list]
    return np.array(seq_arrays),np.array(target_arrays)    


def combine_all_ancestor_for_one_protein(protein_id):
    protein_df = bp_db.loc[bp_db['ID']==protein_id]
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
        
#BP CATEGORY



####################### TEST#####


# exploring dataset

#how many columns
#print(bp_db.columns)


#Unique IDS
#uniques_ids=bp_d

#what are the rows corresponding to same ID
#combine_all_ancestor_for_one_protein('O95497')
#print('old dataframe length',len(bp_db))
#print('new data frame length',preprocess_data_set(bp_db))
#samples_ID=bp_db.loc[bp_db['ID']=='Q9Y2P4']
#print(bp_db.loc[samples_ID.index[0],'seq'])

#print(type(samples_ID))
#print(len(samples_ID))
#print(samples_ID)

#new_df = preprocess_data_set(bp_db)
#print("new data set length",len(new_df))

#grouped=bp_db.groupby(bp_db['ID'])
#print(grouped[1,:])
#this function aggregate all go_terms associated with a protein
##def aggregate_samples(data_frame,group_by):
##    ls_new_datafram=[]
##    for name, group in db_p.groupby(data_frame[group_by]):

        




'''
one_row=bp_db.loc[0,['labels','go_term','seq']]
#print(one_row['go_term'])
#print(one_row['seq'])
#print(one_row['labels'])
#term_list=one_row['labels']
#find_intersection(term_list)
#index_list=[540,2,102784]

s,t=prepare_one_sample(one_row)
#s,t=prepare_batch(index_list)
print(s.shape)
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
##BP_DATA_N_PATH=os.path.join(config.data_path(),GO_TYPE,"processed_new_v1_bp.csv")
##bp_db.to_csv("a.csv", encoding='utf-8', index=False)
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

#print(len(bp_db))

#print(get_all_index())

#print(len(bp_db.index))
'''
