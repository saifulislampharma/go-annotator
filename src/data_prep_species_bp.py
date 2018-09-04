import pandas as pd
import os
import numpy as np
from .. import config

def combine_all_ancestor_for_one_protein(protein_id):
    protein_df = DATA.loc[DATA['protien_id']==protein_id]
    go_label_ls = protein_df["go_terms"].tolist()

    return ",".join(list(set(go_label_ls)))


def preprocess_data_set(old_df):
    protein_id_ls = list(set(old_df['protien_id'].tolist()))
    new_data_ls=[]
    for protien in protein_id_ls:
        one_sample_ls=[]

        first_level_ancestor_ls = combine_all_ancestor_for_one_protein(protien)
        
        one_sample_ls.append(protien)
        one_sample_ls.append(first_level_ancestor_ls)

        new_data_ls.append(one_sample_ls)

    names=["protien_id","go_terms"]

    return pd.DataFrame(new_data_ls,columns=names)

'''

DATA_PATH = os.path.join(config.data_path(),"BP","species","combined_ex_2759.csv")
DATA = pd.read_csv(DATA_PATH)
print(" data loading complete")

DATA.columns = ["protien_id","go_terms"]

DATAs = preprocess_data_set(DATA)

print("start writing to csv file")

DATAs.to_csv(os.path.join(config.data_path(),"BP","species","agg_2759.csv"),sep=",")

'''
with open("data1.txt") as f:
    lis=[line.split() for line in f] 


