import os

ROOT_PATH=os.path.dirname(__file__)

#####################DATA RELATED PATHS

def data_path():
    return os.path.join(ROOT_PATH,'data')

def data_mf_path():
    return os.path.join(data_path(),'MF',"combined")

def data_bp_path():
    return os.path.join(data_path(),'BP',"combined")

def data_cc_path():
    return os.path.join(data_path(),'CC',"combined")
def weight_path():
    return os.path.join(ROOT_PATH,"weights")
    
def output_path():
    return os.path.join(ROOT_PATH,"outputs")
######## SRC RELATED PATHS
def src_path():
    return os.path.join(ROOT_PATH,'src')
def log_path():
    return os.path.join(ROOT_PATH,'log')
