import sys, os
sys.path.append('..')

os.environ['TRANSFORMERS_CACHE'] = './cache/'

from src.dataset import *
from src.utils   import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


#######################################################################################################

class_ratios = [1,2,4,6,8,10]

for class_ratio in class_ratios:

    train_dataset = pickle.load(open(f'../data/classi/classichunk-R{class_ratio}-train-ds.pkl', "rb"))
    val_dataset   = pickle.load(open(f'../data/classi/classichunk-R{class_ratio}-val-ds.pkl', "rb"))
    test_dataset  = pickle.load(open(f'../data/classi/classichunk-R{class_ratio}-test-ds.pkl', "rb"))

    chunk_ds = { 'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset }

    MASKER_ds = get_MASKER_ds(
                       class_ratio    =  class_ratio ,
                       chunkdatasets  =  chunk_ds ,
                       checkpoint     = 'bert-base-uncased',
                        masking_method = 'random',
                        keyword_path   = None,
                        keyword_name   = 'random')

    del train_dataset, val_dataset, test_dataset, chunk_ds, MASKER_ds, class_ratio

#######################################################################################################

# masking_method = 'random',
# keyword_path   = None,
# keyword_name   = 'random')


# masking_method = 'keywords',
# keyword_path   = '../keyword/01-log-odds-topbot1500-R1-stops.txt',
# keyword_name   = 'logodds')


# masking_method = 'keywords',
# keyword_path   = '../keyword/02-tfidf-top1500-R1-stops.txt',
# keyword_name   = 'tfidf')


# masking_method = 'keywords',
# keyword_path   = '../keyword/03-depression-lexicon.txt',
# keyword_name   = 'deplex')


# masking_method = 'keywords',
# keyword_path   = '../keyword/04-sum-attention-3000.txt',
# keyword_name   = 'sumatt')

###########################################################################################
###########################################################################################