import sys, os
os.environ['TRANSFORMERS_CACHE'] = './cache/'

from src.dataset import *
from src.utils   import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

train_dataset = pickle.load(open('./data/domain/domainchunk-R10-train-ds.pkl', "rb"))
val_dataset   = pickle.load(open('./data/domain/domainchunk-R10-val-ds.pkl', "rb"))
test_dataset  = pickle.load(open('./data/domain/domainchunk-R10-test-ds.pkl', "rb"))

chunk_r10_domain = {'train': train_dataset , 'val': val_dataset , 'test': test_dataset }


###########################################################################################
###########################################################################################

MLM_ds = get_MLM_ds(chunk_r10_domain,
                   checkpoint     = 'bert-base-uncased',
                   masking_method = 'random',
                   keyword_path   = None,
                   keyword_name   = 'random')


# MLM_ds = get_MLM_ds(chunk_r10_domain,
#                    checkpoint     = 'bert-base-uncased',
#                    masking_method = 'keywords',
#                    keyword_path   = './keyword/01-logodds-topbot1500-R1-nostops.txt',
#                    keyword_name   = 'logodds')


# MLM_ds = get_MLM_ds(chunk_r10_domain,
#                    checkpoint     = 'bert-base-uncased',
#                    masking_method = 'keywords',
#                    keyword_path   = './keyword/02-tfidf-depcon3000-R1-nostops.txt',
#                    keyword_name   = 'tfidf')

# MLM_ds = get_MLM_ds(chunk_r10_domain,
#                    checkpoint     = 'bert-base-uncased',
#                    masking_method = 'keywords',
#                    keyword_path   = './keyword/03-depression-lexicon.txt',
#                    keyword_name   = 'deplex')

# MLM_ds = get_MLM_ds(chunk_r10_domain,
#                    checkpoint     = 'bert-base-uncased',
#                    masking_method = 'keywords',
#                    keyword_path   = './keyword/04-sum-attention-3000.txt',
#                    keyword_name   = 'sumatt')

###########################################################################################
###########################################################################################