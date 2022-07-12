#!/usr/bin/env python
# coding: utf-8

# # Use trained model to do inference on the Domain dataset
# ** Don't forget to load the trained weights

# In[1]:


from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForTokenClassification
from torch.utils.data import DataLoader
from IPython.display import clear_output
import torch.nn as nn
import copy
import sys, os
sys.path.append('..')

os.environ['TRANSFORMERS_CACHE'] = './cache/'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from src.dataset import *
from src.utils   import *
# from src.traineval  import *

SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = get_freer_gpu()
# device = 'cpu'
print('device', device)


# In[2]:


import argparse
parser  = argparse.ArgumentParser()
parser.add_argument('-f', '--ratio', help='ratio of control user to depression user' , type=int, required=True)
args     = parser.parse_args()


# In[4]:


# class_ratio = 1
class_ratio          = args.ratio


# In[5]:


train_dataset = pickle.load(open(f'../data/classi/classichunk-R{class_ratio}-train-ds.pkl', "rb"))
val_dataset   = pickle.load(open(f'../data/classi/classichunk-R{class_ratio}-val-ds.pkl', "rb"))
test_dataset  = pickle.load(open(f'../data/classi/classichunk-R{class_ratio}-test-ds.pkl', "rb"))

datasets = {'train' : train_dataset, 'val' : val_dataset, 'test': test_dataset }

for sample in train_dataset:
    print(sample['input_ids'])
    print(sample['word_ids'])
    print(sample['attention_mask'])
    print(sample['orig_text'])
    print(sample['labels'])
    break
    
print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))


# In[6]:


num_labels  = 2 # CE
checkpoint = 'bert-base-uncased'
model       = AutoModelForTokenClassification.from_pretrained(checkpoint, num_labels = num_labels).to(device)
model.classifier.dropout = nn.Dropout(p = 0.1, inplace = False)

# load model from trained model
path = '../save/PROP-classitoken/best-model-2000.tar'
print("Load model from : ", path)

loaded_checkpoint = torch.load(path)
model.load_state_dict(loaded_checkpoint['model_state_dict'])
print(model.load_state_dict(loaded_checkpoint['model_state_dict'])) # <All keys matched successfully>

model.eval()

tokenizer     = BertTokenizerFast.from_pretrained(checkpoint)
model_mask_id = tokenizer(tokenizer.mask_token)['input_ids'][1]
print(model_mask_id)


# ## Create masked Domain dataset for FURTHER training (Done but need to be checked)

# In[13]:


class PROPprobmMASKERDataset(Dataset):
    
    def __init__(self, classichunkds, classitoken_model, tokenizer):
        
        self.tokenizer        = tokenizer
        self.model_mask_id    = self.tokenizer(self.tokenizer.mask_token)['input_ids'][1] 
        self.model_cls_id     = self.tokenizer(self.tokenizer.cls_token)['input_ids'][1]
        self.model_sep_id     = self.tokenizer(self.tokenizer.sep_token)['input_ids'][1]  
        self.make_PROPprob_MASKER_ds(classichunkds)
        
        # del classitoken_model, classichunkds
        
    def __len__(self):
        return len(self.list_all_input_ids)

    def __getitem__(self, idx):
        sample = {  'all_input_ids'  : self.list_all_input_ids[idx],
                    'word_ids'       : self.list_word_ids[idx],
                    'orig_text'      : self.list_orig_text[idx],
                    'all_labels'     : self.list_all_labels[idx]}
        return sample
    
    # Use important_words_info to make the label for training TokenClassification on this D(task) dataset
    def make_PROPprob_MASKER_ds(self, classichunkds):
        
        self.list_all_input_ids = []
        self.list_word_ids      = []
        self.list_orig_text     = []
        self.list_all_labels    = []

        for idx, data in enumerate(classichunkds):  
            sys.stdout.write(str(idx))
            
            input_ids    = data['input_ids'].clone().reshape(1,-1).to(device)
            att_mask     = data['attention_mask'].clone().reshape(1,-1).to(device)
            
            word_ids     = data['word_ids'].clone()
            orig_text    = data['orig_text']
            labels       = data['labels']

            orig_input_ids = data['input_ids'].clone().cpu()
            orig_token     = orig_input_ids.clone()
            MKR_tokens     = orig_input_ids.clone()  # masked token (Masked Keywords Reconstruction)
            MER_tokens     = orig_input_ids.clone()  # outlier token (Masked Entropy Regularization)
            
            
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids = input_ids, attention_mask = att_mask)
            
            pred = torch.argmax(torch.softmax(outputs.logits.detach(), dim = 2), dim = 2) # bs, seq_len
            
            # get index of important tokens
            important_idx = (pred == 1).nonzero(as_tuple=True)[1].cpu()
            # print(important_idx)
            
            idx_out_mask = torch.tensor([idx for idx in range(1, orig_input_ids.shape[0]-1) if idx not in important_idx ]).long()
            # print(idx_out_mask)

            
            p = 0.9
            q = 0.9
            # mask the keywords only where p_mask < 0.5
            p_mask = torch.rand(torch.tensor(important_idx).shape)
            important_idx = torch.tensor([idx_k for i, idx_k in enumerate(important_idx) if p_mask[i] < p]).long()
            # print(important_idx)

            # mask the context only where q_mask < 0.9
            q_mask = torch.rand(idx_out_mask.shape)
            idx_out_mask = torch.tensor([idx_k for i, idx_k in enumerate(idx_out_mask) if q_mask[i] < q])
            # print(idx_out_mask)

            
            #  --------------- MKR ---------------
            MKR_tokens[important_idx] = self.model_mask_id
            MKR_tokens[0]  = self.model_cls_id
            MKR_tokens[-1] = self.model_sep_id
            # print(MKR_tokens)
            
            MKR_labels = (torch.ones_like(orig_input_ids) * -100)
            MKR_labels.index_put_(indices = (important_idx, ) , values = torch.index_select(orig_input_ids, dim = 0 , index = important_idx))
            MKR_labels[0]  = -100
            MKR_labels[-1] = -100
            # print(MKR_labels)
            
            #  -------------- MER --------------- 
            
            MER_tokens.index_fill_(dim=0, index = idx_out_mask, value = int(self.model_mask_id))
            MER_tokens[0]  = self.model_cls_id
            MER_tokens[-1] = self.model_sep_id
            
            # print(MER_tokens)
            
            # print(orig_token)
            # print(MKR_tokens)
            # print(MER_tokens)
            # print(MKR_labels)
            
            self.list_all_input_ids.append((orig_token, MKR_tokens, MER_tokens,))
            self.list_word_ids.append(word_ids.clone())
            self.list_orig_text.append(orig_text)
            self.list_all_labels.append((MKR_labels, labels,))
            
            # asdfasdf
            
        assert len(self.list_all_input_ids) == len(self.list_word_ids) == len(self.list_orig_text) == len(self.list_all_labels)


# In[14]:


my_train_dataset = PROPprobmMASKERDataset(train_dataset, model, tokenizer)

with open(f'../data/MASKER/MASKER-PROPprob-R{class_ratio}-train-ds.pkl', 'wb') as outp:
    pickle.dump(my_train_dataset, outp, pickle.HIGHEST_PROTOCOL)


# In[ ]:


my_val_dataset = PROPprobmMASKERDataset(val_dataset, model, tokenizer)

with open(f'../data/MASKER/MASKER-PROPprob-R{class_ratio}-val-ds.pkl', 'wb') as outp:
    pickle.dump(my_val_dataset, outp, pickle.HIGHEST_PROTOCOL)


# In[ ]:


my_test_dataset = PROPprobmMASKERDataset(test_dataset, model, tokenizer)

with open(f'../data/MASKER/MASKER-PROPprob-R{class_ratio}-test-ds.pkl', 'wb') as outp:
    pickle.dump(my_test_dataset, outp, pickle.HIGHEST_PROTOCOL)

