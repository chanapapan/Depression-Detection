#!/usr/bin/env python
# coding: utf-8

# # Check what the Neural Network method mask

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
from src.traineval  import *

SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = get_freer_gpu()
print('device', device)


# In[2]:


train_dataset = pickle.load(open(f'../data/domain/domainchunk-R10-train-ds.pkl', "rb"))

for sample in train_dataset:
    print(sample['input_ids'])
    print(sample['word_ids'])
    print(sample['attention_mask'])
    print(sample['orig_text'])
    print(sample['labels'])
    break


# In[3]:


batch_size = 32

# can shuffle now because we use the model to do inference on any sample
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)


# In[4]:


num_labels  = 2 # CE
checkpoint = 'bert-base-uncased'
model       = AutoModelForTokenClassification.from_pretrained(checkpoint, num_labels = num_labels).to(device)
model.classifier.dropout = nn.Dropout(p = 0.1, inplace = False)

# load model from trained model
path = '../save/NN-02-classitoken-round2/best-model-4200.tar'
print("Load model from : ", path)

loaded_checkpoint = torch.load(path)
model.load_state_dict(loaded_checkpoint['model_state_dict'])
print(model.load_state_dict(loaded_checkpoint['model_state_dict'])) # <All keys matched successfully>

model.eval()

tokenizer     = BertTokenizerFast.from_pretrained(checkpoint)


# In[7]:


class NNMLMDataset(Dataset):
    
    def __init__(self, data_loader, classitoken_model, tokenizer):
        
        self.tokenizer = tokenizer
        self.model_mask_id = self.tokenizer(self.tokenizer.mask_token)['input_ids'][1] 
        self.model_cls_id     = self.tokenizer(self.tokenizer.cls_token)['input_ids'][1]
        self.model_sep_id     = self.tokenizer(self.tokenizer.sep_token)['input_ids'][1]
        self.make_NN_MLM_ds(data_loader, classitoken_model)
        
        
        
        del classitoken_model, data_loader
        
    def __len__(self):
        return len(self.list_input_ids)

    def __getitem__(self, idx):       
        sample = {  'input_ids'      : self.list_input_ids[idx],
                    'word_ids'       : self.list_word_ids[idx],
                    'attention_mask' : self.list_attention_mask[idx],
                    'orig_text'      : self.list_orig_text[idx],
                    'masked_text'    : self.list_masked_text[idx],
                    'labels'         : self.list_labels[idx]}
        return sample
    
    # Use the trained model to do inference on domain dataset to creats masked domain ds
    def make_NN_MLM_ds(self, data_loader, model):
        
        self.list_input_ids      = []
        self.list_word_ids       = []
        self.list_attention_mask = []
        self.list_orig_text      = []
        self.list_masked_text    = []
        self.list_labels         = []
        
        self.list_important_input_ids = []

        for idx, batch in enumerate(data_loader):  
            
            sys.stdout.write(str(idx))
            
            input_ids = batch['input_ids'].clone().to(device)
            att_mask  = batch['attention_mask'].clone().to(device)
            
            word_ids     = batch['word_ids']
            orig_text    = batch['orig_text']
            labels       = batch['labels']
            
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids = input_ids, attention_mask = att_mask)
            
            pred = torch.argmax(torch.softmax(outputs.logits.detach(), dim = 2), dim = 2) # bs, seq_len
            # print(pred.shape)
            
            # get index of important tokens
            important_idx_seq = (pred == 1).nonzero(as_tuple=True)[0]
            important_idx_pos = (pred == 1).nonzero(as_tuple=True)[1]
            
            # print(important_idx_seq[0:5], important_idx_pos[0:5])
            
            important_input_ids = input_ids.clone().detach()[important_idx_seq, important_idx_pos]
            
            self.list_important_input_ids.append(important_input_ids)

# In[8]:


my_train_dataset   = NNMLMDataset(train_loader, model, tokenizer)


# In[12]:


masked_token_freq = {}

for input_ids in my_train_dataset.list_important_input_ids:
    for each_id in input_ids:
        # print(each_id)
        token = tokenizer.decode(each_id)
        # print(token)
        if token not in masked_token_freq.keys():
            # print("new")
            masked_token_freq[token] = 1
        else : 
            # print("old")
            masked_token_freq[token] += 1


# In[13]:


keyword = sorted(masked_token_freq.items(), key=lambda x: x[1], reverse=True)
# print(keyword)

masked_words = [kw for kw, freq in keyword]
# print(masked_words)


# In[11]:


with open(f"./nn_masked_words_train.txt", "w") as f:
    for word in masked_words:
        f.write(word + "\n")


# In[ ]:




