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


# ## Get Non-masked dataset Domain dataset

# In[3]:


train_dataset = pickle.load(open(f'./data/domain/domainchunk-R10-train-ds.pkl', "rb"))
val_dataset   = pickle.load(open(f'./data/domain/domainchunk-R10-val-ds.pkl', "rb"))
test_dataset  = pickle.load(open(f'./data/domain/domainchunk-R10-test-ds.pkl', "rb"))

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


# In[4]:


batch_size = 32

# can shuffle now because we use the model to do inference on any sample
train_loader = DataLoader(datasets['train'], batch_size = batch_size)
val_loader   = DataLoader(datasets['val'],   batch_size = batch_size)
test_loader  = DataLoader(datasets['test'],  batch_size = batch_size)


# In[5]:


num_labels  = 2 # CE
checkpoint = 'bert-base-uncased'
model       = AutoModelForTokenClassification.from_pretrained(checkpoint, num_labels = num_labels).to(device)
model.classifier.dropout = nn.Dropout(p = 0.1, inplace = False)

# load model from trained model
path = "./save/NN-02-classitoken-round2/best-model-4200.tar"
print("Load model from : ", path)
# chanadd/model_RSDD/pretrained/FINALv1/save/NN-02-classitoken-round2/best-model-4200.tar

loaded_checkpoint = torch.load(path)
model.load_state_dict(loaded_checkpoint['model_state_dict'])
print(model.load_state_dict(loaded_checkpoint['model_state_dict'])) # <All keys matched successfully>

model.eval()

tokenizer     = BertTokenizerFast.from_pretrained(checkpoint)


# ## Create masked Domain dataset for FURTHER training

# In[8]:


class NNMLMDataset(Dataset):
    
    def __init__(self, data_loader, classitoken_model, tokenizer):
        
        self.tokenizer     = tokenizer
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
    
    # Use the trained model to do inference on domain dataset to creats maseked domain ds
    def make_NN_MLM_ds(self, data_loader, model):
        
        self.list_input_ids      = []
        self.list_word_ids       = []
        self.list_attention_mask = []
        self.list_orig_text      = []
        self.list_masked_text    = []
        self.list_labels         = []

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
            # print(important_input_ids)
            
            # put [MASK] token at the position of the important tokens
            masked_input_ids = input_ids.detach().clone()
            masked_input_ids[important_idx_seq, important_idx_pos] = self.model_mask_id
            # ensure that the first and last tokens are not masked
            masked_input_ids[:, 0]   = self.model_cls_id
            masked_input_ids[:, -1] = self.model_sep_id
            # print(masked_input_ids.shape)
            
            labels    = torch.ones_like(att_mask).to(device) * -100 # init all labels with -100
            # put original token input_ids at the position of important tokens
            masked_labels  = labels.index_put(indices = (important_idx_seq, important_idx_pos) , values = important_input_ids)
            # ensure that the model do not predict the fist and last tokens
            masked_labels[:, 0]  = -100
            masked_labels[:, -1] = -100
            
            for i in range(input_ids.shape[0]):
            
                self.list_input_ids.append(masked_input_ids[i].clone())
                self.list_word_ids.append(word_ids[i].clone())
                self.list_attention_mask.append(torch.ones_like(masked_input_ids[i]))
                self.list_orig_text.append(orig_text[i])
                self.list_masked_text.append(self.tokenizer.decode(masked_input_ids[i].clone()))
                self.list_labels.append(masked_labels[i].clone())
                
                # print(masked_input_ids[10].clone())
                # print(masked_labels[10].clone())
                
        assert len(self.list_input_ids) == len(self.list_word_ids) == len(self.list_attention_mask) == len(self.list_orig_text) == len(self.list_masked_text) == len(self.list_labels)


# In[9]:


my_train_dataset   = NNMLMDataset(train_loader, model, tokenizer)

with open(f'./data/MLM/MLM-NN-train-ds.pkl', 'wb') as outp:
    pickle.dump(my_train_dataset, outp, pickle.HIGHEST_PROTOCOL)
del my_train_dataset


# In[ ]:


my_val_dataset     = NNMLMDataset(val_loader, model, tokenizer)
with open(f'./data/MLM/MLM-NN-val-ds.pkl', 'wb') as outp:
    pickle.dump(my_val_dataset, outp, pickle.HIGHEST_PROTOCOL)
del my_val_dataset


# In[ ]:


my_test_dataset    = NNMLMDataset(test_loader, model, tokenizer)
with open(f'./data/MLM/MLM-NN-test-ds.pkl', 'wb') as outp:
    pickle.dump(my_test_dataset, outp, pickle.HIGHEST_PROTOCOL)
del my_test_dataset

