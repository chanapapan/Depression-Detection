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


# ## Get Non-masked dataset Domain dataset

# In[3]:


train_dataset = pickle.load(open(f'../data/domain/domainchunk-R10-train-ds.pkl', "rb"))
val_dataset   = pickle.load(open(f'../data/domain/domainchunk-R10-val-ds.pkl', "rb"))
test_dataset  = pickle.load(open(f'../data/domain/domainchunk-R10-test-ds.pkl', "rb"))

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


# In[5]:


batch_size = 32

# can shuffle now because we use the model to do inference on any sample
train_loader = DataLoader(datasets['train'], batch_size = batch_size)
val_loader   = DataLoader(datasets['val'],   batch_size = batch_size)
test_loader  = DataLoader(datasets['test'],  batch_size = batch_size)


# In[6]:


num_labels  = 2 # CE
checkpoint = 'bert-base-uncased'
model       = AutoModelForTokenClassification.from_pretrained(checkpoint, num_labels = num_labels).to(device)
model.classifier.dropout = nn.Dropout(p = 0.1, inplace = False)

# load model from trained model
path = "../save/NN-02-classitoken-round2/best-model-4200.tar"
print("Load model from : ", path)
# chanadd/model_RSDD/pretrained/FINALv1/save/NN-02-classitoken-round2/best-model-4200.tar

loaded_checkpoint = torch.load(path)
model.load_state_dict(loaded_checkpoint['model_state_dict'])
print(model.load_state_dict(loaded_checkpoint['model_state_dict'])) # <All keys matched successfully>

model.eval()

tokenizer     = BertTokenizerFast.from_pretrained(checkpoint)


# ## Create masked Domain dataset for FURTHER training

# In[8]:


class NNMrandomLMDataset(Dataset):
    
    def __init__(self, data_loader, classitoken_model, tokenizer):
        
        self.tokenizer = tokenizer
        self.SPECIAL_TOKENS   = tokenizer.all_special_ids
        self.model_mask_id    = self.tokenizer(self.tokenizer.mask_token)['input_ids'][1] 
        self.model_cls_id     = self.tokenizer(self.tokenizer.cls_token)['input_ids'][1]
        self.model_sep_id     = self.tokenizer(self.tokenizer.sep_token)['input_ids'][1]
        self.vocab          = [ i for i in range(tokenizer.vocab_size) if (i not in self.SPECIAL_TOKENS)]
        
        self.model_max_length = (self.tokenizer.model_max_length)
        self.chunk_max_length = self.model_max_length - 2
        self.limit_mask       = round(0.15 * self.chunk_max_length)
        
        self.make_NNrandom_MLM_ds(data_loader, classitoken_model)
        
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
    def make_NNrandom_MLM_ds(self, data_loader, model):
        
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
            
            for i in range(input_ids.shape[0]):
                
                this_input_ids = input_ids[i].clone().cpu()
                orig_input_ids = input_ids[i].clone().cpu()
            
                idx_mask_all = [int(pos) for index, pos in enumerate(important_idx_pos) if important_idx_seq[index] == i]
                # print(idx_mask_all)
                
                
                if 0 in idx_mask_all:
                    idx_mask_all.remove(0)
                if 511 in idx_mask_all:
                    idx_mask_all.remove(511)
                
                num_mask_now = len(idx_mask_all)
                # print(num_mask_now)
                
                if num_mask_now > self.limit_mask :
                    # print(num_mask)
                    random.shuffle(idx_mask_all)
                    idx_mask_all = torch.tensor(idx_mask_all[:self.limit_mask]).tolist()
                
                elif num_mask_now < self.limit_mask :
                    idx_mask_now  = torch.tensor(idx_mask_all).long()
                    num_mask_more = self.limit_mask - num_mask_now # how many more token to mask randomly

                    # get possible id to mask more ( do not include first and last token )
                    idx_mask_more = [int(i) for i in range(1, this_input_ids.shape[0]-1) if (i not in idx_mask_now)] 
                    random.shuffle(idx_mask_more) # shuffle
                    idx_mask_more = torch.tensor(idx_mask_more[:num_mask_more]).long() # get only as num_mask_more

                    idx_mask_all = torch.cat([idx_mask_now, idx_mask_more], dim=0).tolist()
                
                
                random.shuffle(idx_mask_all)
                idx_mask_all = torch.tensor(idx_mask_all)
                
                # print(idx_mask_all)
                # print(len(idx_mask_all))
                
                idx_maskmask     = idx_mask_all[: int(len(idx_mask_all)*0.8) ] # 80% len = 60
                idx_maskrandom   = idx_mask_all[ int(len(idx_mask_all)*0.8) : int(len(idx_mask_all)*0.9) ] # 10%  len = 8
                idx_maskoriginal = idx_mask_all[ int(len(idx_mask_all)*0.9) : ] # 10% len = 8    
                
                # print(idx_maskmask)
                # print(idx_maskrandom)
                # print(idx_maskoriginal)
                
                this_input_ids.index_fill_(dim=0, index = idx_maskmask, value = torch.tensor(int(self.model_mask_id)))
                # random tokens 10%
                this_input_ids.detach().cpu().index_put_(indices = (idx_maskrandom, ) , values = torch.tensor(random.sample(self.vocab, len(idx_maskrandom))))
                
                label = (torch.ones_like(this_input_ids.cpu()) * -100)
                label.index_put_(indices = (idx_mask_all, ) , values = torch.index_select(orig_input_ids, dim = 0 , index= torch.tensor(idx_mask_all)))
                
                
                assert torch.sum(torch.isin(this_input_ids, torch.tensor([int(self.model_mask_id)]))) == len(idx_maskmask)
                assert torch.sum(torch.isin(label, orig_input_ids)) == len(idx_mask_all)
                
                self.list_input_ids.append(this_input_ids.clone())
                self.list_word_ids.append(word_ids[i].clone())
                self.list_attention_mask.append(torch.ones_like(this_input_ids))
                self.list_orig_text.append(orig_text[i])
                self.list_masked_text.append(self.tokenizer.decode(this_input_ids.clone()))
                self.list_labels.append(label.clone())
                
        assert len(self.list_input_ids) == len(self.list_word_ids) == len(self.list_attention_mask) == len(self.list_orig_text) == len(self.list_masked_text) == len(self.list_labels)


# In[9]:


my_train_dataset   = NNMrandomLMDataset(train_loader, model, tokenizer)

with open(f'../data/MLM/MLM-NNrandom-train-ds.pkl', 'wb') as outp:
    pickle.dump(my_train_dataset, outp, pickle.HIGHEST_PROTOCOL)
del my_train_dataset


# In[ ]:


my_val_dataset     = NNMrandomLMDataset(val_loader, model, tokenizer)
with open(f'../data/MLM/MLM-NNrandom-val-ds.pkl', 'wb') as outp:
    pickle.dump(my_val_dataset, outp, pickle.HIGHEST_PROTOCOL)
del my_val_dataset


# In[ ]:


my_test_dataset    = NNMrandomLMDataset(test_loader, model, tokenizer)
with open(f'../data/MLM/MLM-NNrandom-test-ds.pkl', 'wb') as outp:
    pickle.dump(my_test_dataset, outp, pickle.HIGHEST_PROTOCOL)
del my_test_dataset

