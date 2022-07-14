#!/usr/bin/env python
# coding: utf-8

# # MASKER KEYWORD - Attention
# 
# ### Top words by highest sum of attention score

# - load model classi ratio 1
# - dataset ratio 1

# In[1]:


from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForSequenceClassification

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


load_model_from    = '../save/BASE-classiCEr1/best-model-1500.tar'

checkpoint         = 'bert-base-uncased'

training_obj       = 'classiCE'
masking_method     = None
keyword_path       = None

classifier_p_dropout = 0.1


# In[3]:


num_labels = 2 #CE
model      = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = num_labels).to(device)
model.classifier.dropout = nn.Dropout(p = classifier_p_dropout, inplace = False)


print("Load from ", load_model_from)
checkpoint = torch.load(load_model_from)
model.load_state_dict(checkpoint['model_state_dict'])
print(model.load_state_dict(checkpoint['model_state_dict']))

from transformers import AutoTokenizer
checkpoint   = 'bert-base-uncased'
tokenizer    =  AutoTokenizer.from_pretrained(checkpoint)


# In[ ]:


train_dataset = pickle.load(open(f'../data/classi/classichunk-R1-train-ds.pkl', "rb"))

for sample in train_dataset:
    print(sample['input_ids'])
    print(sample['word_ids'])
    print(sample['attention_mask'])
    print(sample['orig_text'])
    print(sample['labels'])
    break


# ## Original Code

# In[5]:


# ORIGINAL CODE !!!
def get_attention_keyword(dataset, model, tokenizer, device, num_kw):

    loader = DataLoader(dataset, shuffle=False, batch_size = 16)

    SPECIAL_TOKENS = tokenizer.all_special_ids
    vocab_size = len(tokenizer)

    attn_score = torch.zeros(vocab_size)
    attn_freq  = torch.zeros(vocab_size)

    for idx , batch in enumerate(loader):
        
        sys.stdout.write(str(idx))
        
        tokens   = batch['input_ids'].to(device)
        word_ids = batch['word_ids']
        labels   = batch['labels'].cpu()
        
        model.eval()
        with torch.no_grad():
            output    = model(tokens, output_attentions=True) # (batch_size, num_heads, sequence_length, sequence_length)           
            attention = output.attentions[-1] # get attention of last layer (batch_size, num_heads, sequence_length, sequence_length)
        
        pred = torch.argmax(torch.softmax(output.logits.detach(), dim = 1), dim = 1).detach().cpu()
        
        correct_idx = (labels == pred).nonzero(as_tuple=True)[0].detach().cpu()
        # print(correct_idx)
        
        correct_attention = torch.index_select(attention.clone().detach().cpu(), dim = 0 , index = correct_idx)
        # print(correct_attention.shape)
        
        attention = correct_attention.sum(dim = 1) # sum over attention heads (batch_size, sequence_length, sequence_length)
        
        for  i in range(attention.size(0)):  # for each sample in batch
            for j in range(attention.size(-1)):  # max_len
                token = tokens[i][j].item()
                
                if token in SPECIAL_TOKENS:  # skip special token
                    continue

                score = attention[i][0][j]  # 1st token = CLS token

                attn_score[token] += score.item()
                attn_freq[token] += 1

    for tok in range(vocab_size):
        
        if attn_freq[tok] < 10 : # if freq less than 10 REMOVE from the list !
            attn_score[tok] = 0
            
        else:
            attn_score[tok] /= attn_freq[tok]  # normalize by frequency

    keyword = attn_score.argsort(descending=True)[:num_kw].tolist()

    return keyword, attn_score, attn_freq


# In[6]:


keyword, attn_score, attn_freq = get_attention_keyword(train_dataset, model, tokenizer, device, num_kw = 3000)


# In[7]:


keywords = [tokenizer.decode([word]) for word in keyword]
print(len(keywords))


# In[8]:


with open("./04-sum-attention-3000.txt", "w") as f:
    for word in keywords:
        f.write(word + "\n")

