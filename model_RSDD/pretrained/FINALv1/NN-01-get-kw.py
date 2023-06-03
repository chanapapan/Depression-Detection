#!/usr/bin/env python
# coding: utf-8

# # Selective Masking - Train no evil : Find important tokens from D(task)
# 
# * Do for 'train' 'val' 'test' for ratio 1 to 12*

# In[28]:


from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from IPython.display import clear_output
import torch.nn as nn
import copy
import sys, os

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


# ## Load D(task) dataset

# In[33]:


class_ratio     = 1
split           = 'test'
threshold       = 0.01

# path of best model of this class ratio
path       = './save/BASE-classiCEr1/best-model-1500.tar'


# In[30]:


train_dataset = pickle.load(open(f'./data/classi/classichunk-R{class_ratio}-train-ds.pkl', "rb"))
# val_dataset   = pickle.load(open(f'./data/classi/classichunk-R{class_ratio}-val-ds.pkl', "rb"))
# test_dataset  = pickle.load(open(f'./data/classi/classichunk-R{class_ratio}-test-ds.pkl', "rb"))

# datasets = { 'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset }


# In[14]:


# # check if the classichunk dataset are really subset of each other

# class_ratio   = 1
# dataset_1 = pickle.load(open(f'./data/classi/classichunk-R{class_ratio}-train-ds.pkl', "rb")).__getitem__(4630)['input_ids']

# class_ratio   = 2
# dataset_2 = pickle.load(open(f'./data/classi/classichunk-R{class_ratio}-train-ds.pkl', "rb")).__getitem__(4630)['input_ids']

# class_ratio   = 4
# dataset_4 = pickle.load(open(f'./data/classi/classichunk-R{class_ratio}-train-ds.pkl', "rb")).__getitem__(4630)['input_ids']

# class_ratio   = 6
# dataset_6 = pickle.load(open(f'./data/classi/classichunk-R{class_ratio}-train-ds.pkl', "rb")).__getitem__(4630)['input_ids']

# class_ratio   = 8
# dataset_8 = pickle.load(open(f'./data/classi/classichunk-R{class_ratio}-train-ds.pkl', "rb")).__getitem__(4630)['input_ids']

# class_ratio   = 10
# dataset_10 = pickle.load(open(f'./data/classi/classichunk-R{class_ratio}-train-ds.pkl', "rb")).__getitem__(4630)['input_ids']

# print(torch.all(dataset_1 == dataset_2), torch.all(dataset_2 == dataset_4), torch.all(dataset_4 == dataset_6), torch.all(dataset_6 == dataset_8), torch.all(dataset_8 == dataset_10))


# In[16]:


# # check if the classichunk dataset are really subset of each other

# class_ratio   = 1
# dataset_1 = pickle.load(open(f'./data/classi/classichunk-R{class_ratio}-val-ds.pkl', "rb")).__getitem__(460)['input_ids']

# class_ratio   = 2
# dataset_2 = pickle.load(open(f'./data/classi/classichunk-R{class_ratio}-val-ds.pkl', "rb")).__getitem__(460)['input_ids']

# class_ratio   = 4
# dataset_4 = pickle.load(open(f'./data/classi/classichunk-R{class_ratio}-val-ds.pkl', "rb")).__getitem__(460)['input_ids']

# class_ratio   = 6
# dataset_6 = pickle.load(open(f'./data/classi/classichunk-R{class_ratio}-val-ds.pkl', "rb")).__getitem__(460)['input_ids']

# class_ratio   = 8
# dataset_8 = pickle.load(open(f'./data/classi/classichunk-R{class_ratio}-val-ds.pkl', "rb")).__getitem__(460)['input_ids']

# class_ratio   = 10
# dataset_10 = pickle.load(open(f'./data/classi/classichunk-R{class_ratio}-val-ds.pkl', "rb")).__getitem__(460)['input_ids']

# print(torch.all(dataset_1 == dataset_2), torch.all(dataset_2 == dataset_4), torch.all(dataset_4 == dataset_6), torch.all(dataset_6 == dataset_8), torch.all(dataset_8 == dataset_10))


# In[21]:


# # check if the classichunk dataset are really subset of each other

# class_ratio   = 1
# dataset_1 = pickle.load(open(f'./data/classi/classichunk-R{class_ratio}-test-ds.pkl', "rb")).__getitem__(398)['input_ids']

# class_ratio   = 2
# dataset_2 = pickle.load(open(f'./data/classi/classichunk-R{class_ratio}-test-ds.pkl', "rb")).__getitem__(398)['input_ids']

# class_ratio   = 4
# dataset_4 = pickle.load(open(f'./data/classi/classichunk-R{class_ratio}-test-ds.pkl', "rb")).__getitem__(398)['input_ids']

# class_ratio   = 6
# dataset_6 = pickle.load(open(f'./data/classi/classichunk-R{class_ratio}-test-ds.pkl', "rb")).__getitem__(398)['input_ids']

# class_ratio   = 8
# dataset_8 = pickle.load(open(f'./data/classi/classichunk-R{class_ratio}-test-ds.pkl', "rb")).__getitem__(398)['input_ids']

# class_ratio   = 10
# dataset_10 = pickle.load(open(f'./data/classi/classichunk-R{class_ratio}-test-ds.pkl', "rb")).__getitem__(398)['input_ids']

# print(torch.all(dataset_1 == dataset_2), torch.all(dataset_2 == dataset_4), torch.all(dataset_4 == dataset_6), torch.all(dataset_6 == dataset_8), torch.all(dataset_8 == dataset_10))


# ## Make a train_loader that does not shuffle the order of the samples, to preserve the order

# In[31]:


print(len(train_dataset))

# no shuffle because we want to preserve the index 
batch_size    = 32
data_loader  = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)


# ## Load weights of model that has been finetuned on classification task

# In[34]:


checkpoint = 'bert-base-uncased'
tokenizer  = BertTokenizerFast.from_pretrained(checkpoint)

num_labels = 2 #CE
model      = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = num_labels).to(device)

checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])
print(model.load_state_dict(checkpoint['model_state_dict']))


# ## Get confidence scores of all CORRECTLY CLASSIFIED samples >> save as ds

# In[35]:


def get_all_pred_score(batch, batch_idx, batch_size):
    
    '''
    batch_idx = batch index from train_loader
    data      = data of that batch
    
    return (each is a list of len(num_correctly_classified_samples)
    correct_input_ids  : input_ids of the correctly classified samples (list)
    global_idx_correct : sample indices of the correctly classified samples (list)
    correct_labels     : labels of the correctly classified samples (list)
    confidence_scores  : confidence scores of the correctly classifies samples (list)
    '''
    input_ids = batch['input_ids'].to(device)
    att_mask  = batch['attention_mask'].to(device)
    labels    = batch['labels'].to(device)
    
    # texts     = data['orig_text']
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=att_mask, labels=labels)

    # get logits from outputs
    logits  = outputs.logits # (before SoftMax)
    
    logits  = torch.softmax(logits, dim = 1).detach().cpu().numpy()
    preds   = np.argmax(logits, axis = 1)
    # print("LABELS : ", labels)
    # print("PREDS : ", preds)
    
    labels    = labels.long().detach().cpu().numpy()
    correct_mask = labels == preds
    idx_correct  = np.where(correct_mask)[0]
    
    if len(idx_correct) == 0:
        # print("yes")
        return [], [], [], []
    
    # print("idx_correct : ", idx_correct)
    global_idx_correct  = idx_correct + ( batch_idx * batch_size )
    
    correct_labels = labels[correct_mask].tolist()
    # print("correct_labels : ", correct_labels)
    
    # print(len(idx_correct))
    input_ids         = input_ids.detach().cpu().numpy()
    correct_input_ids = np.take(input_ids, idx_correct, axis = 0).tolist()
    
    correct_logits            = np.take(logits, idx_correct, axis = 0).tolist()
    correct_confidence_scores = np.take(correct_logits, correct_labels, axis = 1)[:, 0].tolist()
    # print("confidence_scores : ", confidence_scores)
    
#     print(correct_input_ids)
#     print(correct_labels)
#     print(correct_confidence_scores)
#     print(global_idx_correct)
    
    return correct_input_ids, correct_labels, correct_confidence_scores, global_idx_correct


# In[36]:


# First time only
ds = []
 
all_correct_input_ids    = [] # input_ids of all correct samples
all_correct_labels       = [] # labels of all correct samples
all_correct_pred_score   = [] # prediction scores of all correct samples
all_correct_global_idx   = [] # index of all correct samples

for batch_idx, batch in enumerate(data_loader):   
   
    sys.stdout.write(str(batch_idx)) 
    
    correct_input_ids, correct_labels, correct_confidence_scores, global_idx_correct = get_all_pred_score(batch, batch_idx, batch_size)
    
    all_correct_input_ids.extend(correct_input_ids)   
    all_correct_labels.extend(correct_labels)  
    all_correct_pred_score.extend(correct_confidence_scores)   
    all_correct_global_idx.extend(global_idx_correct)
    
    assert len(all_correct_input_ids) == len(all_correct_labels) == len(all_correct_pred_score) == len(all_correct_global_idx)

# put everything together so we can sort
for i in range(len(all_correct_global_idx)):
    ds.append((all_correct_input_ids[i], all_correct_labels[i], all_correct_pred_score[i], all_correct_global_idx[i]))

    
# sort ds according to prediction scores from HIGHEST TO LOWEST
ds = sorted(ds, key=lambda x : x[-1], reverse=True)
print("Num correct samples : ", len(ds))

# save ds so we can change the number of top_samples later
with open(f'./NN-01-ds-r{class_ratio}.pkl', 'wb') as outp:
    pickle.dump(ds, outp, pickle.HIGHEST_PROTOCOL)


# ## Load ds and get the top samples

# In[37]:


# load ds
ds = pickle.load(open(f'./NN-01-ds-r{class_ratio}.pkl', "rb"))
print("Total number of correctly predicted sample : " , len(ds))

# Select top samples : ALL correct samples
num_top_samples = len(ds)
top_input_ids, top_labels, top_scores, top_index = zip( *ds [0 : num_top_samples] )

# input_ids         of TOP correct samples
# labels            of TOP correct samples
# prediction scores of TOP correct samples
# global_index      of TOP correct samples

assert len(top_input_ids) == len(top_labels) == len(top_scores) == len(top_index)

top_input_ids = list(top_input_ids)
top_labels    = list(top_labels)
top_scores    = list(top_scores)
top_index     = list(top_index)

print("Selected samples : ", len(top_input_ids))


# ## Find important words with Buffer Algorithm
# 
# ** results differ based on the threshold

# In[66]:


def get_buffer_score( buffer_in, buffer_label_in ) :
    '''
    Calculate classification score for buffer_in
    '''
    buffer_in.append(102) # add SEP token
    buffer_in = torch.tensor(buffer_in).long()
    # print(buffer_in)
    
    # pad the buffer to 512
    padded_buffer = torch.zeros((1, 512)).long() # 0 is the padding input_ids
    padded_buffer[ : , : len(buffer_in) ] = buffer_in
    # print(padded_buffer)
    
    # prepare model inputs
    input_ids       = padded_buffer.to(device)
    attention_mask  = torch.ones_like(input_ids)
    attention_mask[ : , len(buffer_in) : ] = 0
    # labels          = torch.tensor(buffer_label_in).to(device)
    
    outputs         = model(input_ids = input_ids, attention_mask = attention_mask)
    logits          = torch.softmax(outputs.logits, dim = 1).detach().cpu().numpy()
    buffer_score    = logits[ : , buffer_label_in ] # get prediction score
    
    # get index of added token in this buffer
    position_of_added_token = (input_ids == 102).nonzero(as_tuple=True)[1] - 1
    # print(position_of_added_token)
    
    # get the input_ids of the added token
    input_ids_of_added_token = input_ids[ : , position_of_added_token]
    # print(input_ids_of_added_token)
    # print("added_token_position : ",  added_token_position)
    # print("added_token_id : ", added_token_id)

    return buffer_score, input_ids_of_added_token


# ## Save important word index and tokens in mask_info
# **also save the buffer and scores in buffer_score_dict

# In[75]:


#########################################################################################################################

# initialize list of buffers for the top samples
buffer_list = []
for i in range(len(top_input_ids)):  
    buffer_list.append(top_input_ids[i][0:2])
# print(buffer_list)

important_words_info = {}
for idx in top_index:                        # global index of this sample
    important_words_info[idx] = {}                      # create dict for this sample
    important_words_info[idx]['token_position'] = []    # index of token to mask in this sample
    important_words_info[idx]['token_id'] = []          # token id to mask in this sample
    important_words_info[idx]['confidence_score'] = []  # confidence score of this sample

# just for checking
buffer_score_dict = {}
buffer_score_dict['buffer'] = []             # buffer
buffer_score_dict['score_diff']  = []        # score of this buffer

###########################################################################################################################

# for each top correct sample
for i, (buffer, real_labels, orig_scores, sample_idx) in enumerate(zip(buffer_list, top_labels, top_scores, top_index)):
    count_important = 0
    
    sys.stdout.write(str(i)) 
    
    position = 1
    
    while position < 511 : # not 512 bc thats the ['SEP']
        
        buffer_score, input_ids_of_added_token = get_buffer_score( buffer , real_labels )
        
        buffer_score_dict['buffer'].append(buffer)
        buffer_score_dict['score_diff'].append(orig_scores - buffer_score)
        
        if (orig_scores >= buffer_score) and (( orig_scores - buffer_score) < threshold ) :
            
            count_important += 1
            
            # the word we just added to the buffer should be the same word with the word that we are going to mark as important
            # print(train_dataset.__getitem__(sample_idx)['input_ids'][position].to(device).shape)
            # print(input_ids_of_added_token.to(device).shape)
            assert train_dataset.__getitem__(sample_idx)['input_ids'][position].cpu() == input_ids_of_added_token.cpu().squeeze(0)
            
            # the original position of the important word in the train_dataset's input_ids
            important_words_info[sample_idx]['token_position'].extend([position]) 
            # the input_ids of the important word
            important_words_info[sample_idx]['token_id'].extend(input_ids_of_added_token.tolist()) 
            # score of this word
            important_words_info[sample_idx]['confidence_score'].extend(buffer_score)
            
            
            buffer.pop() # pop SEP
            buffer.pop() # pop important word

        else : # if added word is not important
            buffer.pop() # pop only SEP
        
        position += 1
        # append next word to the buffer
        buffer.append( top_input_ids[i][position : position + 1][0] )  
        
    print("Number of important words in this sample : ", count_important)
        
        
with open(f'./NN-01-important-word-info-t{threshold}-r{class_ratio}.pkl', 'wb') as outp:
    pickle.dump(important_words_info, outp, pickle.HIGHEST_PROTOCOL)
    
# just for checking
with open(f'./NN-01-buffer-score-dict-t{threshold}-r{class_ratio}.pkl', 'wb') as outp:
    pickle.dump(buffer_score_dict, outp, pickle.HIGHEST_PROTOCOL)


# ## Check top_mask_info and buffer_score_dict

# In[100]:


# class_ratio     = 1
# split           = 'test'
# threshold       = 0.01


# important_words_info  = pickle.load(open(f'./data/NN/SM-01-important-word-info-t{threshold}-r{class_ratio}.pkl', "rb"))
# buffer_score_dict     = pickle.load(open(f'./data/NN/SM-01-buffer-score-dict-t{threshold}-r{class_ratio}.pkl', "rb"))


# In[101]:


# # Buffer score stats

# print(np.mean(buffer_score_dict['score_diff']))
# print(np.min(buffer_score_dict['score_diff']))
# print(np.max(buffer_score_dict['score_diff']))


# In[102]:


# # Average tokens to mask per sample

# all_len_important = []
# for sample_id in important_words_info.keys():
#     all_len_important.append(len(important_words_info[sample_id]['token_position']))
# print(np.mean(all_len_important))


# ### if mask more than 76 get only the top 76

# In[103]:


# for i in important_words_info.keys():
#     if len(important_words_info[i]['confidence_score']) > 76 :
        
#         top_idx = np.argsort(important_words_info[i]['confidence_score'])[::-1] # index of top 76 highest confidence score
        
#         important_words_info[i]['confidence_score'] = [important_words_info[i]['confidence_score'][j] for j in top_idx][:76]
#         important_words_info[i]['token_position']   = [important_words_info[i]['token_position'][j] for j in top_idx][:76]
#         important_words_info[i]['token_id']         = [important_words_info[i]['token_id'][j] for j in top_idx][:76]
        
#     else:
#         pass
    
# all_len_important = []
# for sample_id in important_words_info.keys():
#     all_len_important.append(len(important_words_info[sample_id]['token_position']))
# print(np.mean(all_len_important))


# # Get NN MASKER dataset (for R1-R12 and train/val/test)

# In[104]:


# class NNMASKERDataset(Dataset):
    
#     def __init__(self, dataset, important_words_info, tokenizer):
#         self.tokenizer = tokenizer
#         self.model_mask_id = self.tokenizer(self.tokenizer.mask_token)['input_ids'][1]
#         self.important_words_info  = important_words_info
#         self.make_MASKER_dataset(dataset)
        
#     def __len__(self):
#         return len(self.all_input_ids)

#     def __getitem__(self, idx):
        
#         sample = {'all_input_ids'  : self.all_tokens[idx],
#                   'attention_mask' : torch.ones_like(self.all_tokens[idx][0]),
#                   'all_labels'     : self.all_labels[idx],
#                   'orig_text'      : self.all_orig_text_list[idx]}
#         return sample
    
#     # Use important_words_info to make the label for training TokenClassification on this D(task) dataset
#     def make_MASKER_dataset(self, dataset):
        
#         self.all_tokens         = []
#         self.all_MKR_labels     = []
#         self.all_label_list     = []
#         self.all_orig_text_list = []

#         for idx, data in enumerate(dataset):  
#             sys.stdout.write(str(idx))
            
#             orig_input_ids = data['input_ids'].reshape(1,-1)
#             label          = data['labels']
            
#             MKR_tokens = orig_input_ids.detach().cpu().clone().long() # masked token (Masked Keywords Reconstruction)
#             MER_tokens = orig_input_ids.detach().cpu().clone().long() # outlier token (Masked Entropy Regularization)
            
#             # if this sample contains important tokens
#             if idx in self.important_words_info.keys():
#                 # print("yes")

#                 # print(data['input_ids'])
#                 # print(data['attention_mask'])
#                 # print(data['labels'])

#                 important_token_positions = self.important_words_info[idx]['token_position']
#                 important_token_ids       = self.important_words_info[idx]['token_id']
#                 important_token_ids       = [int(i[0]) for i in important_token_ids]

#                 # check if the token that we want to mask matches with the tokens from our important_word_info
#                 tokens_to_mask = torch.index_select(orig_input_ids, dim = 1, index = torch.tensor(important_token_positions, dtype=torch.int64)).tolist()
#                 # print(important_token_ids , tokens_to_mask)
#                 assert important_token_ids == tokens_to_mask[0]
                
#                 # fill the important token position with [MASK]
#                 MKR_tokens.index_fill_(dim = 1, index = torch.tensor(important_token_positions).long(), value = self.model_mask_id)
#                 # print(MKR_tokens)
                
#                 # put the original tokens as label of the [MASK] tokens
#                 MKR_labels = (torch.ones_like(orig_input_ids) * -100).detach().cpu().long()
#                 MKR_labels.index_put_(indices = (torch.zeros_like(torch.tensor(important_token_positions)).long(), torch.tensor(important_token_positions).long()) , values = torch.tensor(important_token_ids).long()).squeeze(0)
#                 # print(MKR_labels)
                
#                 idx_non_keyword = torch.tensor([i for i in range(1, orig_input_ids.shape[1]-1) if i not in important_token_positions])
#                 # print(idx_non_keyword)
#                 MER_tokens.index_fill_(dim = 1, index = idx_non_keyword, value = int(self.model_mask_id))
                
#             else :
#                 MKR_labels = (torch.ones_like(orig_input_ids) * -100).detach().cpu() # mask nothing ignore all position = -100
#                 MER_tokens = (torch.ones_like(orig_input_ids) * int(self.model_mask_id)).detach().cpu() # mask everything

#             self.all_tokens.append((orig_input_ids, MKR_tokens, MER_tokens))
#             self.all_MKR_labels.append(MKR_labels)
#             self.all_label_list.extend([label])
#             self.all_orig_text_list.extend([self.tokenizer.decode(orig_input_ids.clone().squeeze(0))])
            
#         self.all_labels = [i for i in zip(self.all_MKR_labels, self.all_label_list)]
            
    


# In[105]:


# masker_dataset = NNMASKERDataset(this_dataset, important_words_info, tokenizer)

# with open(f'./data/MASKER/MASKER-NN-{split}-R{class_ratio}.pkl', 'wb') as outp:
#     pickle.dump(masker_dataset, outp, pickle.HIGHEST_PROTOCOL)


# In[42]:


# import pickle

# train_dataset = pickle.load(open(f"./data/MASKER/MASKER-NN-train-R1.pkl", "rb"))


# In[ ]:




