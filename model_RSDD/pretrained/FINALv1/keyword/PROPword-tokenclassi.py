#!/usr/bin/env python
# coding: utf-8

# In[6]:


from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification

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


# ## 01/06 Method

# - Get correctly classified samples from depression and control class
# - Get the top 76 words by word's average attention score in each sample
# - Masked those words

# # Create Token classi dataset from classi train R10 dataset
# 
# ### Mask top 76 tokens with highest attention score in each correctly classified sample

# In[2]:


# load_model_from = '../save/BASE-classiCEr1/best-model-1500.tar'

# checkpoint           = 'bert-base-uncased'

# training_obj       = 'classiCE'

# classifier_p_dropout = 0.1

# num_labels = 2 #CE
# model      = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = num_labels).to(device)
# model.classifier.dropout = nn.Dropout(p = classifier_p_dropout, inplace = False)

# print("Load from ", load_model_from)
# checkpoint = torch.load(load_model_from)
# model.load_state_dict(checkpoint['model_state_dict'])
# print(model.load_state_dict(checkpoint['model_state_dict']))

# # from transformers import AutoTokenizer
# # checkpoint   = 'bert-base-uncased'
# # tokenizer    =  AutoTokenizer.from_pretrained(checkpoint)


# In[7]:


class ProposedTokenClassificationDataset(Dataset):
    
    def __init__(self, dataset, model, device):
        
        self.make_label_tokenclassification(dataset)
        
    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, idx):
        sample = {'input_ids'    : self.all_input_ids[idx],
                'attention_mask' : torch.ones_like(self.all_input_ids[idx]),
                'labels'         : self.all_labels[idx],
                'text'           : self.all_text[idx]
                 }
        return sample
    
    # Use important_words_info to make the label for training TokenClassification on this D(task) dataset
    def make_label_tokenclassification(self, dataset):
        all_input_ids = []
        all_labels    = []
        all_text      = []
        
        
        for idx , data in enumerate(dataset):
            sys.stdout.write(str(idx))
            
            orig_input_ids  = data['input_ids'].to(device).reshape(1,-1)
            orig_labels     = data['labels']
            
            label          = torch.zeros_like(orig_input_ids).cpu()
            
            # ---------------- MKR -----------------
            model.eval()
            with torch.no_grad():
                outputs   = model(orig_input_ids, output_attentions=True)        
                attention = outputs.attentions[-1].sum(dim = 1).detach().cpu() # get attention of last layer (1, 12, 512, 512) >> (1, 512, 512)
            # print(attention.shape)

            # get attention from only CORRECTLY classified samples
            pred = int(torch.argmax(torch.softmax(outputs.logits.detach().cpu(), dim = 1), dim = 1))
            correct = pred == orig_labels
            # print(correct)

            if correct :   
                
                word_ids = data['word_ids']
                word_score_dict = {}
                
                # print(word_ids)
                # print(orig_input_ids)
                # print(attention[0,0,:])
                
                words_score = {} # k = index of word : v = avg score of word
                for word_id in word_ids[1:-1]:
                    idx_this_word = torch.tensor([input_idx for input_idx in range(1,orig_input_ids.shape[1]-1) if word_id == word_ids[input_idx]])
                    
                    if idx_this_word.size()[0] > 1:
                        # print(int(idx_this_word[0]), int(idx_this_word[-1]))
                        avg_word_score = attention[0, 0, int(idx_this_word[0]) : int(idx_this_word[-1])+1]
                        # print(avg_word_score)
                    else:
                        avg_word_score = attention[0, 0, int(idx_this_word)]
                        # print(avg_word_score)
                    
                    # print(avg_word_score)
                    avg_word_score = torch.mean(avg_word_score)
                    # print(avg_word_score.shape)
                    # print(avg_word_score)
                    
                    words_score[idx_this_word] = avg_word_score
                    # print(words_score)
                
                sorted_words_score = sorted(words_score.items(), key=lambda x: x[1], reverse=True)   
                # print(sorted_words_score)
                
                top_indices = []
                for index, score in sorted_words_score:
                    top_indices.extend(index.tolist())
                    if len(top_indices) >= 76:
                        break
                # print(len(top_indices))
            
                label.index_fill_(dim=1, index = torch.tensor(top_indices, dtype=torch.int64), value = 1)
            
            # first and last position of label should be 0 (CLS and SEP)
            label[0,0]    = 0
            label[0,-1]   = 0

            all_input_ids.append(data['input_ids'].squeeze(0))
            all_labels.append(label.squeeze(0))
            all_text.append(data['orig_text'])       
            
        self.all_input_ids = all_input_ids
        self.all_labels    = all_labels
        self.all_text      = all_text
        
        del all_input_ids, all_labels, all_text


# In[5]:


# train_dataset = pickle.load(open(f'../data/classi/classichunk-R1-train-ds.pkl', "rb"))

# tokenclassi_all_ds = ProposedTokenClassificationDataset(train_dataset, model, device)

# print(len(tokenclassi_all_ds))

# with open('../keyword/PROP-tokenclassi-R1dataset.pkl', 'wb') as outp:
#     pickle.dump(tokenclassi_all_ds, outp, pickle.HIGHEST_PROTOCOL)


# In[8]:


tokenclassi_all_ds = pickle.load(open('./PROP-tokenclassi-R1dataset.pkl', "rb"))


# In[9]:


for idx, sample in enumerate(tokenclassi_all_ds):
    print(sample['input_ids'])
    print(sample['labels'])
    print(sample['attention_mask'])
    break


# # Train Token Classification Model

# In[10]:


from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.1
VAL_SIZE  = 0.1

len_ds = len(tokenclassi_all_ds)

all_idx = list(range(len_ds))
random.seed(42)
random.shuffle(all_idx)

train_indices = all_idx [ : int(len_ds*(1-TEST_SIZE-VAL_SIZE)) ]
val_indices   = all_idx [ int(len_ds*(1-TEST_SIZE-VAL_SIZE)) : int(len_ds*(1-TEST_SIZE))]
test_indices  = all_idx [ int(len_ds*(1-TEST_SIZE)) : ]

# generate subset based on indices
train_dataset = Subset(tokenclassi_all_ds, train_indices)
val_dataset   = Subset(tokenclassi_all_ds, val_indices)
test_dataset  = Subset(tokenclassi_all_ds, test_indices)

print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))


# In[12]:


debug = False

tokenizer_checkpoint = 'bert-base-uncased'

training_obj       = 'classitoken'
masking_method     = None
keyword_path       = None 

classifier_p_dropout = 0.1

learning_rate = 5e-6
batch_size    = 32
num_epoch     = 100
 
val_steps     = 200
logging_steps = 50
max_file_save = 5


if debug == True:
    save_dir      = f'../save/debug-PROP-{training_obj}'
    writer        = SummaryWriter(log_dir = f'../tensorboard_runs/FINALv1_runs/debug-PROP-{training_obj}')
    num_epoch     = 2
    val_steps     = 10
    logging_steps = 10
else:
    save_dir      = f'../save/PROP-{training_obj}'
    writer        = SummaryWriter(log_dir = f'../tensorboard_runs/FINALv1_runs/PROP-{training_obj}')


# In[13]:


if training_obj == 'classitoken' :
    num_labels  = 2 # CE
    model       = AutoModelForTokenClassification.from_pretrained(tokenizer_checkpoint, num_labels = num_labels).to(device)
    model.classifier.dropout = nn.Dropout(p = classifier_p_dropout, inplace = False)


# In[14]:


class TrainEvalLoop():
    
   # Token classification : use outputs.loss    
    @classmethod      
    def train_one_step_tokenclassi(cls, data):
        
            cls.current_step += 1
            
            input_ids = data['input_ids'].to(cls.device)
            att_mask  = data['attention_mask'].to(cls.device)
            labels    = data['labels'].to(cls.device).long()

            cls.optimizer.zero_grad()

            outputs = cls.model(input_ids=input_ids, attention_mask=att_mask, labels=labels)

            # classification
            logits = outputs.logits # (before SoftMax)
            loss   = outputs.loss
            
            loss.backward()
            cls.optimizer.step()
            
            cls.all_losses['train_loss'].update(loss.item(), input_ids.shape[0])
                          
            # if it is the eval step then cal Loss and Metrics of THIS STEP
            if (cls.current_step % cls.val_steps == 0 or cls.current_step % cls.logging_steps == 0) and cls.current_step != 0:
                # record TRAIN loss
                cls.writer.add_scalar(tag = "train/loss", scalar_value = loss.item(), global_step = cls.current_step)
                print(f"Epoch : {cls.current_epoch} | Step : {cls.current_step}")
                print("Train Loss    : " , loss.item())
                # TRAIN metrics
                cls.all_metrics['train_metrics'].keep(torch.softmax(logits, dim = 2), labels) # bc our logits did not pass sigmoid or softmax yet
                cls.all_metrics['train_metrics'].calculate(cls.current_step, cls.writer, 'train', cls.training_obj)
        
    # Token classification : use outputs.loss       
    @classmethod
    def evaluate_tokenclassi(cls, mode):
        cls.all_losses[f'{mode}_loss'].reset()
        
        if mode == 'test':
            data_loader = cls.test_loader
        if mode == 'val':
            data_loader = cls.val_loader
            
        for data in data_loader:    
                
            # Every data instance is an input + label pair
            input_ids = data['input_ids'].to(cls.device)
            att_mask  = data['attention_mask'].to(cls.device)
            labels    = data['labels'].to(cls.device).long()

            if mode == 'val':
                outputs = cls.model(input_ids=input_ids, attention_mask=att_mask, labels=labels)
            if mode == 'test':
                outputs = cls.best_model(input_ids=input_ids, attention_mask=att_mask, labels=labels)

            # classification
            logits = outputs.logits # (before SoftMax)
            loss   = outputs.loss
            
            cls.all_losses[f'{mode}_loss'].update(loss.item(), input_ids.shape[0])
            cls.all_metrics[f'{mode}_metrics'].keep(torch.softmax(logits, dim = 2), labels) # bc our logits did not pass sigmoid or softmax yet    
        
        # report Loss and Metrics of the Whole VAL SET
        print(f"{mode} Loss    : " , cls.all_losses[f'{mode}_loss'].average)
        cls.writer.add_scalar(tag = f"{mode}/loss", scalar_value = cls.all_losses[f'{mode}_loss'].average, global_step = cls.current_step)
        cls.all_metrics[f'{mode}_metrics'].calculate(cls.current_step, cls.writer, mode, cls.training_obj)


# In[15]:


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

training_arg = {
                'device'       : device, 
                'model'        : model,
                'training_obj' : training_obj, 
                'criterion'    : None,
                'optimizer'    : optimizer,
                'batch_size'   : batch_size,
                'num_epoch'    : num_epoch,
                
                'train_dataset' : train_dataset,
                'val_dataset'   : val_dataset, 
                'test_dataset'  : test_dataset,
                
                'val_steps'     : val_steps,
                'logging_steps' : logging_steps, 
                'save_dir'      : save_dir,
    
                'max_file_save' : max_file_save,
                'writer'        : writer,
                
                'load_model_from' : None, # saved model path
}

TrainEval = TrainEvalLoop()


# In[16]:


class CustomTrainer():
    def __init__(self, training_arg, TrainEval):
        
        self.load_model_from = training_arg['load_model_from']
        
        self.train_dataset = training_arg['train_dataset']
        self.val_dataset   = training_arg['val_dataset']
        self.test_dataset  = training_arg['test_dataset']
        
        self.device        = training_arg['device']
        self.model         = training_arg['model']
        self.training_obj  = training_arg['training_obj']
        self.criterion     = training_arg['criterion']
        self.optimizer     = training_arg['optimizer']
        self.batch_size    = training_arg['batch_size']
        self.num_epoch     = training_arg['num_epoch']
        
        self.val_steps     = training_arg['val_steps']
        self.logging_steps = training_arg['logging_steps']
        self.save_dir      = training_arg['save_dir']
        self.max_file_save = training_arg['max_file_save']
        self.writer        = training_arg['writer']
        self.best_model    = None
        
        CHECK_FOLDER = os.path.isdir(self.save_dir)
        # If folder doesn't exist, then create it.
        if not CHECK_FOLDER:
            os.makedirs(self.save_dir)
            print("created folder : ", self.save_dir)
        
        self.create_dataloaders()
        
        self.all_losses = dict()
        self.all_losses['train_loss'] = AverageMeter()
        self.all_losses['val_loss']   = AverageMeter()
        self.all_losses['test_loss']  = AverageMeter()
        
        if training_obj == 'classitoken':
            self.all_metrics = dict()
            self.all_metrics['train_metrics'] = ClassiMetricMeter()
            self.all_metrics['val_metrics']   = ClassiMetricMeter()
            self.all_metrics['test_metrics']  = ClassiMetricMeter()
            self.train_one_step = TrainEval.train_one_step_tokenclassi
            self.evaluate       = TrainEval.evaluate_tokenclassi
            print("Training Token Classification with Huggingface's Loss...")
            
        if self.load_model_from is not None:
            self.load_checkpoint()
        
        self.current_step = 0
        self.current_epoch = 0
    
    def create_dataloaders(self, ):
        self.train_loader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True, pin_memory=True)
        del self.train_dataset
        self.val_loader   = DataLoader(self.val_dataset,   batch_size = self.batch_size, shuffle = True, pin_memory=True)
        del self.val_dataset
        self.test_loader  = DataLoader(self.test_dataset,  batch_size = self.batch_size, shuffle = True, pin_memory=True)
        del self.test_dataset
        
        self.total_steps  = int(self.num_epoch * len(self.train_loader))
        print(f"{self.num_epoch} epochs = {self.total_steps} steps")
    
    def train(self,):
        self.best_step  = 0
        self.best_model = copy.deepcopy(self.model)
        tmp_val_loss = 10e+9
        
        for _ in range(self.num_epoch): # for epoch
            
            for i, data in enumerate(self.train_loader): # for data in data_loader
                sys.stdout.write(str(i))
                
                self.model.train(True)
                self.train_one_step.__func__(self, data)
                
                # do evaluation every val_steps
                if self.current_step % self.val_steps == 0 and self.current_step != 0:
                    
                    self.model.eval()
                    with torch.no_grad():
                        self.evaluate.__func__(self, 'val')
                    
                    self.writer.flush()
                    self.save_checkpoint()
                    
                    curr_val_loss = self.all_losses['val_loss'].average
                    
                    # keep a copy of the current best model
                    if curr_val_loss < tmp_val_loss:
                        clear_output(wait=True)
                        tmp_val_loss    = curr_val_loss
                        self.best_step  = self.current_step
                        self.best_model = copy.deepcopy(self.model)
                        print("Best step : ", self.best_step, "Val loss at Best Step : ",  curr_val_loss)
                        
                
            self.all_losses['train_loss'].reset()  # for train loss we reset every epoch 
            self.current_epoch += 1
        
        self.save_best_model()
        self.writer.close()
        print("==================== Finished Training !!! =======================")
        
        self.inference()
        
        
    def inference(self,): # test with best model
        self.best_model.eval()
        with torch.no_grad():
            self.evaluate.__func__(self, 'test')
            
    def save_best_model(self, ):
        torch.save({
            'step'                : self.best_step,
            'model_state_dict'    : self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss'          : self.all_losses['train_loss'].average,
            'val_loss'           : self.all_losses['val_loss'].average,
            
            }, f'{self.save_dir}/best-model-{self.best_step}.tar')
    
          
    def save_checkpoint(self,):
        
        if int(self.current_step - (self.val_steps*self.max_file_save)) > 0:
            os.remove(f'{self.save_dir}/checkpoint-{ int(self.current_step - (self.val_steps*self.max_file_save))}.tar')
        
        torch.save({
            'step'                : self.current_step,
            'model_state_dict'    : self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss'          : self.all_losses['train_loss'].average,
            'val_loss'            : self.all_losses['val_loss'].average,
            
            }, f'{self.save_dir}/checkpoint-{self.current_step}.tar')
        
    def load_checkpoint(self,): # only for continuing the next task (NOT for continue training)
        print("Load from ", self.load_model_from)
        checkpoint = torch.load(self.load_model_from)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(self.model.load_state_dict(checkpoint['model_state_dict'])) # <All keys matched successfully>
        
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # self.current_step = checkpoint['step']
        # loss = checkpoint['train_loss']


# In[17]:


trainer = CustomTrainer(training_arg, TrainEval)


# In[ ]:


trainer.train()

