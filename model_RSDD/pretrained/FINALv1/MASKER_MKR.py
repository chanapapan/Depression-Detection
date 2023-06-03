#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader
from IPython.display import clear_output
import torch.nn as nn
import copy
import sys, os, pickle
import torch.nn.functional as F

os.environ['TRANSFORMERS_CACHE'] = './cache/'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from src.dataset import *
from src.utils   import *
from src.traineval  import *

torch.cuda.empty_cache()

SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = get_freer_gpu()
print('device', device)


# In[2]:


import argparse
parser  = argparse.ArgumentParser()
parser.add_argument('-f', '--ratio', help='ratio of control user to depression user' , type=int, required=True)
parser.add_argument('-k', '--kwname', help='keyword name' , type=str, required=True)
args     = parser.parse_args()


# In[ ]:


#########################################################################
#########################################################################

class_ratio          = args.ratio
# masking_method       = 'random'
keyword_name         = args.kwname # random

# class_ratio          = args.ratio
masking_method       = 'keywords'
# keyword_name         = 'logodds'

# class_ratio          = args.ratio
# masking_method       = 'keywords'
# keyword_name         = 'tfidf'

# class_ratio          = args.ratio
# masking_method       = 'keywords'
# keyword_name         = 'sumatt'

# class_ratio          = args.ratio
# masking_method       = 'keywords'
# keyword_name         = args.kwname

# class_ratio          = args.ratio
# masking_method       = 'keywords'
# keyword_name         = 'PROPprob'

#########################################################################
#########################################################################


# In[5]:


debug = False

training_obj         = 'MASKER'
tokenizer_checkpoint = 'bert-base-uncased'

classifier_p_dropout = 0.1

learning_rate = 1e-6
batch_size    = 8
num_epoch     = 10
 
val_steps     = 100
logging_steps = 30
max_file_save = 5

tokenizer_checkpoint = 'bert-base-uncased'

if debug == True:
    save_dir      = f'./save/debug-MKR-{training_obj}-{keyword_name}-r{class_ratio}'
    writer        = SummaryWriter(log_dir = f'./tensorboard_runs/FINALv1_runs/debug-MKR-{training_obj}-{keyword_name}-r{class_ratio}')
    num_epoch     = 1
    val_steps     = 10
    logging_steps = 10
else:
    save_dir      = f'./save/MKR-{training_obj}-{keyword_name}-r{class_ratio}'
    writer        = SummaryWriter(log_dir = f'./tensorboard_runs/FINALv1_runs/MKR-{training_obj}-{keyword_name}-r{class_ratio}')

print("save_dir       : ", save_dir)
print("class_ratio    : ", class_ratio)
print("training_obj   : ", training_obj)
print("masking_method : ", masking_method)


# In[7]:

print(f"Load Data from ./data/MASKER/MASKER-{keyword_name}-R{class_ratio}-train-ds.pkl")

train_dataset = pickle.load(open(f'./data/MASKER/MASKER-{keyword_name}-R{class_ratio}-train-ds.pkl', "rb"))
val_dataset   = pickle.load(open(f'./data/MASKER/MASKER-{keyword_name}-R{class_ratio}-val-ds.pkl', "rb"))
test_dataset  = pickle.load(open(f'./data/MASKER/MASKER-{keyword_name}-R{class_ratio}-test-ds.pkl', "rb"))

datasets = {'train' : train_dataset, 'val' : val_dataset, 'test':test_dataset}

print(len(datasets['train']))
print(len(datasets['val']))
print(len(datasets['test']))


# In[5]:


for idx, sample in enumerate(datasets['train']):
    orig_token, MKR_tokens, MER_tokens = sample['all_input_ids']
    MKR_labels, classi_labels = sample['all_labels']

    print(MKR_tokens)
    print(MER_tokens)
    print(MKR_labels)
    
    print(classi_labels)
    
    print(sample['word_ids'])
    print(sample['orig_text'])
    break


# In[6]:


class MaskerNet_MKR(nn.Module):
    """ Makser network """

    def __init__(self, backbone,  tokenizer, n_classes = 2,):
        super(MaskerNet_MKR, self).__init__()
        self.backbone = backbone
        self.dropout     = nn.Dropout(0.1)
        self.n_classes   = n_classes
        self.vocab_size  = tokenizer.vocab_size

        self.dense = nn.Linear(768,768)
        self.net_cls = nn.Linear(768, n_classes)  # classification layer
        self.net_ssl = nn.Sequential(  # self-supervision layer
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, self.vocab_size),
        )

    def forward(self, X_s):
        
        x_orig, x_mask =  X_s[0], X_s[1]
        attention_mask = (x_orig > 0).float()

        out_cls = self.backbone(x_orig, attention_mask)[1]  # pooled feature
        out_cls = self.dropout(out_cls)
        out_cls = self.net_cls(out_cls)  # classification

        out_ssl = self.backbone(x_mask, attention_mask)[0]  # hidden feature
        out_ssl = self.dropout(out_ssl)
        out_ssl = self.net_ssl(out_ssl)  # self-supervision

        # out_ood = self.backbone(x_ood, attention_mask)[1]  # pooled feature
        # out_ood = self.dropout(out_ood)
        # out_ood = self.net_cls(out_ood)  # classification (outlier)

        return out_cls, out_ssl
    
    def inference(self, x_orig):
        
        attention_mask = (x_orig > 0).float() # everywhere that is not padding, attention mask is 1

        out_cls = self.backbone(x_orig, attention_mask)[1]  # pooled feature
        out_cls = self.dropout(out_cls)
        out_cls = self.net_cls(out_cls)  # classification
        
        return out_cls


# In[7]:


if training_obj == 'MASKER' :
    CE        = nn.CrossEntropyLoss()
    CE_MKR    = nn.CrossEntropyLoss(ignore_index = -100) # masked keyword reconstruction
    backbone  = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model     = MaskerNet_MKR(backbone, tokenizer).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


training_arg = {
                'device'       : device, 
    
                'model'        : model,
                'training_obj' : training_obj, 
                'criterion'    : (CE, CE_MKR),
                'optimizer'    : optimizer,
                'batch_size'   : batch_size,
                'num_epoch'    : num_epoch,
                
                'train_dataset' : datasets['train'],
                'val_dataset'   : datasets['val'], 
                'test_dataset'  : datasets['test'],
                
                'val_steps'     : val_steps,
                'logging_steps' : logging_steps, 
                'save_dir'      : save_dir,
    
                'max_file_save' : max_file_save,
                
                'writer'        : writer,
                
                'load_model_from' : None, # saved model path
    
}

TrainEval = TrainEvalLoop()


# In[8]:


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
            
        if training_obj == 'MASKER':
            # different losses than others
            self.all_losses = dict()
            self.all_losses['train_sum_loss']  = AverageMeter()
            self.all_losses['train_CE_loss']   = AverageMeter()
            self.all_losses['train_MKR_loss']  = AverageMeter()
            # self.all_losses['train_MER_loss']  = AverageMeter()
            self.all_losses['val_CE_loss']     = AverageMeter()
            self.all_losses['test_CE_loss']    = AverageMeter()

            # metrics of CLS head
            self.all_metrics = dict()
            self.all_metrics['train_metrics'] = ClassiMetricMeter()
            self.all_metrics['val_metrics']   = ClassiMetricMeter()
            self.all_metrics['test_metrics']  = ClassiMetricMeter()
            
            self.train_one_step = TrainEval.train_one_step_MASKER_MKR
            self.evaluate       = TrainEval.evaluate_MASKER
            
            self.criterion_CE  = self.criterion[0]
            self.criterion_MKR = self.criterion[1]
            print("Training MASKER ...")
            
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
        tmp_val_F1 = 10e-9
        
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
                    
                    curr_val_F1 = self.all_metrics['val_metrics'].metrics[trainer.current_step]['F1']
                    
                    # keep a copy of the current best model
                    if curr_val_F1 > tmp_val_F1:
                        clear_output(wait=True)
                        tmp_val_F1    = curr_val_F1
                        self.best_step  = self.current_step
                        self.best_model = copy.deepcopy(self.model)
                        print("Best step : ", self.best_step, "Val F1 at Best Step : ",  curr_val_F1)
                        
                
            # for train loss we reset every epoch 
            self.all_losses['train_sum_loss'].reset()
            self.all_losses['train_CE_loss'].reset()
            self.all_losses['train_MKR_loss'].reset()
            # self.all_losses['train_MER_loss'].reset()
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
            'train_loss'          : self.all_losses['train_CE_loss'].latest_value,
            'val_loss'            : self.all_losses['val_CE_loss'].latest_value,
            
            }, f'{self.save_dir}/best-model-{self.best_step}.tar')
    
          
    def save_checkpoint(self,):
        
        if int(self.current_step - (self.val_steps*self.max_file_save)) > 0:
            os.remove(f'{self.save_dir}/checkpoint-{ int(self.current_step - (self.val_steps*self.max_file_save))}.tar')
        
        torch.save({
            'step'                : self.current_step,
            'model_state_dict'    : self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss'          : self.all_losses['train_CE_loss'].latest_value,
            'val_loss'            : self.all_losses['val_CE_loss'].latest_value,
            
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


# In[9]:


trainer = CustomTrainer(training_arg, TrainEval)


# In[10]:


trainer.train()


# In[ ]:





# In[ ]:




