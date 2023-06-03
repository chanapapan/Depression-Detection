#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForMaskedLM
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


# In[2]:


#########################################################################
#########################################################################

# masking_method     = 'random'
# keyword_name       = 'random'

# masking_method     = 'keywords'
# keyword_name       = 'logodds'

# masking_method     = 'keywords'
# keyword_name       = 'tfidf'

# masking_method     = 'keywords'
# keyword_name       = 'deplex'

# masking_method     = 'keywords'
# keyword_name       = 'sumatt'

# masking_method     = 'keywords'
# keyword_name       = 'NNrandom'

masking_method     = 'keywords'
keyword_name       = 'PROP'

#########################################################################
#########################################################################


# In[3]:


debug = False

training_obj       = 'MLM'

learning_rate = 1e-4 
batch_size    = 16
num_epoch     = 6

val_steps     = 1000
logging_steps = 100
max_file_save = 5

tokenizer_checkpoint = 'bert-base-uncased'

if debug == True:
    save_dir      = f'./save/debug-FURTHER-01-{training_obj}-{keyword_name}'
    writer        = SummaryWriter(log_dir = f'./tensorboard_runs/FINALv1_runs/debug-FURTHER-01-{training_obj}-{keyword_name}')
    num_epoch     = 1
    val_steps     = 10
    logging_steps = 10
else:
    save_dir      = f'./save/FURTHER-01-{training_obj}-{keyword_name}'
    writer        = SummaryWriter(log_dir = f'./tensorboard_runs/FINALv1_runs/FURTHER-01-{training_obj}-{keyword_name}')
    
print("save_dir       : ", save_dir)
print("training_obj   : ", training_obj)
print("masking_method : ", masking_method)
print("keyword_name   : ", keyword_name)


# In[4]:

print(f"Get dataset from ./data/MLM/MLM-{keyword_name}-train-ds.pkl")

train_dataset = pickle.load(open(f'./data/MLM/MLM-{keyword_name}-train-ds.pkl', "rb"))
val_dataset   = pickle.load(open(f'./data/MLM/MLM-{keyword_name}-val-ds.pkl', "rb"))
test_dataset  = pickle.load(open(f'./data/MLM/MLM-{keyword_name}-test-ds.pkl', "rb"))

datasets = {'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset}

print(len(datasets['train']))
print(len(datasets['val']))
print(len(datasets['test']))


# In[5]:


for idx, sample in enumerate(datasets['train']):
    print(sample['input_ids'])
    print(sample['word_ids'])
    print(sample['attention_mask'])
    print(sample['orig_text'])
    print(sample['masked_text'])
    print(sample['labels'])
    break


# In[6]:


if training_obj == 'MLM' :
    model     = AutoModelForMaskedLM.from_pretrained(tokenizer_checkpoint).to(device)
    criterion = nn.CrossEntropyLoss(reduction = 'mean')

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

training_arg = {
                'device'       : device, 
    
                'model'        : model,
                'training_obj' : training_obj, 
                'criterion'    : criterion,
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


# In[7]:


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
        
        print("Creating Data loaders")
        self.create_dataloaders()
        
        self.all_losses = dict()
        self.all_losses['train_loss'] = AverageMeter()
        self.all_losses['val_loss']  = AverageMeter()
        self.all_losses['test_loss'] = AverageMeter()
        
        if training_obj == 'classiBCE':
            self.all_metrics = dict()
            self.all_metrics['train_metrics'] = ClassiMetricMeter()
            self.all_metrics['val_metrics']   = ClassiMetricMeter()
            self.all_metrics['test_metrics']  = ClassiMetricMeter()
            self.train_one_step = TrainEval.train_one_step_BCE
            self.evaluate       = TrainEval.evaluate_BCE
            print("Training Classification with BCE Loss...")
            
        if training_obj == 'classiCE':
            self.all_metrics = dict()
            self.all_metrics['train_metrics'] = ClassiMetricMeter()
            self.all_metrics['val_metrics']   = ClassiMetricMeter()
            self.all_metrics['test_metrics']  = ClassiMetricMeter()
            self.train_one_step = TrainEval.train_one_step_CE
            self.evaluate       = TrainEval.evaluate_CE
            print("Training Classification with CE Loss...")
            
        if training_obj == 'MLM':
            self.all_perplexity = dict()
            self.all_perplexity['train_perplexity'] = AverageMeter()
            self.all_perplexity['val_perplexity']   = AverageMeter()
            self.all_perplexity['test_perplexity']  = AverageMeter()
            self.train_one_step = TrainEval.train_one_step_MLM
            self.evaluate       = TrainEval.evaluate_MLM
            print("Training Masked Language Model...")
            
        if self.load_model_from is not None:
            self.load_checkpoint()
        
        self.current_step = 0
        self.current_epoch = 0
    
    def create_dataloaders(self, ):
        self.train_loader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True) #, pin_memory=True)
        del self.train_dataset
        self.val_loader   = DataLoader(self.val_dataset,   batch_size = self.batch_size, shuffle = True) #, pin_memory=True)
        del self.val_dataset
        self.test_loader  = DataLoader(self.test_dataset,  batch_size = self.batch_size, shuffle = True) #, pin_memory=True)
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
        
        
    def inference(self,):
        self.model.eval()
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


# In[8]:


trainer = CustomTrainer(training_arg, TrainEval)

trainer.train()


# In[ ]:





# In[ ]:




