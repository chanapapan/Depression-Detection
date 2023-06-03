#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[2]:


import argparse
parser  = argparse.ArgumentParser()
parser.add_argument('-f', '--ratio', help='ratio of control user to depression user' , type=int, required=True)
args     = parser.parse_args()


# In[3]:


debug = False

# class_ratio          = 10   # 1,2,4,6,8,10
class_ratio          = args.ratio

training_obj       = 'classiCE'
masking_method     = None
keyword_path       = None

classifier_p_dropout = 0.1

learning_rate = 5e-7 
batch_size    = 32
num_epoch     = 50
 
val_steps     = 100
logging_steps = 30
max_file_save = 5

tokenizer_checkpoint = 'bert-base-uncased'


if debug == True:
    class_ratio   = 1
    save_dir      = f'./save/debug-BASE-{training_obj}r{class_ratio}'
    writer        = SummaryWriter(log_dir = f'./tensorboard_runs/FINALv1_runs/debug-BASE-{training_obj}r{class_ratio}')
    num_epoch     = 1
    val_steps     = 10
    logging_steps = 10
else:
    save_dir      = f'./save/BASE-{training_obj}r{class_ratio}'
    writer        = SummaryWriter(log_dir = f'./tensorboard_runs/FINALv1_runs/BASE-{training_obj}r{class_ratio}')
    
print("save_dir       : ", save_dir)
print("class_ratio    : ", class_ratio)
print("training_obj   : ", training_obj)
print("masking_method : ", masking_method)
print("keyword_path   : ", keyword_path)


# In[4]:


train_dataset = pickle.load(open(f"./data/classi/classichunk-R{class_ratio}-train-ds.pkl", "rb"))
val_dataset   = pickle.load(open(f"./data/classi/classichunk-R{class_ratio}-val-ds.pkl", "rb"))
test_dataset  = pickle.load(open(f"./data/classi/classichunk-R{class_ratio}-test-ds.pkl", "rb"))

datasets = {'train' : train_dataset, 'val' : val_dataset, 'test':test_dataset}

print(len(datasets['train']))
print(len(datasets['val']))
print(len(datasets['test']))


# In[5]:


for sample in train_dataset:
    print(sample['input_ids'])
    print(sample['word_ids'])
    print(sample['attention_mask'])
    print(sample['orig_text'])
    print(sample['labels'])
    break


# In[6]:


if training_obj == 'classiCE' :
    criterion  = nn.CrossEntropyLoss(reduction = 'mean')
    num_labels = 2 # CE
    model      = AutoModelForSequenceClassification.from_pretrained(tokenizer_checkpoint, num_labels = num_labels).to(device)
    model.classifier.dropout = nn.Dropout(p = classifier_p_dropout, inplace = False)
    print(model.classifier)
    
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
        
        self.create_dataloaders()
        
        self.all_losses = dict()
        self.all_losses['train_loss'] = AverageMeter()
        self.all_losses['val_loss']  = AverageMeter()
        self.all_losses['test_loss'] = AverageMeter()
            
        if training_obj == 'classiCE':
            self.all_metrics = dict()
            self.all_metrics['train_metrics'] = ClassiMetricMeter()
            self.all_metrics['val_metrics']   = ClassiMetricMeter()
            self.all_metrics['test_metrics']  = ClassiMetricMeter()
            self.train_one_step = TrainEval.train_one_step_CE
            self.evaluate       = TrainEval.evaluate_CE
            print("Training Sequence Classification with CE Loss...")
            
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
            'train_loss'          : self.all_losses['train_loss'].latest_value,
            'val_loss'            : self.all_losses['val_loss'].latest_value,
            
            }, f'{self.save_dir}/best-model-{self.best_step}.tar')
    
          
    def save_checkpoint(self,):
        
        if int(self.current_step - (self.val_steps*self.max_file_save)) > 0:
            os.remove(f'{self.save_dir}/checkpoint-{ int(self.current_step - (self.val_steps*self.max_file_save))}.tar')
        
        torch.save({
            'step'                : self.current_step,
            'model_state_dict'    : self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss'          : self.all_losses['train_loss'].latest_value,
            'val_loss'            : self.all_losses['val_loss'].latest_value,
            
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


# In[9]:


trainer.train()


# In[ ]:




