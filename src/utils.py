import os, math
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from datetime import datetime

# to fix no display problem
import matplotlib
matplotlib.use('Agg')

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >gpu_free')
    memory_available = [int(x.split()[2]) for x in open('gpu_free', 'r').readlines()]
    gpu = f'cuda:{np.argmax(memory_available)}'
    if os.path.exists("gpu_free"):
        os.remove("gpu_free")
    else:
          print("The file does not exist") 
    return gpu

class AverageMeter(object):
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.latest_value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.latest_value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, latest_value, n=1):
        self.latest_value  = latest_value
        self.sum   += latest_value * n
        self.count += n
        self.average = self.sum / self.count

class ClassiMetricMeter(object):

    def __init__(self):
        self.labels_list = []
        self.logits_list = []
        self.metrics = {} # do not reset bc we keep accumulating the results

    def reset(self):
        self.labels_list = []
        self.logits_list = []
        self.pred = None

    def keep(self, logits, labels):
        self.labels_list.append(labels.cpu().detach().numpy())
        self.logits_list.append(logits.cpu().detach().numpy())
        
    # Calculate, print, then RESET
    def calculate(self, current_step, writer, mode, training_obj):
        
        labels_np = np.concatenate(self.labels_list, axis=0)
        logits_np = np.concatenate(self.logits_list, axis=0)
        
        if training_obj == 'classiBCE':
            self.pred = np.round(logits_np)
        if training_obj in ['classiCE', 'MASKER'] :
            self.pred = np.argmax(logits_np, axis = 1)
        if training_obj == 'classitoken':
            self.pred = np.argmax(logits_np, axis = 2)
            self.pred = self.pred.flatten()
            labels_np = labels_np.flatten()
            
        # print(self.pred.shape)
        # print(labels_np.shape)
        print("Pred   : ", self.pred.flatten().tolist()[:20])
        print("Actual : ", labels_np.flatten().tolist()[:20])
        
        self.metrics[current_step] = {}
        
        # print(self.pred.flatten())
        # print(labels_np.flatten())
        # print(self.metrics)
        # print(mode)
        # print(confusion_matrix(labels_np, self.pred))      
        # print(confusion_matrix(labels_np, self.pred).ravel())
        
        self.metrics[current_step]['TN'], self.metrics[current_step]['FP'], self.metrics[current_step]['FN'], self.metrics[current_step]['TP'] = confusion_matrix (labels_np, self.pred, labels=[0,1]).ravel()
        self.metrics[current_step]['Accuracy']   = accuracy_score   (labels_np, self.pred)
        self.metrics[current_step]['Precision']  = precision_score  (labels_np, self.pred)
        self.metrics[current_step]['Recall']     = recall_score     (labels_np, self.pred)
        self.metrics[current_step]['F1']         = f1_score         (labels_np, self.pred)
        
        try:
            roc_score = roc_auc_score(labels_np, logits_np[:, 1])
        except ValueError :
            roc_score = 0
            
        self.metrics[current_step]['roc_auc_score']  = roc_score
        
        # put TRAIN loss and metrics into tensorboard
        writer.add_scalar(tag = f"{mode}/TN",        scalar_value = self.metrics[current_step]['TN'],        global_step = current_step)
        writer.add_scalar(tag = f"{mode}/FP",        scalar_value = self.metrics[current_step]['FP'],        global_step = current_step)
        writer.add_scalar(tag = f"{mode}/FN",        scalar_value = self.metrics[current_step]['FN'],        global_step = current_step)
        writer.add_scalar(tag = f"{mode}/TP",        scalar_value = self.metrics[current_step]['TP'],        global_step = current_step)
        writer.add_scalar(tag = f"{mode}/Accuracy",  scalar_value = self.metrics[current_step]['Accuracy'],  global_step = current_step)
        writer.add_scalar(tag = f"{mode}/Precision", scalar_value = self.metrics[current_step]['Precision'], global_step = current_step)
        writer.add_scalar(tag = f"{mode}/Recall",    scalar_value = self.metrics[current_step]['Recall'],    global_step = current_step)
        writer.add_scalar(tag = f"{mode}/F1",        scalar_value = self.metrics[current_step]['F1'],        global_step = current_step)
        writer.add_scalar(tag = f"{mode}/roc_auc_score", scalar_value = self.metrics[current_step]['roc_auc_score'], global_step = current_step)
        
        print("Metrics : " , self.metrics[current_step])
        
        if mode == 'test':
            our_fpr, our_tpr, _  = roc_curve    (labels_np, logits_np[:, 1], pos_label=1)

            random_probs    = [0 for i in range(len(labels_np))]
            base_fpr, base_tpr, _ = roc_curve(labels_np, random_probs, pos_label=1)
            
            print(" our_fpr : ", our_fpr)
            print(" our_tpr : ", our_tpr)
            print(" base_fpr : ", base_fpr)
            print(" base_tpr : ", base_tpr)
            
            date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
            
            plt.plot(our_fpr,  our_tpr,  linestyle='--',color='orange')
            plt.plot(base_fpr, base_tpr, linestyle='--', color='blue')
            plt.title('ROC curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive rate')
            plt.legend(loc='best')
            plt.savefig(f'ROC_{mode}_{training_obj}_{current_step}_{date}', dpi=300)
            writer.add_figure('roc_auc_score', plt.gcf())
        
        self.reset()
            
