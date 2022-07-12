import torch
import torch.nn.functional as F
import math

class TrainEvalLoop():
    
    # simple binary classification task : CE Loss    
    @classmethod      
    def train_one_step_CE(cls, data):
        
            cls.current_step += 1
            
            # Every data instance is an input + label pair
            input_ids = data['input_ids'].to(cls.device)
            att_mask  = data['attention_mask'].to(cls.device)
            labels    = data['labels'].to(cls.device).long()

            # Zero your gradients for every batch!
            cls.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = cls.model(input_ids=input_ids, attention_mask=att_mask, labels=labels)

            # classification
            logits = outputs.logits # (before SoftMax)
            
            loss   = cls.criterion(logits, labels)
            
            loss.backward()
            cls.optimizer.step()
            
            # Gather data and report
            cls.all_losses['train_loss'].update(loss.item(), input_ids.shape[0])
                          
            # if it is the eval step then cal Loss and Metrics of THIS STEP
            if (cls.current_step % cls.val_steps == 0 or cls.current_step % cls.logging_steps == 0) and cls.current_step != 0:
                # record TRAIN loss
                cls.writer.add_scalar(tag = "train/loss", scalar_value = loss.item(), global_step = cls.current_step)
                print(f"Epoch : {cls.current_epoch} | Step : {cls.current_step}")
                print("Train Loss    : " , loss.item())
                # TRAIN metrics
                cls.all_metrics['train_metrics'].keep(torch.softmax(logits, dim = 1), labels) # bc our logits did not pass sigmoid or softmax yet
                cls.all_metrics['train_metrics'].calculate(cls.current_step, cls.writer, 'train', cls.training_obj)
        
    # simple binary classification task : CE Loss       
    @classmethod
    def evaluate_CE(cls, mode):
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
            loss   = cls.criterion(logits, labels)
            
            cls.all_losses[f'{mode}_loss'].update(loss.item(), input_ids.shape[0])
            cls.all_metrics[f'{mode}_metrics'].keep(torch.softmax(logits, dim = 1), labels) # bc our logits did not pass sigmoid or softmax yet    
        
        # report Loss and Metrics of the Whole VAL SET
        print(f"{mode} Loss    : " , cls.all_losses[f'{mode}_loss'].average)
        cls.writer.add_scalar(tag = f"{mode}/loss", scalar_value = cls.all_losses[f'{mode}_loss'].average, global_step = cls.current_step)
        cls.all_metrics[f'{mode}_metrics'].calculate(cls.current_step, cls.writer, mode, cls.training_obj)
    
    
    # MLM
    @classmethod       
    def train_one_step_MLM(cls, data):
        
        cls.current_step += 1
        
        # Every data instance is an input + label pair
        input_ids = data['input_ids'].to(cls.device)
        att_mask  = data['attention_mask'].to(cls.device)
        labels    = data['labels'].to(cls.device)

        # Zero your gradients for every batch!
        cls.optimizer.zero_grad()

        # Make predictions for this batch
        outputs = cls.model(input_ids=input_ids, attention_mask=att_mask, labels=labels)

        # MLM Loss
        loss   = outputs.loss
        perplexity = math.exp(loss.item())
        
        loss.backward()
        cls.optimizer.step()
        
        cls.all_losses['train_loss'].update(loss.item(), input_ids.shape[0])
        cls.all_perplexity['train_perplexity'].update(perplexity, input_ids.shape[0])
        
        # if it is the eval step report loss and perplexity of THIS STEP
        if (cls.current_step % cls.val_steps == 0 or cls.current_step % cls.logging_steps == 0) and cls.current_step != 0:
            # record TRAIN loss
            cls.writer.add_scalar(tag = "train/loss",       scalar_value = loss.item(), global_step = cls.current_step)
            cls.writer.add_scalar(tag = "train/perplexity", scalar_value = perplexity,  global_step = cls.current_step)
            print(f"Epoch : {cls.current_epoch} | Step : {cls.current_step}")
            print("Train Loss          : " , loss.item())
            print("Train Perplexity    : " , perplexity)


    # MLM
    @classmethod
    def evaluate_MLM(cls, mode):
        cls.all_losses[f'{mode}_loss'].reset()
        if mode == 'test':
            data_loader = cls.test_loader
        if mode == 'val':
            data_loader = cls.val_loader
            
        for data in data_loader:  
                
            input_ids = data['input_ids'].to(cls.device)
            att_mask  = data['attention_mask'].to(cls.device)
            labels    = data['labels'].to(cls.device)
            
            if mode == 'val':
                outputs = cls.model(input_ids=input_ids, attention_mask=att_mask, labels=labels)
            if mode == 'test':
                outputs = cls.best_model(input_ids=input_ids, attention_mask=att_mask, labels=labels)

            # MLM Loss
            loss   = outputs.loss
            perplexity = math.exp(loss.item())
            
            cls.all_losses[f'{mode}_loss'].update(loss.item(), input_ids.shape[0]) 
            cls.all_perplexity[f'{mode}_perplexity'].update(perplexity, input_ids.shape[0]) 
        
        # report Loss and Perplexity of the Whole VAL SET
        print(f"{mode} Loss       : " , cls.all_losses[f'{mode}_loss'].average)
        print(f"{mode} Perplexity : " , cls.all_perplexity[f'{mode}_perplexity'].average)
        cls.writer.add_scalar(tag = f"{mode}/loss",       scalar_value = cls.all_losses[f'{mode}_loss'].average,           global_step = cls.current_step)
        cls.writer.add_scalar(tag = f"{mode}/perplexity", scalar_value = cls.all_perplexity[f'{mode}_perplexity'].average, global_step = cls.current_step)
        
        
    def train_one_step_MASKER(cls, data):
        
        cls.current_step += 1
        
        orig_token, MKR_tokens, MER_tokens = data['all_input_ids']
        MKR_labels, classi_labels          = data['all_labels']
        
        orig_token, MKR_tokens, MER_tokens = orig_token.to(cls.device), MKR_tokens.to(cls.device), MER_tokens.to(cls.device)
        MKR_labels, classi_labels          = MKR_labels.to(cls.device), classi_labels.to(cls.device)
        
        cls.optimizer.zero_grad()
        
        out_CE, out_MKR, out_MER  = cls.model((orig_token, MKR_tokens, MER_tokens))
        
        # classification loss
        loss_CE   = cls.criterion_CE(out_CE, classi_labels)
        loss_CE   = loss_CE

        # self-supervision loss
        out_MKR    = out_MKR.permute(0, 2, 1)
        
        loss_MKR   = cls.criterion_MKR(out_MKR, MKR_labels) 
        loss_MKR   = loss_MKR * 0.001

        # outlier regularization loss
        out_MER    = F.log_softmax(out_MER, dim=1)  # log-probs
        unif       = uniform_labels(classi_labels, cls.device, n_classes = 2)
        loss_MER   = F.kl_div(out_MER, unif)
        loss_MER   = loss_MER * 0.001
        
        total_loss       = loss_CE + loss_MKR + loss_MER
        
        total_loss.backward()
        cls.optimizer.step()
        
        cls.all_losses['train_sum_loss'].update(total_loss.item(), orig_token.shape[0])
        cls.all_losses['train_CE_loss'] .update(loss_CE.item(), orig_token.shape[0])
        cls.all_losses['train_MKR_loss'].update(loss_MKR.item(), orig_token.shape[0])
        cls.all_losses['train_MER_loss'].update(loss_MER.item(), orig_token.shape[0])
        
        # if it is the eval step then report Loss and Metrics of THIS STEP
        if (cls.current_step % cls.val_steps == 0 or cls.current_step % cls.logging_steps == 0) and cls.current_step != 0:
            
            # TRAIN loss
            cls.writer.add_scalar(tag = "train/sum_loss", scalar_value = total_loss.item(), global_step = cls.current_step)
            cls.writer.add_scalar(tag = "train/CE_loss",  scalar_value = loss_CE.item(),    global_step = cls.current_step)
            cls.writer.add_scalar(tag = "train/MKR_loss", scalar_value = loss_MKR.item(),   global_step = cls.current_step)
            cls.writer.add_scalar(tag = "train/MER_loss", scalar_value = loss_MER.item(),   global_step = cls.current_step)
            
            print(f"Epoch : {cls.current_epoch} | Step : {cls.current_step}")
            print(f"Train  Total Loss    : {total_loss.item()} | CE : {loss_CE.item()} | MKR : {loss_MKR.item()} | MER :  {loss_MER.item()}")
            
            # TRAIN metrics
            cls.all_metrics['train_metrics'].keep(torch.softmax(out_CE, dim = 1), classi_labels) # bc out logits did not pass sigmoid or softmax yet
            cls.all_metrics['train_metrics'].calculate(cls.current_step, cls.writer, 'train', cls.training_obj)
            
    
    @classmethod
    def evaluate_MASKER(cls, mode):
        cls.all_losses[f'{mode}_CE_loss'].reset()
        
        if mode == 'test':
            data_loader = cls.test_loader
        if mode == 'val':
            data_loader = cls.val_loader
            
        for data in data_loader:    
                
            orig_token, _, _    = data['all_input_ids']
            _, classi_labels    = data['all_labels']

            orig_token    = orig_token.to(cls.device)
            classi_labels = classi_labels.to(cls.device)

            if mode == 'val':
                out_cls = cls.model.inference(orig_token)
            if mode == 'test':
                out_cls = cls.best_model.inference(orig_token)
            
            # classification
            CE_loss   = cls.criterion_CE(out_cls, classi_labels)

            cls.all_losses[f'{mode}_CE_loss'].update(CE_loss.item(), orig_token.shape[0])
            cls.all_metrics[f'{mode}_metrics'].keep(torch.softmax(out_cls, dim = 1), classi_labels)
        
        # report Loss and Metrics of the Whole VAL SET
        print(f"{mode} Loss    : " , cls.all_losses[f'{mode}_CE_loss'].average)
        cls.writer.add_scalar(tag = f"{mode}/CE_loss", scalar_value = cls.all_losses[f'{mode}_CE_loss'].average, global_step = cls.current_step)
        cls.all_metrics[f'{mode}_metrics'].calculate(cls.current_step, cls.writer, mode, cls.training_obj)
        
        
    @classmethod
    def evaluate_MASKERsumloss(cls, mode):
        cls.all_losses[f'{mode}_CE_loss'].reset()
        cls.all_losses[f'{mode}_MER_loss'].reset()
        cls.all_losses[f'{mode}_MKR_loss'].reset()
        cls.all_losses[f'{mode}_total_loss'].reset()
        
        
        if mode == 'test':
            data_loader = cls.test_loader
        if mode == 'val':
            data_loader = cls.val_loader
            
        for data in data_loader:    
                
            orig_token, MKR_tokens, MER_tokens = data['all_input_ids']
            MKR_labels, classi_labels          = data['all_labels']

            orig_token, MKR_tokens, MER_tokens = orig_token.to(cls.device), MKR_tokens.to(cls.device), MER_tokens.to(cls.device)
            MKR_labels, classi_labels          = MKR_labels.to(cls.device), classi_labels.to(cls.device)

            if mode == 'val':
                out_CE, out_MKR, out_MER  = cls.model((orig_token, MKR_tokens, MER_tokens))
            if mode == 'test':
                out_CE, out_MKR, out_MER  = cls.best_model((orig_token, MKR_tokens, MER_tokens))
        
            # classification loss
            loss_CE   = cls.criterion_CE(out_CE, classi_labels)
            loss_CE   = loss_CE

            # self-supervision loss
            out_MKR    = out_MKR.permute(0, 2, 1)

            loss_MKR   = cls.criterion_MKR(out_MKR, MKR_labels) 
            loss_MKR   = loss_MKR * 0.001

            # outlier regularization loss
            out_MER    = F.log_softmax(out_MER, dim=1)  # log-probs
            unif       = uniform_labels(classi_labels, cls.device, n_classes = 2)
            loss_MER   = F.kl_div(out_MER, unif)
            loss_MER   = loss_MER * 0.001

            total_loss       = loss_CE + loss_MKR + loss_MER
            
            cls.all_losses[f'{mode}_CE_loss'].update(loss_CE.item(), orig_token.shape[0])
            cls.all_losses[f'{mode}_MER_loss'].update(loss_MER.item(), orig_token.shape[0])
            cls.all_losses[f'{mode}_MKR_loss'].update(loss_MKR.item(), orig_token.shape[0])
            cls.all_losses[f'{mode}_total_loss'].update(total_loss.item(), orig_token.shape[0])
            
            cls.all_metrics[f'{mode}_metrics'].keep(torch.softmax(out_CE, dim = 1), classi_labels)
        
        # report Loss and Metrics of the Whole VAL SET
        print(f"{mode} total Loss    : " , cls.all_losses[f'{mode}_total_loss'].average)
        cls.writer.add_scalar(tag = f"{mode}/CE_loss", scalar_value = cls.all_losses[f'{mode}_CE_loss'].average, global_step = cls.current_step)
        cls.writer.add_scalar(tag = f"{mode}/MER_loss", scalar_value = cls.all_losses[f'{mode}_MER_loss'].average, global_step = cls.current_step)
        cls.writer.add_scalar(tag = f"{mode}/MKR_loss", scalar_value = cls.all_losses[f'{mode}_MKR_loss'].average, global_step = cls.current_step)
        cls.writer.add_scalar(tag = f"{mode}/total_loss", scalar_value = cls.all_losses[f'{mode}_total_loss'].average, global_step = cls.current_step)
        
        cls.all_metrics[f'{mode}_metrics'].calculate(cls.current_step, cls.writer, mode, cls.training_obj)
        
    
    @classmethod
    def train_one_step_MASKER2heads(cls, data):
        
        cls.current_step += 1
        
        orig_token, MKR_tokens, _  = data['all_input_ids']
        MKR_labels, classi_labels  = data['all_labels']
        
        orig_token, MKR_tokens, _ = orig_token.to(cls.device), MKR_tokens.to(cls.device), _
        MKR_labels, classi_labels  = MKR_labels.to(cls.device), classi_labels.to(cls.device)
        
        cls.optimizer.zero_grad()
        
        out_CE, out_MKR  = cls.model((orig_token, MKR_tokens))
        
        # classification loss
        loss_CE   = cls.criterion_CE(out_CE, classi_labels)
        loss_CE   = loss_CE

        # self-supervision loss
        out_MKR    = out_MKR.permute(0, 2, 1)
        
        loss_MKR   = cls.criterion_MKR(out_MKR, MKR_labels) 
        loss_MKR   = loss_MKR # * 0.001

        # outlier regularization loss
        # out_MER    = F.log_softmax(out_MER, dim=1)  # log-probs
        # unif       = uniform_labels(classi_labels, cls.device, n_classes = 2)
        # loss_MER   = F.kl_div(out_MER, unif)
        # loss_MER   = loss_MER * 0.001
        
        total_loss       = loss_CE + loss_MKR # + loss_MER
        
        total_loss.backward()
        cls.optimizer.step()
        
        cls.all_losses['train_sum_loss'].update(total_loss.item(), orig_token.shape[0])
        cls.all_losses['train_CE_loss'] .update(loss_CE.item(), orig_token.shape[0])
        cls.all_losses['train_MKR_loss'].update(loss_MKR.item(), orig_token.shape[0])
        # cls.all_losses['train_MER_loss'].update(loss_MER.item(), orig_token.shape[0])
        
        # if it is the eval step then report Loss and Metrics of THIS STEP
        if (cls.current_step % cls.val_steps == 0 or cls.current_step % cls.logging_steps == 0) and cls.current_step != 0:
            
            # TRAIN loss
            cls.writer.add_scalar(tag = "train/sum_loss", scalar_value = total_loss.item(), global_step = cls.current_step)
            cls.writer.add_scalar(tag = "train/CE_loss",  scalar_value = loss_CE.item(),    global_step = cls.current_step)
            cls.writer.add_scalar(tag = "train/MKR_loss", scalar_value = loss_MKR.item(),   global_step = cls.current_step)
            # cls.writer.add_scalar(tag = "train/MER_loss", scalar_value = loss_MER.item(),   global_step = cls.current_step)
            
            print(f"Epoch : {cls.current_epoch} | Step : {cls.current_step}")
            print(f"Train  Total Loss    : {total_loss.item()} | CE : {loss_CE.item()} | MKR : {loss_MKR.item()}")
            
            # TRAIN metrics
            cls.all_metrics['train_metrics'].keep(torch.softmax(out_CE, dim = 1), classi_labels) # bc out logits did not pass sigmoid or softmax yet
            cls.all_metrics['train_metrics'].calculate(cls.current_step, cls.writer, 'train', cls.training_obj)
            
    @classmethod
    def evaluate_MASKER2head_sumloss(cls, mode):
        cls.all_losses[f'{mode}_CE_loss'].reset()
        cls.all_losses[f'{mode}_MKR_loss'].reset()
        cls.all_losses[f'{mode}_total_loss'].reset()
        
        
        if mode == 'test':
            data_loader = cls.test_loader
        if mode == 'val':
            data_loader = cls.val_loader
            
        for data in data_loader:    
                
            orig_token, MKR_tokens, _  = data['all_input_ids']
            MKR_labels, classi_labels  = data['all_labels']

            orig_token, MKR_tokens, _ = orig_token.to(cls.device), MKR_tokens.to(cls.device), _
            MKR_labels, classi_labels  = MKR_labels.to(cls.device), classi_labels.to(cls.device)
            
            if mode == 'val':
                out_CE, out_MKR  = cls.model((orig_token, MKR_tokens))
            if mode == 'test':
                out_CE, out_MKR  = cls.best_model((orig_token, MKR_tokens))
            
            # classification
            CE_loss   = cls.criterion_CE(out_CE, classi_labels)
            
            out_MKR    = out_MKR.permute(0, 2, 1)
        
            MKR_loss   = cls.criterion_MKR(out_MKR, MKR_labels) 
            MKR_loss   = MKR_loss # * 0.001
            
            total_loss = CE_loss + MKR_loss
            

            cls.all_losses[f'{mode}_CE_loss'].update(CE_loss.item(), orig_token.shape[0])
            cls.all_losses[f'{mode}_MKR_loss'].update(MKR_loss.item(), orig_token.shape[0])
            cls.all_losses[f'{mode}_total_loss'].update(total_loss.item(), orig_token.shape[0])
            
            cls.all_metrics[f'{mode}_metrics'].keep(torch.softmax(out_CE, dim = 1), classi_labels)
        
        # report Loss and Metrics of the Whole VAL SET
        print(f"{mode}  Total Loss    : {total_loss.item()} | CE : {CE_loss.item()} | MKR : {MKR_loss.item()}")
        
        cls.writer.add_scalar(tag = f"{mode}/CE_loss", scalar_value = cls.all_losses[f'{mode}_CE_loss'].average, global_step = cls.current_step)
        cls.writer.add_scalar(tag = f"{mode}/MKR_loss", scalar_value = cls.all_losses[f'{mode}_MKR_loss'].average, global_step = cls.current_step)
        cls.writer.add_scalar(tag = f"{mode}/total_loss", scalar_value = cls.all_losses[f'{mode}_total_loss'].average, global_step = cls.current_step)
        cls.all_metrics[f'{mode}_metrics'].calculate(cls.current_step, cls.writer, mode, cls.training_obj)
        
def uniform_labels(labels, device, n_classes = 2):
    unif = torch.ones(labels.size(0), n_classes).to(device)
    return unif / n_classes

# if training_obj == 'classiBCE' :   
#     criterion  = nn.BCEWithLogitsLoss(reduction = 'mean')
#     num_labels = 1 # BCE
#     model      = AutoModelForSequenceClassification.from_pretrained(tokenizer_checkpoint, num_labels = num_labels).to(device)
#     model.classifier.dropout = nn.Dropout(p = classifier_p_dropout, inplace = False)

 # if training_obj == 'classiBCE':
 #            self.all_metrics = dict()
 #            self.all_metrics['train_metrics'] = ClassiMetricMeter()
 #            self.all_metrics['val_metrics']   = ClassiMetricMeter()
 #            self.all_metrics['test_metrics']  = ClassiMetricMeter()
 #            self.train_one_step = TrainEval.train_one_step_BCE
 #            self.evaluate       = TrainEval.evaluate_BCE
 #            print("Training Sequence Classification with BCE Loss...")

    # simple binary classification task : BCE Loss
#     @classmethod       
#     def train_one_step_BCE(cls, data):
        
#         cls.current_step += 1
        
#         # Every data instance is an input + label pair
#         input_ids = data['input_ids'].to(cls.device)
#         att_mask  = data['attention_mask'].to(cls.device)
#         labels    = data['labels'].view(-1,1).to(cls.device).float() # for bcewithlogitsloss

#         # Zero your gradients for every batch!
#         cls.optimizer.zero_grad()

#         # Make predictions for this batch
#         outputs = cls.model(input_ids=input_ids, attention_mask=att_mask, labels=labels)

#         # classification
#         logits = outputs.logits # (before SoftMax)
#         loss   = cls.criterion(logits, labels)
        
#         loss.backward()
#         cls.optimizer.step()
        
#         # did not use
#         cls.all_losses['train_loss'].update(loss.item(), input_ids.shape[0])

#         # if it is the eval step then report Loss and Metrics of THIS STEP
#         if (cls.current_step % cls.val_steps == 0 or cls.current_step % cls.logging_steps == 0) and cls.current_step != 0:
            
#             # TRAIN loss
#             cls.writer.add_scalar(tag = "train/loss", scalar_value = loss.item(), global_step = cls.current_step)
#             print(f"Epoch : {cls.current_epoch} | Step : {cls.current_step}")
#             print("Train Loss    : " , loss.item())
            
#             # TRAIN metrics
#             cls.all_metrics['train_metrics'].keep(torch.sigmoid(logits), labels) # bc out logits did not pass sigmoid or softmax yet
#             cls.all_metrics['train_metrics'].calculate(cls.current_step, cls.writer, 'train', cls.training_obj)
             
#     # simple binary classification task : BCE Loss     
#     @classmethod
#     def evaluate_BCE(cls, mode):
#         cls.all_losses[f'{mode}_loss'].reset()
        
#         if mode == 'test':
#             data_loader = cls.test_loader
#         if mode == 'val':
#             data_loader = cls.val_loader
        
#         for data in data_loader:    
                
#             # Every data instance is an input + label pair
#             input_ids = data['input_ids'].to(cls.device)
#             att_mask  = data['attention_mask'].to(cls.device)
            
#             # for bcewithlogitsloss
#             labels    = data['labels'].view(-1,1).to(cls.device).float()

#             if mode == 'val':
#                 outputs = cls.model(input_ids=input_ids, attention_mask=att_mask, labels=labels)
#             if mode == 'test':
#                 outputs = cls.best_model(input_ids=input_ids, attention_mask=att_mask, labels=labels)

#             # classification
#             logits = outputs.logits # (before SoftMax)
#             loss   = cls.criterion(logits, labels)
            
#             cls.all_losses[f'{mode}_loss'].update(loss.item(), input_ids.shape[0])
#             cls.all_metrics[f'{mode}_metrics'].keep(torch.sigmoid(logits), labels) # bc out logits did not pass sigmoid or softmax yet    
        
#         # report Loss and Metrics of the Whole VAL SET
#         print(f"{mode} Loss    : " , cls.all_losses[f'{mode}_loss'].average)
#         cls.writer.add_scalar(tag = f"{mode}/loss", scalar_value = cls.all_losses[f'{mode}_loss'].average, global_step = cls.current_step)
#         cls.all_metrics[f'{mode}_metrics'].calculate(cls.current_step, cls.writer, mode, cls.training_obj)