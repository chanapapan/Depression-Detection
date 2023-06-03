import random
import csv, os, sys
csv.field_size_limit(sys.maxsize)
import pickle
from IPython.display import clear_output
from torch.utils.data import Dataset
from csv import reader
import torch
from transformers import BertTokenizerFast # DataCollatorWithPadding
import numpy as np


def save_result_csv( _header_name, _row_data, _path ):
    filename    = _path
    mode        = 'a' if os.path.exists(filename) else 'w'
    with open(f"{filename}", mode) as myfile:
        fileEmpty   = os.stat(filename).st_size == 0
        writer      = csv.DictWriter(myfile, delimiter=',', lineterminator='\n',fieldnames=_header_name)
        if fileEmpty:
            writer.writeheader()  # file doesn't exist yet, write a header

        row_dic = dict(zip(_header_name, _row_data))
        writer.writerow( row_dic )
        myfile.close()

#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################

# GET DOMAIN DATA CSV FOR MLM TRAINING        
def get_domain_traintest_data_files(class_ratio:int=10, subset:float=0.2, test_ratio:float=0.1, val_ratio:float=0.1):
    '''
    get (subset of) concat.txt from normal RSDD dataset
    save as Train and Test .csv
    csv has 2 field 'text', 'labels'
    ex. get_domain_traintest_data_files(class_ratio = 12, subset = 0.2, test_ratio = 0.1, val_ratio = 0.1)
    
    class_ratio = control / depression ratio
    subset      = ratio of subset from the full dataset
    test_ratio  = ratio of test set
    val_ratio   = ratio of validation set
    '''
    
    random.seed(42)
    domain_train_output_file  = f'./data/domain/domain_train_ratio{class_ratio}.csv'
    domain_val_output_file    = f'./data/domain/domain_val_ratio{class_ratio}.csv'
    domain_test_output_file   = f'./data/domain/domain_test_ratio{class_ratio}.csv'
    
    # treated as unlabeled so can only save the whole thing
    domain_train_corpus_output_file  = f'./data/domain/domain_corpus_traindepcon_ratio{class_ratio}.pkl'
    
    domain_corpus = []
    
    all_train_data   = []
    all_train_labels = []
    all_val_data     = []
    all_val_labels   = []
    all_test_data    = []
    all_test_labels  = []

    header = ['text', 'labels']

    # =============== DEPRESSION ===============
    depression_input_file = f'../../data_depression/training/concat_0529.txt'
    with open(depression_input_file) as f:
        d1_all = f.readlines()
        print("All depression user  : ", len(d1_all))
        random.shuffle(d1_all)
        d1_data = [text[:-1] for text in d1_all][:int(subset * len(d1_all))] # rm \n after each line
        label = int(1)
        
        # print("Val start idx : ", int((1-val_ratio-test_ratio) * len(d1_data)),"Test start idx : ", int((1-test_ratio) * len(d1_data)))
        
        train_d1_data = d1_data[ : int((1-val_ratio-test_ratio) * len(d1_data)) ]
        domain_corpus.extend(train_d1_data)
        all_train_data.extend(train_d1_data)
        all_train_labels.extend([1] * len(train_d1_data))
        
        val_d1_data = d1_data[ int((1-val_ratio-test_ratio) * len(d1_data)) : int((1-test_ratio) * len(d1_data)) ]
        all_val_data.extend(val_d1_data)
        all_val_labels.extend([1] * len(val_d1_data))
        
        test_d1_data = d1_data[int((1-test_ratio) * len(d1_data)) : ]
        all_test_data.extend(test_d1_data)
        all_test_labels.extend([1] * len(test_d1_data))
    
    # =============== CONTROL ===============
    control_input_file =  f'../../data_control/training/concat_0529.txt'
    with open(control_input_file) as f:
        d2_all = f.readlines()
        print("All control user  : ", len(d2_all))
        random.shuffle(d2_all)
        d2_data = [text[:-1] for text in d2_all] # we select the number of control according to class_ratio # [:int(subset * len(d2_all))] # rm \n after each line
        num_control = class_ratio * len(d1_data)
        d2_data = d2_data[:num_control]
        label = int(0)
        
        # print("Val start idx : ", int((1-val_ratio-test_ratio) * len(d2_data)),"Test start idx : ",  int((1-test_ratio) * len(d2_data)))
        
        train_d2_data = d2_data[ : int((1-val_ratio-test_ratio) * len(d2_data))]
        domain_corpus.extend(train_d2_data)
        all_train_data.extend(train_d2_data)
        all_train_labels.extend([0] * len(train_d2_data))
        
        val_d2_data = d2_data[ int((1-val_ratio-test_ratio) * len(d2_data)) : int((1-test_ratio) * len(d2_data)) ]
        all_val_data.extend(val_d2_data)
        all_val_labels.extend([0] * len(val_d2_data))

        test_d2_data = d2_data[int((1-test_ratio) * len(d2_data)) : ]
        all_test_data.extend(test_d2_data)
        all_test_labels.extend([0] * len(test_d2_data))
    
    # remove previous files if exist
    if os.path.exists(domain_train_output_file):
        os.remove(domain_train_output_file)
    if os.path.exists(domain_val_output_file):
        os.remove(domain_val_output_file)
    if os.path.exists(domain_test_output_file):
        os.remove(domain_test_output_file)
    
    # ======== CREAT TRAIN AND TEST HUGGING FACE CSV DATA FILE =========
    assert 1 in all_train_labels and 0 in all_train_labels
    print("Depression : ", all_train_labels.count(1), "Control : ",all_train_labels.count(0))
    for idx, text in enumerate(all_train_data):
        label = all_train_labels[idx]
        row = [text, label]
        save_result_csv( header, row, domain_train_output_file)
    
    assert 1 in all_val_labels and 0 in all_val_labels
    print("Depression : ", all_val_labels.count(1), "Control : ", all_val_labels.count(0))
    for idx, text in enumerate(all_val_data):
        label = all_val_labels[idx]
        row = [text, label]
        save_result_csv( header, row, domain_val_output_file)
    
    assert 1 in all_test_labels and 0 in all_test_labels
    print("Depression : ", all_test_labels.count(1), "Control : ", all_test_labels.count(0))
    for idx, text in enumerate(all_test_data):
        label = all_test_labels[idx]
        row = [text, label]
        save_result_csv( header, row, domain_test_output_file)
    
    pickle.dump(domain_corpus, open(domain_train_corpus_output_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print("TRAIN_d1_data : ", len(train_d1_data))
    print("TRAIN_d2_data : ", len(train_d2_data))
    print("VAL_d1_data   : ", len(val_d1_data))
    print("VAL_d2_data   : ", len(val_d2_data))
    print("TEST_d1_data  : ", len(test_d1_data))
    print("TEST_d2_data  : ", len(test_d2_data))
        
    print("Saved file to : ", domain_train_output_file)
    print("Saved file to : ", domain_val_output_file)
    print("Saved file to : ", domain_test_output_file)    
    
    print("Saved corpus file to : ", domain_train_corpus_output_file)


# GET CLASSIFICATION DATA CSV FOR CLASSIFICATION
def get_classi_traintest_data_files(class_ratio:int, subset:float=1.0, test_ratio:float=0.1, val_ratio:float=0.1):
    '''
    get (subset of) concat.txt from Time-RSDD dataset
    save as Train and Test .csv
    csv has 2 field 'text', 'labels'
    ex. get_classi_traintest_data_files(class_ratio = class_ratio, test_ratio = 0.1, val_ratio = 0.1)
    
    class_ratio = control / depression ratio
    subset      = ratio of subset from the full dataset
    test_ratio  = ratio of test set
    val_ratio   = ratio of validation set
    '''

    random.seed(42)
    train_output_file  = f'./data/classi/classi_train_ratio{class_ratio}.csv'
    val_output_file    = f'./data/classi/classi_val_ratio{class_ratio}.csv'
    test_output_file   = f'./data/classi/classi_test_ratio{class_ratio}.csv'
    
    # seperate depression and control for getting depression tokens for masking
    # save list of text as .pkl
    traindepcorpus_output_file = f'./data/classi/classi_corpus_traindep_ratio{class_ratio}.pkl'
    trainconcorpus_output_file = f'./data/classi/classi_corpus_traincon_ratio{class_ratio}.pkl'
        
    all_train_data = []
    all_train_labels = []
    all_val_data = []
    all_val_labels = []
    all_test_data = []
    all_test_labels = []
    
    train_dep = []
    train_con = []

    header = ['text', 'labels']

    # =============== DEPRESSION ===============
    depression_input_file = f'../../data_time/depression/concat_3m_0529.txt'
    with open(depression_input_file) as f:
        d1_all = f.readlines()
        random.shuffle(d1_all)
        d1_data = [text[:-1] for text in d1_all][:int(subset * len(d1_all))] # rm \n after each line
        d1_label = int(1)
        
        # print("Val start idx : ", int((1-val_ratio-test_ratio) * len(d1_data)), "Test start idx : ",int((1-test_ratio) * len(d1_data)))
        
        train_d1_data = d1_data[ : int((1-val_ratio-test_ratio) * len(d1_data))]
        train_dep.extend(train_d1_data)
        all_train_data.extend(train_d1_data)
        all_train_labels.extend([d1_label] * len(train_d1_data))
        
        val_d1_data = d1_data[ int((1-val_ratio-test_ratio) * len(d1_data)) : int((1-test_ratio) * len(d1_data)) ]
        all_val_data.extend(val_d1_data)
        all_val_labels.extend([d1_label] * len(val_d1_data))
        
        test_d1_data = d1_data[ int((1-test_ratio) * len(d1_data)) : ]
        all_test_data.extend(test_d1_data)
        all_test_labels.extend([d1_label] * len(test_d1_data))
            
    # =============== CONTROL ===============
    control_input_file = f'../../data_time/control/concat_3m_0529.txt'
    with open(control_input_file) as f:
        d2_all = f.readlines()
        random.shuffle(d2_all)
        
        # get all d2 data
        d2_data = [text[:-1] for text in d2_all][:int(subset * len(d2_all))] # rm \n after each line
        d2_label = int(0)
        
        # split all d2 data in to 3 parts
        d2_data_all_train = d2_data[ : int((1-val_ratio-test_ratio) * len(d2_data))]
        d2_data_all_val   = d2_data[ int((1-val_ratio-test_ratio) * len(d2_data)) : int((1-test_ratio) * len(d2_data))]
        d2_data_all_test  = d2_data[ int((1-test_ratio) * len(d2_data)) :]
              
        train_d2_data = d2_data_all_train[ : class_ratio * len(train_d1_data)]
        train_con.extend(train_d2_data)
        all_train_data.extend(train_d2_data)
        all_train_labels.extend([d2_label] * len(train_d2_data))
        
        val_d2_data = d2_data_all_val[ : class_ratio * len(val_d1_data)]
        all_val_data.extend(val_d2_data)
        all_val_labels.extend([d2_label] * len(val_d2_data))
        
        test_d2_data = d2_data_all_test[ : class_ratio * len(test_d1_data)]
        all_test_data.extend(test_d2_data)
        all_test_labels.extend([d2_label] * len(test_d2_data))
    
    # remove previous files if exist
    if os.path.exists(train_output_file):
        os.remove(train_output_file)
    if os.path.exists(val_output_file):
        os.remove(val_output_file)
    if os.path.exists(test_output_file):
        os.remove(test_output_file)
        
    if os.path.exists(traindepcorpus_output_file):
        os.remove(traindepcorpus_output_file)
    if os.path.exists(trainconcorpus_output_file):
        os.remove(trainconcorpus_output_file)
   
    # ======== CREAT TRAIN AND TEST HUGGING FACE CSV DATA FILE =========
    assert 1 in all_train_labels and 0 in all_train_labels
    print("Depression : ", all_train_labels.count(1), "Control : ", all_train_labels.count(0))
    for idx, text in enumerate(all_train_data):
        label = all_train_labels[idx]
        row = [text, label]
        save_result_csv( header, row, train_output_file)
    
    assert 1 in all_val_labels and 0 in all_val_labels
    print("Depression : ", all_val_labels.count(1), "Control : ", all_val_labels.count(0))
    for idx, text in enumerate(all_val_data):
        label = all_val_labels[idx]
        row = [text, label]
        save_result_csv( header, row, val_output_file)
    
    assert 1 in all_test_labels and 0 in all_test_labels
    print("Depression : ", all_test_labels.count(1), "Control : ", all_test_labels.count(0))
    for idx, text in enumerate(all_test_data):
        label = all_test_labels[idx]
        row = [text, label]
        save_result_csv( header, row, test_output_file)
        
    print("TRAIN_d1_data : ", len(train_d1_data))
    print("TRAIN_d2_data : ", len(train_d2_data))
    print("VAL_d1_data   : ", len(val_d1_data))
    print("VAL_d2_data   : ", len(val_d2_data))
    print("TEST_d1_data  : ", len(test_d1_data))
    print("TEST_d2_data  : ", len(test_d2_data))
    
    # =============================
    pickle.dump(train_dep, open(traindepcorpus_output_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_con, open(trainconcorpus_output_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Saved file to : ", train_output_file)
    print("Saved file to : ", val_output_file)
    print("Saved file to : ", test_output_file)
    
    print("Saved corpus file to : ", traindepcorpus_output_file)
    print("Saved corpus file to : ", trainconcorpus_output_file)


#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################

def get_chunk_ds(class_ratio, csv_path, checkpoint, domain = False):
    
    tokenizer     = BertTokenizerFast.from_pretrained(checkpoint)
    
    train_dataset = ChunkDataset(csv_path['train'], tokenizer)
    if domain :
        train_ds_path = f'./data/domain/domainchunk-R{class_ratio}-train-ds.pkl'
        with open(train_ds_path, 'wb') as outp:
            pickle.dump(train_dataset, outp, pickle.HIGHEST_PROTOCOL)
        print(f"Saved train_dataset to {train_ds_path}")
    else:
        train_ds_path = f'./data/classi/classichunk-R{class_ratio}-train-ds.pkl'
        with open(train_ds_path, 'wb') as outp:
            pickle.dump(train_dataset, outp, pickle.HIGHEST_PROTOCOL)
        print(f"Saved train_dataset to {train_ds_path}")
    
    val_dataset   = ChunkDataset(csv_path['val'],   tokenizer)
    if domain :
        val_ds_path = f'./data/domain/domainchunk-R{class_ratio}-val-ds.pkl'
        with open(val_ds_path, 'wb') as outp:
            pickle.dump(val_dataset, outp, pickle.HIGHEST_PROTOCOL)
        print(f"Saved val_dataset to {val_ds_path}")   
    else:
        val_ds_path = f'./data/classi/classichunk-R{class_ratio}-val-ds.pkl'
        with open(val_ds_path, 'wb') as outp:
            pickle.dump(val_dataset, outp, pickle.HIGHEST_PROTOCOL)
        print(f"Saved val_dataset to {val_ds_path}")   
    
    test_dataset  = ChunkDataset(csv_path['test'],  tokenizer)
    if domain :
        test_ds_path = f'./data/domain/domainchunk-R{class_ratio}-test-ds.pkl'
        with open(test_ds_path, 'wb') as outp:
            pickle.dump(test_dataset, outp, pickle.HIGHEST_PROTOCOL)
        print(f"Saved test_dataset to {test_ds_path}")
    else :
        test_ds_path = f'./data/classi/classichunk-R{class_ratio}-test-ds.pkl'
        with open(test_ds_path, 'wb') as outp:
            pickle.dump(test_dataset, outp, pickle.HIGHEST_PROTOCOL)
        print(f"Saved test_dataset to {test_ds_path}")
    
    print("TRAIN set : ", len(train_dataset))
    print("VAL set   : ", len(val_dataset))
    print("TEST set  : ", len(test_dataset))

    return {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    
class ChunkDataset(Dataset):
    
    def __init__(self, file_path, tokenizer):
        
        self.tokenizer        = tokenizer
        self.SPECIAL_TOKENS   = tokenizer.all_special_ids
        self.model_cls_id     = self.tokenizer(self.tokenizer.cls_token)['input_ids'][1]
        self.model_sep_id     = self.tokenizer(self.tokenizer.sep_token)['input_ids'][1]
        self.model_max_length = (self.tokenizer.model_max_length)
        
        print(f"Preparing ChunkDataset from {file_path}")
        
        self.all_input_ids     = []
        self.all_word_ids_list  = []
        self.all_orig_text     = []
        self.all_depcon_label  = []
        
        # skip first line (header)
        with open(file_path, 'r') as read_obj:
            csv_reader = reader(read_obj)
            header = next(csv_reader)
            if header != None:
                # Iterate over each row after the header in the csv
                for i, row in enumerate(csv_reader):
                    sys.stdout.write(str(i))
                    # row variable is a list that represents a row in csv
                    text = row[0]
                    label = int(row[1])
                    self.batchify(text, label)
        
        print("len ChunkDataset = ", len(self.all_depcon_label))
    
    def batchify(self, text, label):
        
        tokenized_text = self.tokenizer(text, padding = False, truncation = False)
        word_ids       = tokenized_text.word_ids()
        # print(len(word_ids))
        
        self.chunk_max_length = self.model_max_length - 2 # for ['CLS'] and ['SEP']
        self.limit_mask       = round(0.15 * self.chunk_max_length)
        
        # Remove the old [CLS] and [SEP]
        input_ids      = tokenized_text['input_ids'][1:-1]
        word_ids       = word_ids[1:-1]
        # print(len(word_ids))
        
        # Divide the dataset into chunks
        nbatch    = len(input_ids) // self.chunk_max_length
        
        # Trim off any extra elements that wouldn't cleanly fit.
        input_ids = input_ids[:nbatch * self.chunk_max_length]
        word_ids  = word_ids[:nbatch * self.chunk_max_length]
        # print(len(word_ids))
        
        # Evenly divide the data across the bsz batches.
        chunk_input_ids = torch.tensor(input_ids).view(nbatch, self.chunk_max_length)
        chunk_word_ids  = torch.tensor(word_ids).view(nbatch, self.chunk_max_length)
        # print(chunk_word_ids.shape)

        # Add new [CLS] and [SEP] token to all chunks
        chunk_input_ids = torch.cat([self.model_cls_id * torch.ones(chunk_input_ids.shape[0], 1), chunk_input_ids, self.model_sep_id * torch.ones(chunk_input_ids.shape[0], 1)], dim=1).contiguous().long()
        chunk_word_ids = torch.cat([-99999 * torch.ones(chunk_word_ids.shape[0], 1), chunk_word_ids, -99999 * torch.ones(chunk_word_ids.shape[0], 1)], dim=1).contiguous().long()
        
        # for each chunk do
        for idx in range(chunk_input_ids.shape[0]):
            
            # do not include samples with > 15% repeated tokens = SPAM !
            output, counts = torch.unique(chunk_input_ids[idx], return_counts = True)
            max_count = int(max(counts))
            del counts, output
            if max_count > self.limit_mask:
                # print(max_count)
                continue
                
            else:
                self.all_input_ids.append(chunk_input_ids[idx].clone())
                self.all_word_ids_list.append(chunk_word_ids[idx].clone())
                self.all_orig_text.append(self.tokenizer.decode(chunk_input_ids[idx].clone()))
                self.all_depcon_label.append(label)           

    def __len__(self):
        return len(self.all_depcon_label)

    def __getitem__(self, idx):
        sample = {  'input_ids'      : self.all_input_ids[idx],
                    'word_ids'       : self.all_word_ids_list[idx],
                    'attention_mask' : torch.ones_like(self.all_input_ids[idx]),
                    'orig_text'      : self.all_orig_text[idx],
                    'labels'         : self.all_depcon_label[idx]}
        return sample

#############################################################################################################
#############################################################################################################
    
def get_MLM_ds(chunkdatasets, checkpoint, masking_method = None, keyword_path = None, keyword_name = None):
    
    tokenizer = BertTokenizerFast.from_pretrained(checkpoint)
        
    keyword_list = None
    if masking_method == 'keywords':
        assert keyword_path is not None
        
        print(f"Getting {masking_method} from {keyword_path}")
        with open(keyword_path) as f:
            lexicon = f.readlines()
        lexicon = [ lex[:-1] for lex in lexicon]
        print(len(lexicon))
        print(lexicon)
        
        # tokenize the lexicon and keep the input_ids of each word together in a list 
        keyword_input_ids = [ torch.tensor((tokenizer(lex).input_ids)[1:-1]) for lex in lexicon ]
        print(len(keyword_input_ids))
        print(keyword_input_ids)
        keyword_list = keyword_input_ids
        assert keyword_list is not None
    
    # ----- CREATE DATASET -----
    train_dataset = MLMDataset(chunkdatasets['train'],
                              tokenizer,
                              masking_method = masking_method,
                              keyword_list   = keyword_list)
    
    train_ds_path = f'./data/MLM/MLM-{keyword_name}-train-ds.pkl'
    with open(train_ds_path, 'wb') as outp:
        pickle.dump(train_dataset, outp, pickle.HIGHEST_PROTOCOL)
    print(f"Saved train_dataset to {train_ds_path}")
        
    val_dataset = MLMDataset(chunkdatasets['val'], 
                              tokenizer,
                              masking_method = masking_method,
                              keyword_list   = keyword_list)

    val_ds_path = f'./data/MLM/MLM-{keyword_name}-val-ds.pkl'
    with open(val_ds_path, 'wb') as outp:
        pickle.dump(val_dataset, outp, pickle.HIGHEST_PROTOCOL)
    print(f"Saved val_dataset to {val_ds_path}")   
    
    test_dataset  = MLMDataset(chunkdatasets['test'], 
                              tokenizer,
                              masking_method = masking_method,
                              keyword_list   = keyword_list)

    test_ds_path = f'./data/MLM/MLM-{keyword_name}-test-ds.pkl'
    with open(test_ds_path, 'wb') as outp:
        pickle.dump(test_dataset, outp, pickle.HIGHEST_PROTOCOL)
    print(f"Saved test_dataset to {test_ds_path}")
    
    print("TRAIN set : ", len(train_dataset))
    print("VAL set   : ", len(val_dataset))
    print("TEST set  : ", len(test_dataset))

    return {'train': train_dataset , 'val': val_dataset, 'test': test_dataset}


class MLMDataset(Dataset):
    
    def __init__(self, chunkdataset, tokenizer, masking_method, keyword_list):
        
        self.tokenizer        = tokenizer
        self.SPECIAL_TOKENS   = tokenizer.all_special_ids
        self.model_mask_id    = self.tokenizer(self.tokenizer.mask_token)['input_ids'][1]
        self.model_cls_id     = self.tokenizer(self.tokenizer.cls_token)['input_ids'][1]
        self.model_sep_id     = self.tokenizer(self.tokenizer.sep_token)['input_ids'][1]
        self.model_unk_id     = self.tokenizer(self.tokenizer.unk_token)['input_ids'][1]
        self.model_max_length = (self.tokenizer.model_max_length)
        self.vocab            = [ i for i in range(tokenizer.vocab_size) if (i not in self.SPECIAL_TOKENS)]
        
        self.chunkdataset     = chunkdataset
        self.masking_method   = masking_method
        self.keyword_list     = keyword_list
        self.chunk_max_length = self.model_max_length - 2
        self.limit_mask       = round(0.15 * self.chunk_max_length)
        
        self.create_dataset()

    def __len__(self):
        return len(self.chunkdataset)

    def create_dataset(self):
        
        self.list_input_ids      = []
        self.list_word_ids       = []
        self.list_attention_mask = []
        self.list_orig_text      = []
        self.list_masked_text    = []
        self.list_labels         = []
        
        for idx in range(len(self.chunkdataset)):
            
            sys.stdout.write(str(idx))
        
            chunk_sample = self.chunkdataset.__getitem__(idx)

            input_ids    = chunk_sample['input_ids'].clone()
            word_ids     = chunk_sample['word_ids'].clone()
            orig_text    = chunk_sample['orig_text']
            labels       = chunk_sample['labels']

            orig_input_ids = chunk_sample['input_ids'].clone()

            # ========== Do masking for MLM =============        

            if self.masking_method == 'random':
                output, counts = torch.unique(word_ids, return_counts = True)
                output, counts = output[1:], counts[1:]
                # print(output)
                # print(counts)

                idx_all = list(range(output.shape[0]))
                random.shuffle(idx_all)
                # print(idx_all)

                output = torch.index_select(output, dim = 0 , index = torch.tensor(idx_all))
                counts = torch.index_select(counts, dim = 0 , index = torch.tensor(idx_all))
                # print(output)
                # print(counts)


                # get threshold where cumsum of counts is > 76
                cumsum = 0
                for count_idx, c in enumerate(counts):
                    cumsum += c
                    if cumsum >= self.limit_mask:
                        till_here = count_idx + 1
                        break
                # print(till_here, cumsum)

                word_ids_to_mask           = output[:till_here]
                counts_of_word_ids_to_mask = counts[:till_here]

                # print(word_ids_to_mask)
                # print(counts_of_word_ids_to_mask)

            if self.masking_method == 'keywords':

                word_ids_to_mask = []
                num_mask_now     = 0
                word_length_list = []

                # print(input_ids)
                # print(word_ids)

                # get word_ids of the keywords in the sequence
                output, counts = torch.unique(word_ids, return_counts = True)
                output, counts = output[1:], counts[1:]
                # print(output, counts)

                for output_idx, word_id in enumerate(output):
                    start_idx_word = word_ids.tolist().index(int(word_id))
                    word_length    = counts[output_idx]
                    # print(start_idx_word, word_length)

                    word_token_ids = input_ids[start_idx_word : start_idx_word + word_length]
                    # print(word_token_ids)

                    for kw in self.keyword_list:
                        if torch.equal(word_token_ids , kw) :
                            # print('yes')
                            word_ids_to_mask.append(word_id)
                            word_length_list.append(word_length)
                            num_mask_now += word_length

                # print("num tokens masked by KW : ", num_mask_now)

                # print(word_ids_to_mask)
                # print(word_length_list)
                # print(num_mask_now)


                # if masked less than 15 % mask more Whole Word Until reach 15 %
                if num_mask_now < self.limit_mask :

                    num_mask_more = self.limit_mask - num_mask_now # how many more token to mask randomly

                    non_kw_output = torch.tensor([int(word_id) for word_id in output if word_id not in word_ids_to_mask])
                    non_kw_counts = torch.tensor([int(count) for ind,count in enumerate(counts) if output[ind] not in word_ids_to_mask])
                    # print(non_kw_output)
                    # print(non_kw_counts)

                    #---
                    idx_all = list(range(non_kw_output.shape[0]))
                    random.shuffle(idx_all)
                    # print(idx_all)

                    non_kw_output = torch.index_select(non_kw_output, dim = 0 , index = torch.tensor(idx_all))
                    non_kw_counts = torch.index_select(non_kw_counts, dim = 0 , index = torch.tensor(idx_all))
                    # print(non_kw_output)
                    # print(non_kw_counts)

                    # get threshold where cumsum of counts is > num_mask_more
                    cumsum = 0
                    for count_idx, c in enumerate(non_kw_counts):
                        cumsum += c
                        if cumsum >= num_mask_more:
                            till_here = count_idx + 1
                            break
                    # print(till_here, cumsum)

                    word_ids_to_mask_more           = non_kw_output[:till_here]
                    counts_of_word_ids_to_mask_more = non_kw_counts[:till_here]
                    # print(word_ids_to_mask_more)
                    # print(counts_of_word_ids_to_mask_more)

                    word_ids_to_mask = torch.cat([torch.tensor(word_ids_to_mask), word_ids_to_mask_more], dim=0)
                    num_mask         = cumsum + num_mask_now
                    # print(word_ids_to_mask)
                    # print(num_mask)


                # if masked too many, remove the excess
                elif num_mask_now > self.limit_mask :

                    idx_all = list(range(len(word_ids_to_mask)))
                    random.shuffle(idx_all)
                    # print(idx_all)

                    word_ids_to_mask  = torch.index_select(torch.tensor(word_ids_to_mask), dim = 0 , index = torch.tensor(idx_all))
                    word_length_list = torch.index_select(torch.tensor(word_length_list), dim = 0 , index = torch.tensor(idx_all))
    #                         print(word_ids_to_mask)
    #                         print(word_length_list)

                    # get threshold where cumsum of counts is > 76
                    cumsum = 0
                    for count_idx, c in enumerate(word_length_list):
                        cumsum += c
                        if cumsum >= self.limit_mask : # 
                            till_here = count_idx + 1
                            break
                    # print(till_here, cumsum)

                    word_ids_to_mask   = word_ids_to_mask[:till_here]
                    word_length_list   = word_length_list[:till_here]
                    # print(word_ids_to_mask)
                    # print(word_length_list)

            if isinstance(word_ids_to_mask, list):
                pass
            else :
                word_ids_to_mask = word_ids_to_mask.tolist()
            random.shuffle(word_ids_to_mask)
            word_ids_to_mask = torch.tensor(word_ids_to_mask)
            
            
            word_ids_maskmask     = word_ids_to_mask[: int(len(word_ids_to_mask)*0.8) ] # 80% len = 60
            word_ids_maskrandom   = word_ids_to_mask[ int(len(word_ids_to_mask)*0.8) : int(len(word_ids_to_mask)*0.9) ] # 10%  len = 8
            word_ids_maskoriginal = word_ids_to_mask[ int(len(word_ids_to_mask)*0.9) : ] # 10% len = 8 
            # print(word_ids_maskmask)
            # print(word_ids_maskrandom)
            # print(word_ids_maskoriginal)

            # get the index of these word_ids to mask       
            idx_mask_all     = torch.tensor([mask_idx for word_id in word_ids_to_mask for mask_idx in range(word_ids.shape[0]) if word_id == word_ids[mask_idx]]).long()
            idx_maskmask     = torch.tensor([mask_idx for word_id in word_ids_maskmask for mask_idx in range(word_ids.shape[0]) if word_id == word_ids[mask_idx]]).long()
            idx_maskrandom   = torch.tensor([mask_idx for word_id in word_ids_maskrandom for mask_idx in range(word_ids.shape[0]) if word_id == word_ids[mask_idx]]).long()
            idx_maskoriginal = torch.tensor([mask_idx for word_id in word_ids_maskoriginal for mask_idx in range(word_ids.shape[0]) if word_id == word_ids[mask_idx]]).long()

            # print(len(idx_mask_all))
            # print(idx_mask_all)
            # print(idx_maskmask)
            # print(idx_maskrandom)
            # print(idx_maskoriginal)

            # [MASK] 80%
            input_ids.index_fill_(dim=0, index = idx_maskmask, value = torch.tensor(int(self.model_mask_id)))
            # random tokens 10%
            input_ids.index_put_(indices = (idx_maskrandom, ) , values = torch.tensor(random.sample(self.vocab, len(idx_maskrandom))).long())
            # original 10% = do nothing

            label = (torch.ones_like(input_ids) * -100)
            label.index_put_(indices = (idx_mask_all, ) , values = torch.index_select(orig_input_ids, dim = 0 , index = torch.tensor(idx_mask_all)))

            assert torch.sum(torch.isin(input_ids, torch.tensor([int(self.model_mask_id)]))) == len(idx_maskmask)
            assert torch.sum(torch.isin(label, orig_input_ids)) == len(idx_mask_all)

            self.list_input_ids.append(input_ids.clone())
            self.list_word_ids.append(word_ids.clone())
            self.list_attention_mask.append(torch.ones_like(input_ids))
            self.list_orig_text.append(orig_text)
            self.list_masked_text.append(self.tokenizer.decode(input_ids.clone()))
            self.list_labels.append(label.clone())
            
        assert len(self.list_input_ids) == len(self.list_word_ids) == len(self.list_attention_mask) == len(self.list_orig_text) == len(self.list_masked_text) == len(self.list_labels)
    
    def __getitem__(self, idx):
        sample = {  'input_ids'      : self.list_input_ids[idx],
                    'word_ids'       : self.list_word_ids[idx],
                    'attention_mask' : self.list_attention_mask[idx],
                    'orig_text'      : self.list_orig_text[idx],
                    'masked_text'    : self.list_masked_text[idx],
                    'labels'         : self.list_labels[idx]}
        
        return sample

#############################################################################################################
#############################################################################################################
    
def get_MASKER_ds(class_ratio, chunkdatasets, checkpoint, masking_method = None, keyword_path = None, keyword_name = None):
    
    tokenizer = BertTokenizerFast.from_pretrained(checkpoint)
        
    keyword_list = None
    if masking_method == 'keywords':
        assert keyword_path is not None
        
        print(f"Getting {masking_method} from {keyword_path}")
        with open(keyword_path) as f:
            lexicon = f.readlines()
        lexicon = [ lex[:-1] for lex in lexicon]
        print(len(lexicon))
        print(lexicon)
        
        # tokenize the lexicon and keep the input_ids of each word together in a list 
        keyword_input_ids = [ torch.tensor((tokenizer(lex).input_ids)[1:-1]) for lex in lexicon ]
        print(len(keyword_input_ids))
        print(keyword_input_ids)
        keyword_list = keyword_input_ids
        assert keyword_list is not None
    
    # ----- CREATE DATASET -----
    train_dataset = MASKERDataset(chunkdatasets['train'],
                              tokenizer,
                              masking_method = masking_method,
                              keyword_list   = keyword_list)
    
    train_ds_path = f'./data/MASKER/MASKER-{keyword_name}-R{class_ratio}-train-ds.pkl'
    with open(train_ds_path, 'wb') as outp:
        pickle.dump(train_dataset, outp, pickle.HIGHEST_PROTOCOL)
    print(f"Saved train_dataset to {train_ds_path}")
        
    val_dataset = MASKERDataset(chunkdatasets['val'], 
                              tokenizer,
                              masking_method = masking_method,
                              keyword_list   = keyword_list)
    
    val_ds_path = f'./data/MASKER/MASKER-{keyword_name}-R{class_ratio}-val-ds.pkl'
    with open(val_ds_path, 'wb') as outp:
        pickle.dump(val_dataset, outp, pickle.HIGHEST_PROTOCOL)
    print(f"Saved val_dataset to {val_ds_path}")   
    
    test_dataset  = MASKERDataset(chunkdatasets['test'], 
                              tokenizer,
                              masking_method = masking_method,
                              keyword_list   = keyword_list)
    
    test_ds_path = f'./data/MASKER/MASKER-{keyword_name}-R{class_ratio}-test-ds.pkl'
    with open(test_ds_path, 'wb') as outp:
        pickle.dump(test_dataset, outp, pickle.HIGHEST_PROTOCOL)
    print(f"Saved test_dataset to {test_ds_path}")
    
    print("TRAIN set : ", len(train_dataset))
    print("VAL set   : ", len(val_dataset))
    print("TEST set  : ", len(test_dataset))

    return {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}    

class MASKERDataset(Dataset):
    
    def __init__(self, chunkdataset, tokenizer, masking_method, keyword_list):
        
        self.tokenizer        = tokenizer
        self.SPECIAL_TOKENS   = tokenizer.all_special_ids
        self.model_mask_id    = self.tokenizer(self.tokenizer.mask_token)['input_ids'][1]
        self.model_cls_id     = self.tokenizer(self.tokenizer.cls_token)['input_ids'][1]
        self.model_sep_id     = self.tokenizer(self.tokenizer.sep_token)['input_ids'][1]
        self.model_unk_id     = self.tokenizer(self.tokenizer.unk_token)['input_ids'][1]
        self.model_max_length = (self.tokenizer.model_max_length)
        self.vocab            = [ i for i in range(tokenizer.vocab_size) if (i not in self.SPECIAL_TOKENS)]
        
        self.chunkdataset   = chunkdataset
        self.masking_method = masking_method
        self.keyword_list   = keyword_list
        self.chunk_max_length = self.model_max_length - 2
        self.limit_mask     = round(0.15 * self.chunk_max_length)
        
        self.create_dataset()

    def __len__(self):
        return len(self.chunkdataset)

    def create_dataset(self):
        
        self.list_all_input_ids = []
        self.list_word_ids      = []
        self.list_orig_text     = []
        self.list_all_labels    = []
        
        for idx in range(len(self.chunkdataset)):
            
            sys.stdout.write(str(idx))
        
            chunk_sample = self.chunkdataset.__getitem__(idx)

            input_ids    = chunk_sample['input_ids'].clone()
            word_ids     = chunk_sample['word_ids'].clone()
            orig_text    = chunk_sample['orig_text']
            labels       = chunk_sample['labels']

            orig_input_ids = chunk_sample['input_ids'].clone()
            orig_token     = orig_input_ids.clone()
            MKR_tokens     = orig_input_ids.clone()  # masked token (Masked Keywords Reconstruction)
            MER_tokens     = orig_input_ids.clone()  # outlier token (Masked Entropy Regularization)

            # ========== Do masking for MLM =============        

            if self.masking_method == 'keywords':

                word_ids_to_mask = []
                num_mask_now     = 0
                word_length_list = []

                # print(chunk_input_ids[idx])
                # print(chunk_word_ids[idx])

                # get word_ids of the keywords in the sequence
                output, counts = torch.unique(word_ids, return_counts = True)
                output, counts = output[1:], counts[1:]
                # print(output, counts)

                for output_idx, word_id in enumerate(output):
                    start_idx_word = word_ids.tolist().index(int(word_id))
                    word_length    = counts[output_idx]
                    # print(start_idx_word, word_length)

                    word_token_ids = input_ids[start_idx_word : start_idx_word + word_length]
                    # print(word_token_ids)

                    for kw in self.keyword_list:
                        if torch.equal(word_token_ids , kw) :
                            word_ids_to_mask.append(word_id)
                            word_length_list.append(word_length)
                            num_mask_now += word_length    

                # print(num_mask_now)

                p = 0.9
                q = 0.9
                # mask the keywords only where p_mask < 0.5
                p_mask = torch.rand(torch.tensor(word_ids_to_mask).shape)
                word_ids_to_mask = torch.tensor([idx_k for i, idx_k in enumerate(word_ids_to_mask) if p_mask[i] < p]).long()

                word_ids_out_mask  = torch.tensor([int(out) for out in output if out not in word_ids_to_mask])
                # mask the context only where q_mask < 0.9
                q_mask = torch.rand(word_ids_out_mask.shape)
                word_ids_out_mask = torch.tensor([idx_k for i, idx_k in enumerate(word_ids_out_mask) if q_mask[i] < q])

                # print(word_ids_to_mask)
                # print(word_ids_out_mask)                    

            if self.masking_method == 'random':
                # print(word_ids)

                output, counts = torch.unique(word_ids, return_counts = True)
                output, counts = output[1:], counts[1:]

                idx_all = list(range(output.shape[0]))
                random.shuffle(idx_all)

                output = torch.index_select(output, dim = 0 , index = torch.tensor(idx_all))
                counts = torch.index_select(counts, dim = 0 , index = torch.tensor(idx_all))

                # get threshold where cumsum of counts is > 76
                cumsum = 0
                for count_idx, c in enumerate(counts):
                    cumsum += c
                    if cumsum >= self.limit_mask:
                        till_here = count_idx + 1
                        break

                word_ids_to_mask   = output[:till_here]
                word_ids_out_mask  = [int(out) for out in output if out not in word_ids_to_mask]
                # print(word_ids_to_mask)
                # print(word_ids_out_mask)

            idx_key_mask     = torch.tensor([mask_idx for word_id in word_ids_to_mask for mask_idx in range(word_ids.shape[0]) if word_id == word_ids[mask_idx]], dtype=torch.int64)
            idx_out_mask     = torch.tensor([mask_idx for word_id in word_ids_out_mask for mask_idx in range(word_ids.shape[0]) if word_id == word_ids[mask_idx]], dtype=torch.int64)

            # print(idx_key_mask)
            # print(idx_out_mask)

            #  --------------- MKR ---------------
            MKR_tokens.index_fill_(dim=0, index = idx_key_mask, value = int(self.model_mask_id))
            MKR_labels = (torch.ones_like(input_ids) * -100)
            MKR_labels.index_put_(indices = (idx_key_mask, ) , values = torch.index_select(orig_input_ids, dim = 0 , index = idx_key_mask))

            #  -------------- MER --------------- 
            MER_tokens.index_fill_(dim=0, index = idx_out_mask, value = int(self.model_mask_id))

            # if idx == 10:
            #     print(orig_token)
            #     print(MKR_tokens)
            #     print(MER_tokens)
            #     print(MKR_labels)
            #     asdfasdf
            
            self.list_all_input_ids.append((orig_token, MKR_tokens, MER_tokens,))
            self.list_word_ids.append(word_ids.clone())
            self.list_orig_text.append(orig_text)
            self.list_all_labels.append((MKR_labels, labels,))
            
        assert len(self.list_all_input_ids) == len(self.list_word_ids) == len(self.list_orig_text) == len(self.list_all_labels)
    
    
    def __getitem__(self, idx):
        sample = {  'all_input_ids'  : self.list_all_input_ids[idx],
                    'word_ids'       : self.list_word_ids[idx],
                    'orig_text'      : self.list_orig_text[idx],
                    'all_labels'     : self.list_all_labels[idx]}
        return sample
    
    
    
    
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################

class PROPMLMDataset(Dataset):
    
    def __init__(self, data_loader, classitoken_model, model_mask_id, tokenizer):
        
        self.tokenizer = tokenizer
        self.make_PROP_MLM_ds(data_loader, classitoken_model, model_mask_id)
        
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
    
    # Use the trained model to do inference on domain dataset to creats masked domain ds
    def make_PROP_MLM_ds(self, data_loader, model, model_mask_id):
        
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
            masked_input_ids[important_idx_seq, important_idx_pos] = model_mask_id
            # print(masked_input_ids.shape)
            
            labels    = torch.ones_like(att_mask).to(device) * -100 # init all labels with -100
            # put original token input_ids at the position of important tokens
            masked_labels  = labels.index_put(indices = (important_idx_seq, important_idx_pos) , values = important_input_ids)
            
            for i in range(input_ids.shape[0]):
            
                self.list_input_ids.append(masked_input_ids[i].clone())
                self.list_word_ids.append(word_ids[i].clone())
                self.list_attention_mask.append(torch.ones_like(masked_input_ids[i]))
                self.list_orig_text.append(orig_text[i])
                self.list_masked_text.append(self.tokenizer.decode(masked_input_ids[i].clone()))
                self.list_labels.append(masked_labels[i].clone())
                
                # print(masked_input_ids[i].clone())
                # print(masked_labels[i].clone())
                
        assert len(self.list_input_ids) == len(self.list_word_ids) == len(self.list_attention_mask) == len(self.list_orig_text) == len(self.list_masked_text) == len(self.list_labels)
        
        
class PROPMASKERDataset(Dataset):
    
    def __init__(self, classichunkds, classitoken_model, tokenizer):
        
        self.tokenizer        = tokenizer
        self.model_mask_id    = self.tokenizer(self.tokenizer.mask_token)['input_ids'][1] 
        self.model_cls_id     = self.tokenizer(self.tokenizer.cls_token)['input_ids'][1]
        self.model_sep_id     = self.tokenizer(self.tokenizer.sep_token)['input_ids'][1]  
        self.make_PROP_MASKER_ds(classichunkds)
        
        del classitoken_model, classichunkds
        
    def __len__(self):
        return len(self.list_all_input_ids)

    def __getitem__(self, idx):
        sample = {  'all_input_ids'  : self.list_all_input_ids[idx],
                    'word_ids'       : self.list_word_ids[idx],
                    'orig_text'      : self.list_orig_text[idx],
                    'all_labels'     : self.list_all_labels[idx]}
        return sample
    
    # Use important_words_info to make the label for training TokenClassification on this D(task) dataset
    def make_PROP_MASKER_ds(self, classichunkds):
        
        self.list_all_input_ids = []
        self.list_word_ids      = []
        self.list_orig_text     = []
        self.list_all_labels    = []

        for idx, data in enumerate(classichunkds):  
            sys.stdout.write(str(idx))
            
            input_ids    = data['input_ids'].clone().reshape(1,-1).to(device)
            att_mask     = data['attention_mask'].clone().reshape(1,-1).to(device)
            
            word_ids     = data['word_ids'].clone()
            orig_text    = data['orig_text']
            labels       = data['labels']

            orig_input_ids = data['input_ids'].clone().cpu()
            orig_token     = orig_input_ids.clone()
            MKR_tokens     = orig_input_ids.clone()  # masked token (Masked Keywords Reconstruction)
            MER_tokens     = orig_input_ids.clone()  # outlier token (Masked Entropy Regularization)
            
            
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids = input_ids, attention_mask = att_mask)
            
            pred = torch.argmax(torch.softmax(outputs.logits.detach(), dim = 2), dim = 2) # bs, seq_len
            
            # get index of important tokens
            important_idx = (pred == 1).nonzero(as_tuple=True)[1].cpu()
            # print(important_idx)
            
            #  --------------- MKR ---------------
            MKR_tokens[important_idx] = self.model_mask_id
            MKR_tokens[0]  = self.model_cls_id
            MKR_tokens[-1] = self.model_sep_id
            # print(MKR_tokens)
            
            MKR_labels = (torch.ones_like(orig_input_ids) * -100)
            MKR_labels.index_put_(indices = (important_idx, ) , values = torch.index_select(orig_input_ids, dim = 0 , index = important_idx))
            MKR_labels[0]  = -100
            MKR_labels[-1] = -100
            # print(MKR_labels)
            
            #  -------------- MER --------------- 
            idx_out_mask = torch.tensor([idx for idx in range(1,orig_input_ids.shape[0]-1) if idx not in important_idx ]).long()
            # print(idx_out_mask)
            
            MER_tokens.index_fill_(dim=0, index = idx_out_mask, value = int(self.model_mask_id))
            MER_tokens[0]  = self.model_cls_id
            MER_tokens[-1] = self.model_sep_id
            # print(MER_tokens)
            
            # print(orig_token)
            # print(MKR_tokens)
            # print(MER_tokens)
            # print(MKR_labels)
            
            self.list_all_input_ids.append((orig_token, MKR_tokens, MER_tokens,))
            self.list_word_ids.append(word_ids.clone())
            self.list_orig_text.append(orig_text)
            self.list_all_labels.append((MKR_labels, labels,))
            
        assert len(self.list_all_input_ids) == len(self.list_word_ids) == len(self.list_orig_text) == len(self.list_all_labels)
        

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
        
        
class NNMASKERDataset(Dataset):
    
    def __init__(self, classichunkds, classitoken_model, tokenizer):
        
        self.tokenizer        = tokenizer
        self.model_mask_id    = self.tokenizer(self.tokenizer.mask_token)['input_ids'][1] 
        self.model_cls_id     = self.tokenizer(self.tokenizer.cls_token)['input_ids'][1]
        self.model_sep_id     = self.tokenizer(self.tokenizer.sep_token)['input_ids'][1]  
        self.make_NN_MASKER_ds(classichunkds)
        
        del classitoken_model, classichunkds
        
    def __len__(self):
        return len(self.list_all_input_ids)

    def __getitem__(self, idx):
        sample = {  'all_input_ids'  : self.list_all_input_ids[idx],
                    'word_ids'       : self.list_word_ids[idx],
                    'orig_text'      : self.list_orig_text[idx],
                    'all_labels'     : self.list_all_labels[idx]}
        return sample
    
    # Use important_words_info to make the label for training TokenClassification on this D(task) dataset
    def make_NN_MASKER_ds(self, classichunkds):
        
        self.list_all_input_ids = []
        self.list_word_ids      = []
        self.list_orig_text     = []
        self.list_all_labels    = []

        for idx, data in enumerate(classichunkds):  
            sys.stdout.write(str(idx))
            
            input_ids    = data['input_ids'].clone().reshape(1,-1).to(device)
            att_mask     = data['attention_mask'].clone().reshape(1,-1).to(device)
            
            word_ids     = data['word_ids'].clone()
            orig_text    = data['orig_text']
            labels       = data['labels']

            orig_input_ids = data['input_ids'].clone().cpu()
            orig_token     = orig_input_ids.clone()
            MKR_tokens     = orig_input_ids.clone()  # masked token (Masked Keywords Reconstruction)
            MER_tokens     = orig_input_ids.clone()  # outlier token (Masked Entropy Regularization)
            
            
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids = input_ids, attention_mask = att_mask)
            
            pred = torch.argmax(torch.softmax(outputs.logits.detach(), dim = 2), dim = 2) # bs, seq_len
            
            # get index of important tokens
            important_idx = (pred == 1).nonzero(as_tuple=True)[1].cpu()
            # print(important_idx)
            
            #  --------------- MKR ---------------
            MKR_tokens[important_idx] = self.model_mask_id
            MKR_tokens[0]  = self.model_cls_id
            MKR_tokens[-1] = self.model_sep_id
            # print(MKR_tokens)
            
            MKR_labels = (torch.ones_like(orig_input_ids) * -100)
            MKR_labels.index_put_(indices = (important_idx, ) , values = torch.index_select(orig_input_ids, dim = 0 , index = important_idx))
            MKR_labels[0]  = -100
            MKR_labels[-1] = -100
            # print(MKR_labels)
            
            #  -------------- MER --------------- 
            idx_out_mask = torch.tensor([idx for idx in range(1,orig_input_ids.shape[0]-1) if idx not in important_idx ]).long()
            # print(idx_out_mask)
            
            MER_tokens.index_fill_(dim=0, index = idx_out_mask, value = int(self.model_mask_id))
            MER_tokens[0]  = self.model_cls_id
            MER_tokens[-1] = self.model_sep_id
            # print(MER_tokens)
            
            # print(orig_token)
            # print(MKR_tokens)
            # print(MER_tokens)
            # print(MKR_labels)
            
            self.list_all_input_ids.append((orig_token, MKR_tokens, MER_tokens,))
            self.list_word_ids.append(word_ids.clone())
            self.list_orig_text.append(orig_text)
            self.list_all_labels.append((MKR_labels, labels,))
            
        assert len(self.list_all_input_ids) == len(self.list_word_ids) == len(self.list_orig_text) == len(self.list_all_labels)
        
class PMIMASKERDataset(Dataset):
    
    def __init__(self, chunkdataset, tokenizer, keyword_list):
        
        self.tokenizer        = tokenizer
        self.SPECIAL_TOKENS   = tokenizer.all_special_ids
        self.model_mask_id    = self.tokenizer(self.tokenizer.mask_token)['input_ids'][1] 
        self.model_cls_id     = self.tokenizer(self.tokenizer.cls_token)['input_ids'][1]
        self.model_sep_id     = self.tokenizer(self.tokenizer.sep_token)['input_ids'][1]
        self.model_max_length = (self.tokenizer.model_max_length)
        self.vocab            = [ i for i in range(tokenizer.vocab_size) if (i not in self.SPECIAL_TOKENS)]
        
        self.chunkdataset     = chunkdataset
        self.keyword_list     = keyword_list
        self.chunk_max_length = self.model_max_length - 2
        self.limit_mask       = round(0.15 * self.chunk_max_length)
        
        self.make_PMI_MASKER_ds()
        
    def __len__(self):
        return len(self.list_all_input_ids)

    def __getitem__(self, idx):
        sample = {  'all_input_ids'  : self.list_all_input_ids[idx],
                    'word_ids'       : self.list_word_ids[idx],
                    'orig_text'      : self.list_orig_text[idx],
                    'all_labels'     : self.list_all_labels[idx]}
        return sample

    def make_PMI_MASKER_ds(self):
        
        self.list_all_input_ids = []
        self.list_word_ids      = []
        self.list_orig_text     = []
        self.list_all_labels    = []
        
        for idx in range(len(self.chunkdataset)):
            
            sys.stdout.write(str(idx))
        
            chunk_sample = self.chunkdataset.__getitem__(idx)

            input_ids    = chunk_sample['input_ids'].clone()
            word_ids     = chunk_sample['word_ids'].clone()
            orig_text    = chunk_sample['orig_text']
            labels       = chunk_sample['labels']

            orig_input_ids = chunk_sample['input_ids'].clone()
            orig_token     = orig_input_ids.clone()
            MKR_tokens     = orig_input_ids.clone()  # masked token (Masked Keywords Reconstruction)
            MER_tokens     = orig_input_ids.clone()  # outlier token (Masked Entropy Regularization)
            
            
            # find index of keywords tokens
            idx_mask_all = []
            for i in range(1, input_ids.shape[0]-1):
                for j in range(len(self.keyword_list)):
                    len_key = len(self.keyword_list[j]) # length of this keyword
                    if torch.equal(input_ids[i:i+len_key] , self.keyword_list[j]) :
                        # print(chunk_input_ids[idx][i:i+len_key])
                        # print(self.keyword_list[j])
                        idx_mask_all.extend([list(range(i,i+len_key))])
            
            to_rm = []
            # use larger unit for masking
            for sub_list in idx_mask_all:
                for test_list in idx_mask_all:
                    if (any(x in test_list for x in sub_list)):
                        if len(test_list) > len(sub_list):
                            to_rm.append(sub_list)
                        if (len(test_list) == len(sub_list)) and (test_list != sub_list) :
                            to_rm.append(sub_list)
            
            idx_mask_all = [i for i in idx_mask_all if i not in to_rm]
            num_mask_now = sum([len(i) for i in idx_mask_all])
            
            idx_mask_all = torch.tensor([ j for i in idx_mask_all for j in i ]).long()
            
             #  --------------- MKR ---------------
            MKR_tokens[idx_mask_all] = self.model_mask_id
            MKR_labels = (torch.ones_like(orig_input_ids) * -100)
            MKR_labels.index_put_(indices = (idx_mask_all, ) , values = torch.index_select(orig_input_ids, dim = 0 , index = idx_mask_all))
            
            #  -------------- MER --------------- 
            idx_out_mask = torch.tensor([idx for idx in range(1,orig_input_ids.shape[0]-1) if idx not in idx_mask_all ]).long()
            # print(idx_out_mask)
            
            MER_tokens.index_fill_(dim=0, index = idx_out_mask, value = int(self.model_mask_id))
            MER_tokens[0]  = self.model_cls_id
            MER_tokens[-1] = self.model_sep_id
            
            # print(orig_token)
            # print(MKR_tokens)
            # print(MER_tokens)
            # print(MKR_labels)         

            self.list_all_input_ids.append((orig_token, MKR_tokens, MER_tokens,))
            self.list_word_ids.append(word_ids.clone())
            self.list_orig_text.append(orig_text)
            self.list_all_labels.append((MKR_labels, labels,))
            
        assert len(self.list_all_input_ids) == len(self.list_word_ids) == len(self.list_orig_text) == len(self.list_all_labels)
        
        
class NNMprobASKERDataset(Dataset):
    
    def __init__(self, classichunkds, classitoken_model, tokenizer):
        
        self.tokenizer        = tokenizer
        self.model_mask_id    = self.tokenizer(self.tokenizer.mask_token)['input_ids'][1] 
        self.model_cls_id     = self.tokenizer(self.tokenizer.cls_token)['input_ids'][1]
        self.model_sep_id     = self.tokenizer(self.tokenizer.sep_token)['input_ids'][1]  
        self.make_NNprob_MASKER_ds(classichunkds)
        
        del classitoken_model, classichunkds
        
    def __len__(self):
        return len(self.list_all_input_ids)

    def __getitem__(self, idx):
        sample = {  'all_input_ids'  : self.list_all_input_ids[idx],
                    'word_ids'       : self.list_word_ids[idx],
                    'orig_text'      : self.list_orig_text[idx],
                    'all_labels'     : self.list_all_labels[idx]}
        return sample
    
    # Use important_words_info to make the label for training TokenClassification on this D(task) dataset
    def make_NNprob_MASKER_ds(self, classichunkds):
        
        self.list_all_input_ids = []
        self.list_word_ids      = []
        self.list_orig_text     = []
        self.list_all_labels    = []

        for idx, data in enumerate(classichunkds):  
            sys.stdout.write(str(idx))
            
            input_ids    = data['input_ids'].clone().reshape(1,-1).to(device)
            att_mask     = data['attention_mask'].clone().reshape(1,-1).to(device)
            
            word_ids     = data['word_ids'].clone()
            orig_text    = data['orig_text']
            labels       = data['labels']

            orig_input_ids = data['input_ids'].clone().cpu()
            orig_token     = orig_input_ids.clone()
            MKR_tokens     = orig_input_ids.clone()  # masked token (Masked Keywords Reconstruction)
            MER_tokens     = orig_input_ids.clone()  # outlier token (Masked Entropy Regularization)
            
            
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids = input_ids, attention_mask = att_mask)
            
            pred = torch.argmax(torch.softmax(outputs.logits.detach(), dim = 2), dim = 2) # bs, seq_len
            
            # get index of important tokens
            important_idx = (pred == 1).nonzero(as_tuple=True)[1].cpu()
            print(important_idx)
            
            idx_out_mask = torch.tensor([idx for idx in range(1, orig_input_ids.shape[0]-1) if idx not in important_idx ]).long()
            print(idx_out_mask)

            
            p = 0.9
            q = 0.9
            # mask the keywords only where p_mask < 0.5
            p_mask = torch.rand(torch.tensor(important_idx).shape)
            important_idx = torch.tensor([idx_k for i, idx_k in enumerate(important_idx) if p_mask[i] < p]).long()
            print(important_idx)

            # mask the context only where q_mask < 0.9
            q_mask = torch.rand(idx_out_mask.shape)
            idx_out_mask = torch.tensor([idx_k for i, idx_k in enumerate(idx_out_mask) if q_mask[i] < q])
            print(idx_out_mask)

            
            #  --------------- MKR ---------------
            MKR_tokens[important_idx] = self.model_mask_id
            MKR_tokens[0]  = self.model_cls_id
            MKR_tokens[-1] = self.model_sep_id
            # print(MKR_tokens)
            
            MKR_labels = (torch.ones_like(orig_input_ids) * -100)
            MKR_labels.index_put_(indices = (important_idx, ) , values = torch.index_select(orig_input_ids, dim = 0 , index = important_idx))
            MKR_labels[0]  = -100
            MKR_labels[-1] = -100
            # print(MKR_labels)
            
            #  -------------- MER --------------- 
            
            MER_tokens.index_fill_(dim=0, index = idx_out_mask, value = int(self.model_mask_id))
            MER_tokens[0]  = self.model_cls_id
            MER_tokens[-1] = self.model_sep_id
            
            # print(MER_tokens)
            
            # print(orig_token)
            # print(MKR_tokens)
            # print(MER_tokens)
            # print(MKR_labels)

            self.list_all_input_ids.append((orig_token, MKR_tokens, MER_tokens,))
            self.list_word_ids.append(word_ids.clone())
            self.list_orig_text.append(orig_text)
            self.list_all_labels.append((MKR_labels, labels,))
            
            # asdfasdf
            
        assert len(self.list_all_input_ids) == len(self.list_word_ids) == len(self.list_orig_text) == len(self.list_all_labels)
        
class PROPprobmMASKERDataset(Dataset):
    
    def __init__(self, classichunkds, classitoken_model, tokenizer):
        
        self.tokenizer        = tokenizer
        self.model_mask_id    = self.tokenizer(self.tokenizer.mask_token)['input_ids'][1] 
        self.model_cls_id     = self.tokenizer(self.tokenizer.cls_token)['input_ids'][1]
        self.model_sep_id     = self.tokenizer(self.tokenizer.sep_token)['input_ids'][1]  
        self.make_PROPprob_MASKER_ds(classichunkds)
        
        # del classitoken_model, classichunkds
        
    def __len__(self):
        return len(self.list_all_input_ids)

    def __getitem__(self, idx):
        sample = {  'all_input_ids'  : self.list_all_input_ids[idx],
                    'word_ids'       : self.list_word_ids[idx],
                    'orig_text'      : self.list_orig_text[idx],
                    'all_labels'     : self.list_all_labels[idx]}
        return sample
    
    # Use important_words_info to make the label for training TokenClassification on this D(task) dataset
    def make_PROPprob_MASKER_ds(self, classichunkds):
        
        self.list_all_input_ids = []
        self.list_word_ids      = []
        self.list_orig_text     = []
        self.list_all_labels    = []

        for idx, data in enumerate(classichunkds):  
            sys.stdout.write(str(idx))
            
            input_ids    = data['input_ids'].clone().reshape(1,-1).to(device)
            att_mask     = data['attention_mask'].clone().reshape(1,-1).to(device)
            
            word_ids     = data['word_ids'].clone()
            orig_text    = data['orig_text']
            labels       = data['labels']

            orig_input_ids = data['input_ids'].clone().cpu()
            orig_token     = orig_input_ids.clone()
            MKR_tokens     = orig_input_ids.clone()  # masked token (Masked Keywords Reconstruction)
            MER_tokens     = orig_input_ids.clone()  # outlier token (Masked Entropy Regularization)
            
            
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids = input_ids, attention_mask = att_mask)
            
            pred = torch.argmax(torch.softmax(outputs.logits.detach(), dim = 2), dim = 2) # bs, seq_len
            
            # get index of important tokens
            important_idx = (pred == 1).nonzero(as_tuple=True)[1].cpu()
            # print(important_idx)
            
            idx_out_mask = torch.tensor([idx for idx in range(1, orig_input_ids.shape[0]-1) if idx not in important_idx ]).long()
            # print(idx_out_mask)

            
            p = 0.9
            q = 0.9
            # mask the keywords only where p_mask < 0.5
            p_mask = torch.rand(torch.tensor(important_idx).shape)
            important_idx = torch.tensor([idx_k for i, idx_k in enumerate(important_idx) if p_mask[i] < p]).long()
            # print(important_idx)

            # mask the context only where q_mask < 0.9
            q_mask = torch.rand(idx_out_mask.shape)
            idx_out_mask = torch.tensor([idx_k for i, idx_k in enumerate(idx_out_mask) if q_mask[i] < q])
            # print(idx_out_mask)

            
            #  --------------- MKR ---------------
            MKR_tokens[important_idx] = self.model_mask_id
            MKR_tokens[0]  = self.model_cls_id
            MKR_tokens[-1] = self.model_sep_id
            # print(MKR_tokens)
            
            MKR_labels = (torch.ones_like(orig_input_ids) * -100)
            MKR_labels.index_put_(indices = (important_idx, ) , values = torch.index_select(orig_input_ids, dim = 0 , index = important_idx))
            MKR_labels[0]  = -100
            MKR_labels[-1] = -100
            # print(MKR_labels)
            
            #  -------------- MER --------------- 
            
            MER_tokens.index_fill_(dim=0, index = idx_out_mask, value = int(self.model_mask_id))
            MER_tokens[0]  = self.model_cls_id
            MER_tokens[-1] = self.model_sep_id
            
            # print(MER_tokens)
            
            # print(orig_token)
            # print(MKR_tokens)
            # print(MER_tokens)
            # print(MKR_labels)
            
            self.list_all_input_ids.append((orig_token, MKR_tokens, MER_tokens,))
            self.list_word_ids.append(word_ids.clone())
            self.list_orig_text.append(orig_text)
            self.list_all_labels.append((MKR_labels, labels,))
            
            # asdfasdf
            
        assert len(self.list_all_input_ids) == len(self.list_word_ids) == len(self.list_orig_text) == len(self.list_all_labels)
        
        
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
                
                # print(this_input_ids)
                # print(label)
                # print(word_ids[i])
                # print(orig_text[i])
                
                # print(masked_input_ids[10].clone())
                # print(masked_labels[10].clone())
                
        assert len(self.list_input_ids) == len(self.list_word_ids) == len(self.list_attention_mask) == len(self.list_orig_text) == len(self.list_masked_text) == len(self.list_labels)
        
class PROPrandomMLMDataset(Dataset):
    
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
        
        self.make_PROPrandom_MLM_ds(data_loader, classitoken_model)
        
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
    
    # Use the trained model to do inference on domain dataset to creats masked domain ds
    def make_PROPrandom_MLM_ds(self, data_loader, model):
        
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
                
                # print(this_input_ids)
                # print(label)
                # print(word_ids[i])
                # print(orig_text[i])
                
                # print(masked_input_ids[10].clone())
                # print(masked_labels[10].clone())
                
        assert len(self.list_input_ids) == len(self.list_word_ids) == len(self.list_attention_mask) == len(self.list_orig_text) == len(self.list_masked_text) == len(self.list_labels)
