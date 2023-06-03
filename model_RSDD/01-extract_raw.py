import csv, os 
import jsonlines

def save_result_csv( _header_name, _row_data, _path ):
    filename    = _path
    mode        = 'a' if os.path.exists(filename) else 'w'
    with open(f"{filename}", mode) as myfile:
        fileEmpty   = os.stat(filename).st_size == 0
        writer      = csv.DictWriter(myfile, delimiter='|', lineterminator='\n',fieldnames=_header_name)
        if fileEmpty:
            writer.writeheader()  # file doesn't exist yet, write a header

        row_dic = dict(zip(_header_name, _row_data))
        writer.writerow( row_dic )
        myfile.close()
        


for dataset in ['training', 'testing', 'validation']:
    
    print(dataset)

    filepath = f'../OP_datasets/RSDD/{dataset}'
    header = ['user_id', 'timestamp', 'text']

    depression_path = f'data_depression/{dataset}/raw.csv'
    control_path    = f'data_control/{dataset}/raw.csv'

    count_user = 0
    count_control = 0
    count_depression = 0

    with jsonlines.open(filepath) as reader:
        
        for obj in reader:     
            count_user += 1
            print("Count user : ", count_user)
            
            my_dict = obj[0] # dict containing 'id', 'label, 'posts' as keys
            
            user_id = my_dict['id']
            label   = my_dict['label']
            
            # Separate into one file for each class
            if label == 'control':
                count_control += 1
                for post_list in my_dict['posts']:
                    timestamp = post_list[0]
                    text      = post_list[1]
                    text      = ' '.join(str(text).split()) # remove consecutive whitespace \n \t \s HERE instead of the dataframe later
                    row       = [user_id, timestamp,  text]
                    save_result_csv( header, row, control_path)
                    
                    
            if label == 'depression':
                count_depression += 1
                for post_list in my_dict['posts']:
                    timestamp = post_list[0]
                    text      = post_list[1]
                    text      = ' '.join(str(text).split()) # remove consecutive whitespace \n \t \s HERE instead of the dataframe later
                    row       = [user_id, timestamp,  text]
                    save_result_csv( header, row, depression_path)
                
    print("Dataset : ", dataset)
    print("All user : ", count_user)     
    print("Control user : ", count_control)
    print("Count_depression : ", count_depression)
    print("="*20)