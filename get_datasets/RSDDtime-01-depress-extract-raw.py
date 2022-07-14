#!/usr/bin/env python
# coding: utf-8

# ### RSDD-Time : Depression users only

# In[6]:


import csv, os 
import jsonlines
from IPython.display import clear_output
import datetime
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

filepath = f'../OP_datasets/RSDD-Time/RSDD-Time.json'

count_current  = 0
count_doubt    = 0
count_falsepos = 0
count_valid = 0

import jsonlines
with jsonlines.open(filepath) as reader:
    for obj in reader:
        my_dict = obj
        # print("user_id = ", my_dict['id'])
        # print("created date = ", my_dict['created_utc'])
    
        conditionstatus  = my_dict['consensus']['diagnosis']['conditionstatus']
        diagnosisindoubt = my_dict['consensus']['diagnosis']['diagnosisindoubt']
        falsepositive    = my_dict['consensus']['diagnosis']['falsepositive']
        
        if conditionstatus == 1 :
            count_current += 1
        if diagnosisindoubt == True: 
            count_doubt += 1
        if falsepositive == True: 
            count_falsepos += 1
            
        if (conditionstatus == 1) and (diagnosisindoubt == False) and (falsepositive == False):
            count_valid += 1
        
print(count_current)
print(count_doubt)
print(count_falsepos)
print(count_valid)


# In[7]:


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


# In[8]:


filepath = f'../OP_datasets/RSDD-Time/RSDD-Time.json'
output_path = '../data_time/RSDD_time_user.csv'

# get user_id and the date they posted self declared
header = ['user_id', 'created_date']

count_valid = 0
all_user_id = []
all_created_date = []

with jsonlines.open(filepath) as reader:
    
    for obj in reader:           
        my_dict = obj
        
        user_id          = my_dict['id']
        created_date     = my_dict['created_utc']
        conditionstatus  = my_dict['consensus']['diagnosis']['conditionstatus']
        diagnosisindoubt = my_dict['consensus']['diagnosis']['diagnosisindoubt']
        falsepositive    = my_dict['consensus']['diagnosis']['falsepositive']

        if (conditionstatus == 1) and (diagnosisindoubt == False) and (falsepositive == False):
            count_valid += 1
            all_user_id.append(user_id)
            all_created_date.append(created_date)
            row = [user_id, created_date]
            # save_result_csv( header, row, output_path)
        
            
print("count_valid : ", count_valid)
print(len(all_user_id))
print(len(all_created_date))


# In[9]:


# key = user_id, value = self declare timestamp
user_time = { int(id):datetime.fromtimestamp(all_created_date[i]) for i,id in enumerate(all_user_id) }
# print(user_time)


#     conditionstatus = 1 'current' : 245 users
#     diagnosisindoubt = True       : 16 users
#     falsepositive = True          : 25 users
# 
#     (conditionstatus == 1) and (diagnosisindoubt == False) and (falsepositive == False) : 244 users !!!

# ### Get ALL posts from VALID USERS
# 
# 213 users => 372081 posts
# 
# (no time filter yet)

# In[22]:


count_matched = 0
count_posts = 0
matched_id = []

dataset = ['training', 'validation', 'testing'] # validation , testing
output_path = '../data_time/raw_all.csv'

for ds in dataset :
    filepath = f'../OP_datasets/RSDD/{ds}'
    

    header = ['user_id', 'created_time', 'text']
    with jsonlines.open(filepath) as reader:
        
        for obj in reader:     
            my_dict = obj[0] # dict containing 'id', 'label, 'posts' as keys
            user_id = my_dict['id']
            
            if user_id in all_user_id :
                count_matched += 1
                matched_id.append(user_id)
                
                for each_post in my_dict['posts']:
                    count_posts += 1
                    created_time = each_post[0]
                    text         = each_post[1]
                    text = ' '.join(str(text).split()) # remove consecutive whitespace \n \t \s HERE instead of the dataframe later
                    row = [user_id, created_time, text]
                    # save_result_csv( header, row, output_path)
            
print(count_matched)
print(count_posts)


# In[ ]:


missing_id = []
for id in all_user_id:
    if id not in matched_id:
        missing_id.append(id)
print(len(missing_id))


#     matched in 'training' = 76
#     matched in 'validation' = 69
#     matched in 'testing' = 68
#     Matched Total = 213
# 
#     missing = 28
# 
#     213 + 28 = 241 (still 3 missing)
# 

# ### Get Posts from Specific Timeframe
# ( 1/ 2/ 3/ months after declared tweets)

# In[10]:


raw_text = pd.read_csv('../data_time/depression/raw_all.csv', sep='|', header = 0, lineterminator='\n', dtype= str, index_col=False)
raw_text["user_id"] = raw_text['user_id'].apply(pd.to_numeric)
raw_text["created_time"] = raw_text['created_time'].apply(lambda x: datetime.fromtimestamp(int(x)))
print(raw_text.shape)


# In[ ]:


raw_text.head()


# In[12]:


months = [1, 2, 3]
header = ['user_id', 'text']

for month in months:
    output_path = f'../data_time/raw_{month}m.csv'

    count_posts = 0
    count_match = 0
    count_no_match = 0

    for user_id, user_declare_time in user_time.items():
        # print(user_id, user_declare_time)
        all_this_user = raw_text.loc[raw_text['user_id'] == user_id]
        if all_this_user.shape[0] == 0:
            count_no_match += 1
            # print('no match found')
            
        count_match += 1
        # print(all_this_user.shape)
        
        last_valid_time = user_declare_time + relativedelta(months=month)
        # print(user_declare_time)
        # print(last_valid_time)
        
        valid_posts = all_this_user.loc[all_this_user['created_time'] >= user_declare_time]
        valid_posts = valid_posts.loc[valid_posts['created_time'] <= last_valid_time]
        
        # print(valid_posts.shape)
        
        for idx in range(valid_posts.shape[0]):
            count_posts += 1
            user_id = valid_posts.iloc[idx,:].loc['user_id']
            text = valid_posts.iloc[idx,:].loc['text']
            text = ' '.join(str(text).split()) # IMPORTANT!! remove consecutive whitespace \n \t \s
            row = [user_id, text]
            save_result_csv( header,  row,  output_path)
        
    print(month)
    print("saved to : ", output_path)
    print(count_posts)
    print(count_match)
    print(count_no_match)


#     1 Month
#     18883 posts
#     241 users
#     
#     2 Months
#     34626 posts
#     241 users
#     
#     3 Months
#     46892 posts
#     241 users

# 
