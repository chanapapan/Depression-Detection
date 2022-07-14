#!/usr/bin/env python
# coding: utf-8

# ### RSDD-Time : Control users only

# In[1]:


import csv, os 
import jsonlines
from IPython.display import clear_output
import pandas as pd
import datetime
from datetime import datetime
from dateutil.relativedelta import relativedelta


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


# ### Get unique user id

# In[3]:


d1_text_path = f'../data_time/control/user_id_all.txt'
with open(d1_text_path) as text_file:
    all_user_id = text_file.readlines()
    print(len(all_user_id))


# In[5]:


# number of control users that we want
n = 3000

import random
unique_user = list(set(all_user_id))
print(len(unique_user))

random_user_id = random.sample(unique_user, n)
random_user_id = [int(x) for x in random_user_id]
print(len(random_user_id))


#     Get 3000 users first

# ### Get ALL posts from RANDOM USERS

# In[6]:


count_matched = 0
count_posts = 0
matched_id = []

output_path = '../data_time/control/raw_random.csv'
filepath = f'../OP_datasets/RSDD/training'

header = ['user_id', 'created_time', 'text']
with jsonlines.open(filepath) as reader:
    
    for obj in reader:
        clear_output(wait=True)
        my_dict = obj[0] # dict containing 'id', 'label, 'posts' as keys
        user_id = int(my_dict['id'])
        
        if user_id in random_user_id : # if user_id is in random_user_id
            count_matched += 1
            matched_id.append(user_id)
            
            for each_post in my_dict['posts']:
                count_posts += 1
                created_time = datetime.fromtimestamp(int(each_post[0]))
                text         = each_post[1]
                text = ' '.join(str(text).split()) # remove consecutive whitespace \n \t \s HERE instead of the dataframe later
                row = [user_id, created_time, text]
                save_result_csv( header, row, output_path)

        print(count_matched)
        print(count_posts)


# ### Get Posts from Specific Timeframe
# ( 1/ 2/ 3/ months after declared tweets)

# In[7]:


raw_text = pd.read_csv('../data_time/control/raw_random.csv', sep='|', header = 0, lineterminator='\n', dtype= str, index_col=False)
raw_text["user_id"] = raw_text['user_id'].apply(pd.to_numeric)
raw_text["created_time"] = raw_text['created_time'].apply(pd.to_datetime)
print(raw_text.shape)


# In[ ]:


raw_text.head()


# In[9]:


months = [1, 2, 3]
header = ['user_id', 'text']

for month in months:
    output_path = f'../data_time/control/raw_{month}m.csv'

    count_posts = 0
    count_match = 0
    count_no_match = 0

    for user_id in random_user_id:

        all_this_user = raw_text.loc[raw_text['user_id'] == user_id] # get all posts of this user
        user_declare_time = all_this_user.sample(n=1).iloc[0]['created_time'] # randomly select 1 posts as the beginning of time
        
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
#     247368 posts
#      users
#     
#     2 Months
#     380996 posts
#      users
#     
#     3 Months
#     470356 posts
#      users

# 
