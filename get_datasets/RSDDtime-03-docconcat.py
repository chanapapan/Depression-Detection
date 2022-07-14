#!/usr/bin/env python
# coding: utf-8

# # Select Class

# In[4]:


d_class      = 'depression' # 'depression'

dataset_type = 'time'
month        = 3


# In[5]:


d1_text_path = f'../data_{dataset_type}/{d_class}/finclean_{month}m_0529.txt'
with open(d1_text_path) as text_file:
    d1_text_all = text_file.readlines()
    print(len(d1_text_all))
    
d1_user_path = f'../data_{dataset_type}/{d_class}/user_id_{month}m.txt'
with open(d1_user_path) as user_id:
    d1_user_all = user_id.readlines()
    d1_user_all = [int(x) for x in d1_user_all]
    unique_user = [] 
    [unique_user.append(x) for x in d1_user_all if x not in unique_user]
    print("unique_user = ", len(unique_user))    


# ### CONCAT

# In[6]:


# one line = all posts from one user
count_users = 0

d1_concat_path = f'../data_{dataset_type}/{d_class}/concat_{month}m_0529.txt'

with open(d1_concat_path, 'w') as d1_concat:
    for user in unique_user:
        # print("user_id = ", user)
        
        # Get all posts from this user 
        num_user_posts = len([i for i,x in enumerate(d1_user_all) if x == user])
        
        if num_user_posts < 10: # if the user has less than 10 psots in the past month, we do not include them in the dataset
            print("less than 10 posts !")
            continue
        
        user_posts = [d1_text_all[i][:-1] for i,x in enumerate(d1_user_all) if x == user] # [:-1] bc not includeing \n at the end of every line
        # print("user_posts = ", user_posts)
        
        user_posts = ' '.join(user_posts)
        # print("Joined user_post = ", user_posts)
        
        # remove space again
        user_posts = ' '.join(user_posts.split())
        # print("user_post again = ", user_posts)
        
        if len(user_posts) == 0: # if all tweets has len = 0
            continue 
        
        count_users += 1
        
        d1_concat.write(user_posts + "\n")
print(count_users)

