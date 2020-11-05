#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

path = "ml-latest-small/"
# # output - save to data file folder
out_path = "../data/" + path
import os
if not os.path.exists(out_path):
    os.mkdir(out_path)


ratingfile = path + 'ratings.csv'
sep = ','

def load_ratings(ratingfile, sep):
    ratings = pd.read_csv(ratingfile, sep=sep, header=0)
    ratings.columns = ['UserId', 'ItemId', 'rating', 'timestamp']
    return ratings
ratings = load_ratings(ratingfile, sep)

#%%

# print(ratings)

# # filter strategy
# 1. users with more than 10 interactions
user_num_ratings = ratings.groupby(['UserId']).size()
users = user_num_ratings[user_num_ratings>=10].index
print("How many users do not fit the condition?", ratings['UserId'].nunique()-len(users))
ratings = ratings[ratings['UserId'].isin(users)]


# # item publish time
# import re
# itemfile = path + 'movies.csv'
# def load_item_pt(itemfile):
#     items = pd.read_csv(itemfile, sep=sep, header=0)
#     items.columns = ['ItemId', 'Name', 'Category']
#     names = items['Name'] # the name includes the pulished year
#     years = []
#     p1 = re.compile(r'[(](.*?)[)]', re.S)
#     for name in names.to_list():
#         if re.findall(p1, name):
#             years.append(int(re.findall(p1, name)[-1][:4]))
#         else:
#             # print(name)
#             years.append(-1)
#     item_pt = pd.Series(years, index=items['ItemId'].to_list(), name='publish_year')
#     return item_pt[item_pt>0]
# pt_year = load_item_pt(itemfile)

# pt_file = "item_pt_year.csv"
# pt_year.to_csv(out_path + pt_file, sep=sep, index_label='ItemId', header='Year')



#%%
# # generate item dynamic pop
year_seconds = 365*24*60*60
items = ratings['ItemId']
times_rating = ratings['timestamp']
times_start = times_rating - year_seconds
pop, ctr = [], []
begin_at = time.time()
for idx in range(len(items)):
    item = items[idx]
    end_t = times_rating[idx]
    start_t = times_start[idx]
    num_total_clicks = (times_rating.between(start_t, end_t, inclusive=True)).sum()
    pop.append((times_rating[items==item].between(start_t, end_t, inclusive=True)).sum())
    ctr.append(pop[-1] * 1.0 / num_total_clicks)
    # print(idx, item, pop[-1], ctr[-1])
    if (idx + 1) % 1000 == 0:
        print(idx, "need", time.time()-begin_at, 's')

ratings['ctr'] = np.array(ctr, dtype=float)
# %%
# id change to idx
users = ratings['UserId'].unique()
items = ratings['ItemId'].unique()
UserId2Idx = pd.Series(data=np.arange(len(users)), index=users, dtype=int)
ItemId2Idx = pd.Series(data=np.arange(len(items)), index=items, dtype=int)
UserIdxs = UserId2Idx[ratings['UserId']].values
ItemIdxs = ItemId2Idx[ratings['ItemId']].values
UserId2Idx.to_csv(out_path + "UserId2Idx.csv")
ItemId2Idx.to_csv(out_path + "ItemId2Idx.csv")
ratings['UserId'] = UserIdxs
ratings['ItemId'] = ItemIdxs

#%%
# data split: leave-one-out
df_train, df_test = [], []
for u, group in ratings.groupby(['UserId']):
    sorted_ratings = group.sort_values('timestamp', ascending=True)
    # print(u, group, sorted_ratings)
    
    df_train.append(sorted_ratings[:-1])
    df_test.append(sorted_ratings[-1:])
    # print(df_train, '\n', df_test)
    # break
df_train = pd.concat(df_train, axis=0)
df_test = pd.concat(df_test, axis=0)

df_train.to_csv(out_path+'train.csv', sep=sep, index=False)
df_test.to_csv(out_path+'test.csv', sep=sep, index=False)


# %%
