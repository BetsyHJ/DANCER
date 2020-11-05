#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

path = "./ml-latest-small/"

ratingfile = path + 'ratings.csv'
sep = ','

def load_ratings(ratingfile, sep):
    ratings = pd.read_csv(ratingfile, sep=sep, header=0)
    ratings.columns = ['UserId', 'ItemId', 'rating', 'timestamp']
    return ratings

ratings = load_ratings(ratingfile, sep)
# print(ratings)

#%%
# pt: when the item is pulished
import re
itemfile = path + 'movies.csv'
def load_item_pt(itemfile):
    items = pd.read_csv(itemfile, sep=sep, header=0)
    items.columns = ['ItemId', 'Name', 'Category']
    names = items['Name'] # the name includes the pulished year
    years = []
    p1 = re.compile(r'[(](.*?)[)]', re.S)
    for name in names.to_list():
        if re.findall(p1, name):
            years.append(int(re.findall(p1, name)[-1][:4]))
        else:
            # print(name)
            years.append(-1)
    item_pt = pd.Series(years, index=items['ItemId'].to_list(), name='publish_year')
    return item_pt[item_pt>0]

pt_year = load_item_pt(itemfile)
# print(pt_year)
# exit(0)


def timestamp2year(timelist):
    years = []
    for t in timelist:
        years.append(time.localtime(int(t))[0])
    return years

def bias_overyear(ratings, pt_year, MaxAge=False):
    # ratings_sorted = ratings.sort_values(by=['timestamp'])
    # start_Unix, end_Unix = ratings_sorted['timestamp'][:1].values[0], ratings_sorted['timestamp'][-1:].values[0]
    # print(start_Unix, end_Unix)
    # start_year, end_year = time.localtime(int(start_Unix))[0], time.localtime(int(end_Unix))[0]
    # print('start_year and end_year are %d, %d' % (start_year, end_year))
    # ratings_sorted['timestamp'] = pd.to_datetime(ratings_sorted['timestamp'], unit='s')
    df = ratings.merge(pt_year, left_on='ItemId', right_index=True)
    ct = timestamp2year(df['timestamp'].to_list())
    pt = df['publish_year'].to_list()
    deltat = np.array(ct, dtype=int) - np.array(pt, dtype=int)
    # ## draw for \delta_Year-click
    # plt.hist(deltat, bins='auto')
    # # np.histogram(deltat)
    # plt.title("Year-Clicks")
    # plt.show()
    ## draw for \delta_Year-avg(i)
    scores = df['rating'].to_list()
    years = list(set(deltat))
    years.sort()
    pop, avg = [], []
    for year in years:
        n_clicks = np.where(deltat==year, 1, 0).sum()
        pop.append(n_clicks / (df[deltat==year]['ItemId'].nunique()))
        avg.append(np.where(deltat==year, scores, 0).sum() * 1.0 / n_clicks)

    if MaxAge:
        pop = pop[:MaxAge]
        years = years[:MaxAge]
        avg = avg[:MaxAge]

    plt.clf()
    ## double y-axis
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(years, pop)
    ax1.set_ylabel('Number of Clicks')
    ax1.tick_params(axis='y', colors='#1f77b4')
    ax1.yaxis.label.set_color('#1f77b4')
    ax2 = ax1.twinx()
    ax2.plot(years, avg, 'r')
    ax2.set_ylabel('Average rating')
    ax2.spines['left'].set_color('#1f77b4')
    ax2.spines['right'].set_color('r')
    ax2.tick_params(axis='y', colors='r')
    ax2.yaxis.label.set_color('r')
    plt.title("ItemAge (year) - #Clicks or AvgRatings")
    fig.set_size_inches(5, 3.5)
    fig.tight_layout()
    if MaxAge:
        fig.savefig("ItemAge-Clicks_AvgR_"+str(MaxAge)+".pdf")
    else:
        fig.savefig("ItemAge-Clicks_AvgR.pdf")
    # plt.show()
    fig.clf()

    return deltat

bias_overyear(ratings, pt_year, MaxAge=50)
