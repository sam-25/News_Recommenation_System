import numpy as np     
import random    
import pandas as pd  
import os
import textwrap

news_data=pd.read_csv("preprocessed_news.csv")

# The following headings act like 'arms'
heading_list = ["Life", 'Entertainment', "World", 'Video']

cat_dict={}
for i in range(len(heading_list)):
    cat_dict[i]=heading_list[i]
    
number_of_categories=len(heading_list)

def createHeading(item):
    if item in ["lifestyle", "health", "weather", "foodanddrink", "travel", "kids"]:
        return "Life"
    elif item in ['entertainment', 'tv', 'movies', 'music', 'sports']:
        return "Entertainment"
    elif item in ['news', 'finance', 'middleeast', 'northamerica', 'autos']:
        return "World"
    else:
        return "Video"
    
news_data["Heading"] = news_data["category"].map(createHeading)

# Initializing prior alpha and beta values of beta distributions corresponding to different arms
alpha=np.ones(number_of_categories)
beta=np.ones(number_of_categories)
visited_category=[]

while True:
    high_prob=-1
    for i in range(number_of_categories):
        prob = np.random.beta(alpha[i],beta[i])
        if prob>high_prob:
            high_prob=prob
            category=i
    visited_category.append(category)
    random_news=news_data[news_data['Heading']==cat_dict[category]].sample(n=1)
    print(cat_dict[category])
    print('\t'+random_news['title'][random_news.index[0]])
    abstarct_news = "\t\t" + '\n\t\t'.join(textwrap.wrap(random_news['abstract'][random_news.index[0]], 60, break_long_words=False))
    print(abstarct_news)
    # print('\t\t'+random_news['abstract'][random_news.index[0]])
    did_user_like=int(input('\n\nEnter 1 if u like or 0 if u dislike any other number to exit: '))
    if did_user_like!=0 and did_user_like!=1:
        break
    alpha[category]+=did_user_like
    beta[category]+=(1-did_user_like)
    os.system('cls')