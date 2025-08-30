import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import textwrap
from IPython.display import clear_output
import os

userId=np.load('userId.npy')
userEmbedding=np.load('user_embedding.npy')
newsId=np.load('itemId.npy')
newsEmbedding=np.load('news_embeddings_numpy.npy')
news_data=pd.read_csv('preprocessed_news.csv')
category_embeddings=np.load('category_embeddings.npy')
subcategory_embeddings=np.load('subcategory_embeddings.npy')
title_embeddings=np.load('title_embeddings.npy')
abstract_embeddings=np.load('abstract_embeddings.npy')

def L2(u,a):
    return np.linalg.norm(u-a)

dict={}
for i in range(len(newsId)):
    dict[newsId[i]]=newsEmbedding[i]

def findUserEmbedding(user_read_articles):
    new_user_embedding=np.zeros(384)
    for i in user_read_articles:
        new_user_embedding+=dict[i]
    new_user_embedding/=len(user_read_articles)
    return new_user_embedding

# Finding articles closest to a given user
def updateArticles(new_user_embedding):
    distances=[]
    for i in range(len(newsEmbedding)):
        distances.append((L2(new_user_embedding,newsEmbedding[i]),i))
    distances=sorted(distances)
    return distances

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

heading_list = ["Life", 'Entertainment', "World", 'Video']

def displayMainPage(head_wise_news):
    for heading in heading_list:
        temp = head_wise_news[head_wise_news['Heading'] == heading]
        print(heading)
        for i in temp.index.values:
            details = str(i) + '\t' + temp['category'][i] + '\t' + temp['subcategory'][i] + '\n\t\t' + temp['title'][i]
            print(details)
            abstractText = "\t\t\t" + temp["abstract"][i]
            print('\n\t\t\t'.join(textwrap.wrap(abstractText, 60, break_long_words=False)))
            print()

def displayRecommendations(distances, k):
    for i in distances[:k]:
        details = str(i[1]) + '\t' + news_data['category'][i[1]] + '\t' + news_data['subcategory'][i[1]] + '\n\t\t' + news_data['title'][i[1]]
        print(details)
        abstractText = "\t\t\t" + news_data['abstract'][i[1]]
        print('\n\t\t\t'.join(textwrap.wrap(abstractText, 60, break_long_words=False)))
        print()


# Enter date
count = 0
itemsRead = []
dynamic_user_embedding = 0
while True:
    # Display category wise articles according to user state
    count += 1
    choice = 0
    if count >= 3:
        # Give user the choice to select main page or recommendations
        print('Choose 0 for Main Page, 1 for Recommendations, -1 to exit')
        choice=int(input(""))
        if choice == -1:
            break

    # Give user a choice to select article
    if choice == 0:
        # Show main page
        explore_news = news_data.groupby('Heading').sample(n=5)
        explore_news = explore_news.reset_index()
        displayMainPage(explore_news)
        articlesRead=input('Enter article number you want to read:')
        itemsRead.extend([explore_news['itemId'][int(s)] for s in articlesRead.split()])
    else:
        # Show recommendations
        displayRecommendations(distances, 5)
        articlesRead=input('Enter article number you want to read:')
        itemsRead.extend([news_data['itemId'][int(s)] for s in articlesRead.split()])

    # Update user state
    dynamic_user_embedding = findUserEmbedding(itemsRead)
    distances=updateArticles(dynamic_user_embedding)
    # clear_output(wait=True)
    os.system('cls')
os.system('cls')