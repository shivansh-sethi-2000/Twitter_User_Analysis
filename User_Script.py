from multiprocessing import connection
import streamlit as st
from os.path import exists
import functions as fnc
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from wordcloud import WordCloud
import functions as fnc
from absl import logging
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from streamlit_tags import st_tags
import networkx as nx
from pyvis.network import Network
from stvis import pv_static
from deep_translator import GoogleTranslator


def convert_df(df):
     return df.to_csv()

def get_count(lst, word):
    cnt = 0
    for x in lst:
        if word.lower() in x:
            cnt+=1
    return cnt

st.set_page_config(layout="wide")
st.title('Twitter Users Analysis')
usernames = st_tags(
    label='## Enter Usernames without @:',
    text='Press enter to add more',
    maxtags=100)
words = st_tags(
    label='## Enter Words for sentiment analysis:',
    text='Press enter to add more',
    maxtags=100)

filepath = './datasets/Users_data_'
timeline_path = filepath+'timeline'
authors_path = filepath+'info'

if st.button('Load Authors Data'):
    
    print('getting Authors data')
    with st.spinner('Getting Authors Data...'):
        fnc.get_user_info(filename=authors_path, user_names=usernames, user_field=['created_at','protected','verified', 'public_metrics'])
    st.success('Done!!')

    authors = pd.read_csv(authors_path+'.csv', names=['id','created_at', 'name', 'username', 'verified', 'protected', 'followers', 'following', 'tweets_count'] ,parse_dates=['created_at'])
    authors['short_date'] = authors['created_at'].apply(lambda x : str(x.year)+'-'+str(x.month))    

    authors.id = authors.id.astype(str)
    st.download_button(
        label="Download Filtered Authors Data as CSV",
        data=convert_df(authors),
        file_name='auhtors_info.csv',
        mime='text/csv',
    )

    fig = plt.figure(figsize=(10,4))
    plt.xticks(rotation=90)
    sns.scatterplot(data = authors, x='username', y='short_date')
    plt.ylabel('creation date')
    plt.xlabel('author_id')
    st.pyplot(fig)

    fig = plt.figure(figsize=(10,4))
    plt.xticks(rotation=90)
    sns.barplot(data=authors, x='username', y='tweets_count')
    plt.ylabel('Total Number of Tweets')
    plt.xlabel('author_id')
    st.pyplot(fig)

    fig = plt.figure(figsize=(10,4))
    plt.xticks(rotation=90)
    sns.barplot(data=authors, x='username', y='followers')
    plt.ylabel('Total Number of followers')
    plt.xlabel('author_id')
    st.pyplot(fig)

    fig = plt.figure(figsize=(10,4))
    plt.xticks(rotation=90)
    sns.barplot(data=authors, x='username', y='following')
    plt.ylabel('Total Number user following')
    plt.xlabel('author_id')
    st.pyplot(fig)

    

    for idx in usernames:
        followers_filepath = './datasets/'+idx+'_followers'
        following_filepath = './datasets/'+idx+'_following'
        followers = []
        followings = []
        if not exists(followers_filepath+'.csv'):
            followers = fnc.get_followers(idx)
            fnc.get_user_info(filename=followers_filepath, user_ids=followers, user_field=['created_at','protected','verified', 'public_metrics'])

        if not exists(following_filepath+'.csv'):
            followings =fnc.get_following(idx)
            fnc.get_user_info(filename=following_filepath, user_ids=followings, user_field=['created_at','protected','verified', 'public_metrics'])
        
        

    for idx in usernames:
        # id_followers = pd.DataFrame(followers[idx])
        # id_following = pd.DataFrame(following[idx])
        id_following = pd.read_csv('./datasets/'+idx+'_following.csv', names=['following_id','created_at', 'following_name', 'following_username', 'verified', 'protected', 'followers', 'following', 'tweets_count'] ,parse_dates=['created_at'])
        id_followers = pd.read_csv('./datasets/'+idx+'_followers.csv', names=['follower_id','created_at', 'follower_name', 'follower_username', 'verified', 'protected', 'followers', 'following', 'tweets_count'] ,parse_dates=['created_at'])
        st.download_button(
            label="Download Followers of "+idx,
            data=convert_df(id_followers),
            file_name=idx+'_followers.csv',
            mime='text/csv',
        )
        st.download_button(
            label="Download Following of "+idx,
            data=convert_df(id_following),
            file_name=idx+'_following.csv',
            mime='text/csv',
        )

    # Creating Network Graph
    G = nx.Graph()
    cnt = 0

    for idx in usernames:
        G.add_node(idx,color='red', title=idx)

    followers = {}
    following = {}
    for idx in usernames:
        id_following = pd.read_csv('./datasets/'+idx+'_following.csv', names=['following_id','created_at', 'following_name', 'following_username', 'verified', 'protected', 'followers', 'following', 'tweets_count'] ,parse_dates=['created_at'])
        id_followers = pd.read_csv('./datasets/'+idx+'_followers.csv', names=['follower_id','created_at', 'follower_name', 'follower_username', 'verified', 'protected', 'followers', 'following', 'tweets_count'] ,parse_dates=['created_at'])
        followers[idx] = id_followers
        following[idx] = id_following

    for idx in usernames:
        for node in following[idx].following_username:
            lst = []
            for idx2 in usernames:
                if node in followers[idx2].follower_username:
                    lst.append(idx2)
                elif node in following[idx2].following_username:
                    lst.append(idx2)
            if len(lst) > 1:
                if not G.has_node(str(node)):
                    G.add_node(str(node),color='blue', title=str(node))
                for source in lst:
                    G.add_edge(source, str(node))

        for node in followers[idx].follower_username:
            lst = []
            for idx2 in usernames:
                if node in followers[idx2].follower_username:
                    lst.append(idx2)
                elif node in following[idx2].following_username:
                    lst.append(idx2)
            if len(lst) > 1:
                if not G.has_node(str(node)):
                    G.add_node(str(node),color='blue', title=str(node))
                for source in lst:
                    G.add_edge(source, str(node))

    st.header('Common Connections of Users')
    net = Network("1000px", "2000px",notebook=True, font_color='#10000000')
    net.from_nx(G)
    pv_static(net)



limit = st.number_input('Enter Numbr Of Tweets for Analysis', min_value=1, format='%i')
if st.button('Timelines Analysis'):

    authors = pd.read_csv(authors_path+'.csv', names=['id','created_at', 'name', 'username', 'verified', 'protected', 'followers', 'following', 'tweets_count'] ,parse_dates=['created_at'])
    authors['short_date'] = authors['created_at'].apply(lambda x : str(x.year)+'-'+str(x.month))
    print('Getting timelines')
    with st.spinner('Getting Timeline Data...'):
        fnc.get_timeline(timeline_path, authors.id ,datetime.datetime.now(),['id','created_at', 'public_metrics', 'source', 'lang'], lim=limit)
    st.success('Done!!')

    timelines = pd.read_csv(timeline_path+'.csv', names = ['author_id', 'tweet_id','source', 'tweet_time', 'text', 'lang', 'likes', 'retwets'], parse_dates=['tweet_time'])
    timelines['created_at'] = timelines['author_id'].apply(lambda x : authors[authors['id'] == x]['created_at'].values[0])
    timelines['author_username'] = timelines.author_id.apply(lambda x : authors[authors.id == x]['username'].values[0])

    st.download_button(
        label="Download Timelines of Authors as CSV",
        data=convert_df(timelines),
        file_name='authors_timelines.csv',
        mime='text/csv',
    )
    lang = list(GoogleTranslator().get_supported_languages(as_dict=True).values())
    dataX_nottext = timelines[~timelines.lang.isin(lang)]
    dataX_text = timelines[timelines.lang.isin(lang)]
    

    fig = plt.figure(figsize=(20,4))
    st.header('comparison of total and non text tweets of users')
    plt.xticks(rotation=90)
    sns.barplot(x=timelines.author_username.value_counts().index, y=timelines.author_username.value_counts().values, label='all tweets', color='orange')
    sns.barplot(x=dataX_nottext.author_username.value_counts().index , y=dataX_nottext.author_username.value_counts().values, label='no text tweets', color='limegreen')
    plt.ylabel('count')
    plt.xlabel('author username')
    plt.legend()
    st.pyplot(fig)

    st.download_button(
        label="Download authors non text tweets as CSV",
        data=convert_df(dataX_nottext),
        file_name='authors_nontext.csv',
        mime='text/csv',
    )

    fig = plt.figure(figsize=(20,4))
    plt.xticks(rotation=90)
    st.header('comparison of total and 9-5 tweets of users')
    sns.barplot(x=timelines.author_username.value_counts().index, y=timelines.author_username.value_counts().values, label='all tweets', color='orange')
    sns.barplot(x=timelines[(timelines.tweet_time.dt.hour >= 9) & (timelines.tweet_time.dt.hour <= 17)].author_username.value_counts().index , y=timelines[(timelines.tweet_time.dt.hour >= 9) & (timelines.tweet_time.dt.hour <= 17)].author_username.value_counts().values, label='9-5 tweets', color='royalblue')
    plt.ylabel('count')
    plt.xlabel('author username')
    plt.legend()
    st.pyplot(fig)

    st.download_button(
        label="Download authors 9-5 tweets as CSV",
        data=convert_df(timelines[(timelines.tweet_time.dt.hour >= 9) & (timelines.tweet_time.dt.hour <= 17)]),
        file_name='authors_9-5.csv',
        mime='text/csv',
    )

    if 'sentiment' not in dataX_text.columns:
        with st.spinner('Text Pre Processing...'):
            translated = []
            for x,lang in zip(dataX_text.text, dataX_text.lang):
                if lang != 'en':
                    translated.append(fnc.get_translate(x,lang))
                else :
                    translated.append(x)
            dataX_text['text'] = translated
            try:
                dataX_text['text'] = dataX_text['text'].apply(lambda x : fnc.clean_text(x))
            except:
                print('data not cleaned')
            try:
                dataX_text['text'] = dataX_text['text'].apply(lambda x : fnc.clean_stopWords(x))
            except:
                print('stopwords not removed')
            try:
                dataX_text['text_tokens'] = dataX_text['text'].apply(lambda x : fnc.tokenize(x))
            except:
                print('tokens not created')
            try:
                dataX_text['text_lemmatized'] = dataX_text['text_tokens'].apply(lambda x : fnc.lemmatize(x))
            except:
                print('not lemmatized')
            if 'text_lemmatized' in dataX_text.columns:
                dataX_text['sentiment'] = dataX_text['text_lemmatized'].apply(lambda x : fnc.get_sentiment(x))
            else :
                dataX_text['sentiment'] = dataX_text['text'].apply(lambda x : fnc.get_sentiment(x))
        st.success('Done!!')

    
    for id,username in zip(authors.id, authors.username):
        st.subheader('Frequent Words Used by author - '+username)
        fig = plt.figure(figsize=(15,5))
        negative = dataX_text[dataX_text.author_id == id].text_tokens
        negative = [" ".join(negative.values[i]) for i in range(len(negative))]
        negative = [" ".join(negative)][0]
        wc = WordCloud(min_font_size=3,max_words=200,width=1600,height=720, colormap = 'Set1', background_color='black').generate(negative)
        plt.imshow(wc,interpolation='bilinear')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        st.pyplot(fig)

    fig = plt.figure(figsize=(25,4))
    st.header('counts of neutral, negtive and positive tweets for authors')
    dataX_text.author_id = dataX_text.author_id.astype(str)
    plt.xticks(rotation=90)
    sns.histplot(data=dataX_text, x='author_username', hue=dataX_text.sentiment, multiple='dodge', palette='bright', discrete=True, shrink=.9)
    st.pyplot(fig)
    dataX_text.author_id = dataX_text.author_id.astype(int)

    author_sentiments = {'username' : [], 'positive tweets' : [], 'neutral tweets' : [], 'negative tweets': []}

    for idx in dataX_text.author_id.unique():
        author_sentiments['username'].append(authors[authors.id == idx]['username'].values[0])
        for ind, va in zip(dataX_text[dataX_text.author_id == idx].sentiment.value_counts().index, dataX_text[dataX_text.author_id == idx].sentiment.value_counts().values):
            if ind == 'Positive':
                author_sentiments['positive tweets'].append(va)
            elif ind == 'Negative':
                author_sentiments['negative tweets'].append(va)
            else:
                author_sentiments['neutral tweets'].append(va)

        if len(author_sentiments['positive tweets']) < len(author_sentiments['username']):
            author_sentiments['positive tweets'].append(0)
        if len(author_sentiments['negative tweets']) < len(author_sentiments['username']):
            author_sentiments['negative tweets'].append(0)
        if len(author_sentiments['neutral tweets']) < len(author_sentiments['username']):
            author_sentiments['neutral tweets'].append(0)

    sentiment_df = pd.DataFrame(author_sentiments)
    st.download_button(
        label="Download All Sentiment Counts of Authors as CSV",
        data=convert_df(sentiment_df),
        file_name='authors_sentiments.csv',
        mime='text/csv',
    )

    for word in words:
        dataX_text['word_present'] = dataX_text.text_tokens.apply(lambda x : get_count(x,word))
        sent = {}
        cnt = 0
        for idx,x,sen in zip(dataX_text.author_username, dataX_text.word_present, dataX_text.sentiment):
            if sen == 'Positive' and x == 1:
                sen = 1
            elif sen == 'Negative' and x == 1:
                sen = -1
            else:
                sen = 0
            if idx in sent:
                sent[idx] += sen
            else:
                sent[idx] = sen
            cnt+=1
        
        for x in sent:
            sent[x] /= cnt
        fig = plt.figure(figsize=(20,5))
        st.header('average tweet sentiment of authors who have used the word ' +word)
        plt.xticks(rotation=90)
        plt.xlabel('author_username')
        plt.ylabel('average sentiment of tweets')
        sns.barplot(x=list(sent.keys()), y=list(sent.values()))
        st.pyplot(fig)

    if len(usernames) > 1:
        similar_tweets = {'pair of authors ids' : [] , 'tweet 1' : [], 'tweet 2' : [] , 'similarity value' : []}
        nlp = spacy.load("en_core_web_lg")
        ids = dataX_text.tweet_id.values
        corpus = dataX_text.text.values
        logging.set_verbosity(logging.ERROR)
        text_embeddings = fnc.get_embeding(corpus)
        for i in range(len(ids)):
            for j in range(i+1,len(ids)):
                sim = cosine_similarity(np.array(text_embeddings[i]).reshape(1,-1), np.array(text_embeddings[j]).reshape(1,-1))
                if sim > 0.7 and dataX_text[dataX_text.tweet_id == ids[i]]['author_id'].values[0] != dataX_text[dataX_text.tweet_id == ids[j]]['author_id'].values[0]:
                    author_idxs = []
                    author_idxs.append(dataX_text[dataX_text.tweet_id == ids[i]]['author_id'].values[0])
                    author_idxs.append(dataX_text[dataX_text.tweet_id == ids[j]]['author_id'].values[0])
                    author_idxs.sort()
                    for i in range(len(author_idxs)):
                        author_idxs[i] = authors[authors.id == author_idxs[i]]['username'].values[0]
                    author_idxs = tuple(author_idxs)
                    similar_tweets['pair of authors ids'].append(author_idxs)
                    similar_tweets['tweet 1'].append(dataX_text[dataX_text.tweet_id == ids[i]]['text'].values[0])
                    similar_tweets['tweet 2'].append(dataX_text[dataX_text.tweet_id == ids[j]]['text'].values[0])
                    similar_tweets['similarity value'].append(sim)

        if len(similar_tweets) > 1:
            similar_Tweets_df = pd.DataFrame.from_dict(similar_tweets)
            st.download_button(
                label="Download Similar Tweets as CSV",
                data=convert_df(similar_Tweets_df),
                file_name='authors_similarities.csv',
                mime='text/csv',
            )