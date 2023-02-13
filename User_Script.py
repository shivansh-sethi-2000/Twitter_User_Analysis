import streamlit as st
from os.path import exists
import os
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
from deep_translator import GoogleTranslator
import time


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
    maxtags=150)

ids = st_tags(
    label='## Enter User Ids:',
    text='Press enter to add more',
    maxtags=100)
words = st_tags(
    label='## Enter Words for sentiment analysis:',
    text='Press enter to add more',
    maxtags=100)

for i in range(len(ids)):
    ids[i] = int(ids[i])

if not exists('./datasets'):
    os.makedirs('./datasets')

filepath = './datasets/Users_data_'
timeline_path = filepath+'timeline'
authors_path = filepath+'info'

if st.button('Load Authors Data'):

    print('getting Authors data')
    with st.spinner('Getting Authors Data...'):
        fnc.get_user_info(filename=authors_path, user_names=usernames,user_ids=ids, user_field=['created_at','protected','verified', 'public_metrics', 'description', 'profile_image_url'])
    st.success('Done!!')

    authors = pd.read_csv(authors_path+'.csv' ,parse_dates=['created_at'])
    authors['short_date'] = authors['created_at'].apply(lambda x : str(x.year)+'-'+str(x.month))    

    usernames = list(authors.username)
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



start = st.date_input('enter start date', datetime.datetime.now() - datetime.timedelta(1))
end = st.date_input('enter end date', datetime.datetime.now())
limit = st.number_input("Instead of start-end date you can specify the latest number of tweets for timeline", 10)
only_limit = st.checkbox('user limit instead of start/end dates')
if st.button('Timelines Analysis'):

    authors = pd.read_csv(authors_path+'.csv' ,parse_dates=['created_at'])
    authors['short_date'] = authors['created_at'].apply(lambda x : str(x.year)+'-'+str(x.month))
    print('Getting timelines')
    with st.spinner('Getting Timeline Data...'):
        if only_limit:
            fnc.get_timeline(filename= timeline_path, ids= authors.id ,tweet_field=['id','created_at', 'public_metrics', 'source', 'lang'], lim=limit)
        else:
            fnc.get_timeline(filename= timeline_path, ids= authors.id ,end_date = datetime.datetime(end.year, end.month, end.day), start_date=datetime.datetime(start.year, start.month, start.day),tweet_field=['id','created_at', 'public_metrics', 'source', 'lang'], lim=1e9)
    st.success('Done!!')

    
    timelines = pd.read_csv(timeline_path+'.csv', parse_dates=['tweet_time'])

    timelines['created_at'] = timelines['author_id'].apply(lambda x : authors[authors.id == x]['created_at'].values[0])
    timelines['author_username'] = timelines.author_id.apply(lambda x : authors[authors.id == x]['username'].values[0])
    timelines['tweet_hour'] = timelines.tweet_time.dt.hour.astype(int)

    st.download_button(
        label="Download Timelines of Authors as CSV",
        data=convert_df(timelines),
        file_name='authors_timelines.csv',
        mime='text/csv',
    )
    lang = list(GoogleTranslator().get_supported_languages(as_dict=True).values())
    dataX_nottext = timelines[~timelines.lang.isin(lang)]
    dataX_text = timelines[timelines.lang.isin(lang)]
    

    non_text_dic = dict(zip(list(dataX_nottext.author_username.value_counts().index) ,list(dataX_nottext.author_username.value_counts().values)))
    for user in timelines.author_username.value_counts().index:
        if user not in non_text_dic.keys():
            non_text_dic[user] = 0
    
    fig = plt.figure(figsize=(20,4))
    st.header('comparison of total and non text tweets of users')
    plt.xticks(rotation=90)
    sns.barplot(x=timelines.author_username.value_counts().index, y=timelines.author_username.value_counts().values, label='all tweets', color='orange')
    sns.barplot(x=list(non_text_dic.keys()) , y=list(non_text_dic.values()), label='no text tweets', color='limegreen')
    plt.ylabel('count')
    plt.xlabel('author username')
    plt.legend()
    st.pyplot(fig)

    tweets_nontext = {}
    tweets_nontext['username'] = list(non_text_dic.keys())
    tweets_nontext['non text tweets count'] = list(non_text_dic.values())

    st.download_button(
        label="Download authors non text tweets as CSV",
        data=convert_df(pd.DataFrame.from_dict(tweets_nontext)),
        file_name='authors_nontext.csv',
        mime='text/csv',
    )

    # heatmap for users

    g = timelines.groupby(['tweet_hour','author_username'])
    tweet_cnt = g.tweet_id.nunique()
    tweet_cnt = tweet_cnt.reset_index().pivot(index='tweet_hour', columns='author_username', values='tweet_id')
    tweet_cnt.fillna(0,inplace=True)
    tweet_cnt = tweet_cnt.reindex(range(0,24), axis=0, fill_value=0).astype(int)

    fig = plt.figure(figsize=(20,4))
    sns.heatmap(tweet_cnt, cmap='coolwarm')
    st.pyplot(fig)


    dic_9_5 = dict(zip(timelines[(timelines.tweet_time.dt.hour >= 9) & (timelines.tweet_time.dt.hour <= 17)].author_username.value_counts().index, timelines[(timelines.tweet_time.dt.hour >= 9) & (timelines.tweet_time.dt.hour <= 17)].author_username.value_counts().values))
    for user in timelines.author_username.value_counts().index:
        if user not in dic_9_5.keys():
            dic_9_5[user] = 0
    fig = plt.figure(figsize=(20,4))
    plt.xticks(rotation=90)
    st.header('comparison of total and 9-5 tweets of users')
    sns.barplot(x=timelines.author_username.value_counts().index, y=timelines.author_username.value_counts().values, label='all tweets', color='orange')
    sns.barplot(x=list(dic_9_5.keys()) , y=list(dic_9_5.values()), label='9-5 tweets', color='royalblue')
    plt.ylabel('count')
    plt.xlabel('author username')
    plt.legend()
    st.pyplot(fig)

    tweets_9_5 = {}

    tweets_9_5['username'] = list(dic_9_5.keys())
    tweets_9_5['9 to 5 tweets count'] = list(dic_9_5.values())
    
    st.download_button(
        label="Download authors 9-5 tweets as CSV",
        data=convert_df(pd.DataFrame.from_dict(tweets_9_5)),
        file_name='authors_9-5.csv',
        mime='text/csv',
    )

    start = time.time()
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
                dataX_text['text'] =  dataX_text['text'].apply(lambda x : fnc.clean_text(x))
            except:
                print("couldn't remove website/links")
            try:
                dataX_text['text_tokens'] = dataX_text['text'].apply(lambda x : fnc.tokenize(x))
            except:
                print('tokens not created')
            dataX_text['sentiment'] = dataX_text['text'].apply(lambda x : fnc.get_sentiment(x))
        st.success('Time Taken -> '+str(time.time()-start))


    for id,username in zip(authors.id, authors.username):
        try: 
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
        except:
            print('no words found in tweets of :- ' + username)

    fig = plt.figure(figsize=(25,4))
    st.header('counts of neutral, negative and positive tweets for authors')
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
        start1 = time.time()
        with st.spinner('Getting Similarities...'):
            similar_tweets = {'pair of authors ids' : [] , 'tweet 1' : [], 'tweet 2' : [] , 'similarity value' : []}
            nlp = spacy.load("en_core_web_lg")
            ids = dataX_text.tweet_id.values
            corpus = dataX_text.text.values
            logging.set_verbosity(logging.ERROR)
            text_embeddings = fnc.get_embeding(corpus)
            for i in range(len(ids)):
                for j in range(i+1,len(ids)):
                    start3 = time.time()
                    sim = cosine_similarity(np.array(text_embeddings[i]).reshape(1,-1), np.array(text_embeddings[j]).reshape(1,-1))
                    if sim > 0.5 and dataX_text[dataX_text.tweet_id == ids[i]]['author_id'].values[0] != dataX_text[dataX_text.tweet_id == ids[j]]['author_id'].values[0]:
                        author_idxs = []
                        author_idxs.append(dataX_text[dataX_text.tweet_id == ids[i]]['author_id'].values[0])
                        author_idxs.append(dataX_text[dataX_text.tweet_id == ids[j]]['author_id'].values[0])
                        for k in range(len(author_idxs)):
                            author_idxs[k] = authors[authors.id == author_idxs[k]]['username'].values[0]
                        similar_tweets['pair of authors ids'].append(author_idxs)
                        similar_tweets['tweet 1'].append(dataX_text[dataX_text.tweet_id == ids[i]]['text'].values[0])
                        similar_tweets['tweet 2'].append(dataX_text[dataX_text.tweet_id == ids[j]]['text'].values[0])
                        similar_tweets['similarity value'].append(sim[0][0])
                    
        st.success('Overall Time Taken-> ' + str(time.time()-start1))
        if len(similar_tweets) > 1:
            similar_Tweets_df = pd.DataFrame.from_dict(similar_tweets)
            st.download_button(
                label="Download Similar Tweets as CSV",
                data=convert_df(similar_Tweets_df),
                file_name='authors_similarities.csv',
                mime='text/csv',
            )