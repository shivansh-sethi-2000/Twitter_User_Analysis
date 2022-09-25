from nltk.corpus import stopwords
from deep_translator import GoogleTranslator
import my_tokens
from nltk.stem import WordNetLemmatizer
import csv
import regex as re
import tweepy
import numpy as np
from nltk.tokenize import TweetTokenizer
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import pipeline
from torch.utils.data import DataLoader
import numpy as np
import boto3
import tensorflow_hub as hub
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')
import cv2 
import pickle
import matplotlib.pyplot as plt


auth = tweepy.OAuthHandler(my_tokens.API_KEY, my_tokens.API_SECRET)
auth.set_access_token(my_tokens.ACCESS_TOKEN, my_tokens.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)
client = tweepy.Client(bearer_token=my_tokens.BEARER_TOKEN, consumer_key=my_tokens.API_KEY, consumer_secret=my_tokens.API_SECRET, access_token=my_tokens.ACCESS_TOKEN, access_token_secret=my_tokens.ACCESS_TOKEN_SECRET)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
tt = TweetTokenizer()

model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
model_path = "universal-sentence-encoder_4"
model = hub.load(model_path)
print ("module %s loaded" % model_path)

def get_embeding(input):
  return model(input)

def get_Tweets(filename, query, tweet_field=None, user_field = None, start_date=None, end_date=None):
    
    print(start_date, end_date)
    tweets = tweepy.Paginator(
        client.search_all_tweets, 
        query=query, 
        max_results=500, 
        start_time=start_date, 
        end_time=end_date,
        user_fields=user_field, 
        expansions=['author_id','entities.mentions.username'],
        tweet_fields=tweet_field).flatten()

    tweets_for_csv = []
    for tweet in tweets:
        all_mentions = []
        ex_links = []
        all_hashtags = []
        if tweet.entities:
            if 'urls' in tweet.entities:
                for link in tweet.entities['urls']:
                    ex_links.append(link['expanded_url'])
            if 'mentions' in tweet.entities:
                for mention in tweet.entities['mentions']:
                    all_mentions.append(mention['username'])
            if 'hashtags' in tweet.entities:
                for mention in tweet.entities['hashtags']:
                    all_hashtags.append(mention['tag'])
        tweets_for_csv.append([tweet.id, tweet.author_id, tweet.source,tweet.created_at, tweet.text, tweet.lang, ex_links, all_mentions, all_hashtags])
    outfile = filename + ".csv"
    print("writing to " + outfile)
    with open(outfile, 'w+', encoding='utf-8') as file: 
        writer = csv.writer(file, delimiter=',')
        writer.writerows(tweets_for_csv)

def get_user_info(filename, user_names=None, user_ids=None,user_field=None):

    index = 0
    users_for_csv = []
    if user_names:
        while index < len(user_names):
            uids = user_names[index : min(index+100, len(user_names))]
            users = client.get_users(usernames=uids, user_fields=user_field)
            for user in users.data:
                users_for_csv.append([user.id, user.created_at ,user.name, user.username, user.verified, user.protected, user.public_metrics['followers_count'], user.public_metrics['following_count'], user.public_metrics['tweet_count']])
            index += 100
    elif user_ids:
         while index < len(user_ids):
            uids = user_ids[index : min(index+100, len(user_ids))]
            users = client.get_users(ids=uids, user_fields=user_field)
            for user in users.data:
                users_for_csv.append([user.id, user.created_at ,user.name, user.username, user.verified, user.protected, user.public_metrics['followers_count'], user.public_metrics['following_count'], user.public_metrics['tweet_count']])
            index += 100

    outfile = filename + ".csv"
    print("writing to " + outfile)
    with open(outfile, 'w+', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(users_for_csv)

def get_timeline(filename, ids, end_date=None, tweet_field=None, lim=200):

    user_timeline = []
    for idx in ids:
        tweets = tweepy.Paginator(
            client.get_users_tweets,
            id=idx, 
            max_results=100, 
            tweet_fields=tweet_field,
            expansions=['referenced_tweets.id'],
            exclude=['retweets'],
            end_time = end_date).flatten(limit=lim)
        # for t in tweets:
        for tweet in tweets:
            user_timeline.append([idx, tweet.id, tweet.source, tweet.created_at, tweet.text, tweet.lang,tweet.public_metrics['like_count'], tweet.public_metrics['retweet_count']])
            
    outfile = filename + ".csv"
    print("writing to " + outfile)
    with open(outfile, 'w+', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(user_timeline)


def tweet_lookup(df, ids,media_fields=None, tweet_fields=None, user_fields=None, expansions=None):
    column_links = []
    column_type = []
    index = 0
    while index < len(ids):
        idx = ids[index : min(index+100, len(ids))]
        tweets = client.get_tweets(ids=idx,media_fields=media_fields, user_fields=user_fields, tweet_fields=tweet_fields, expansions=expansions)
        ridx = []
        # print(len(idx), len(tweets.data))
        for tweet in tweets.data:
            ridx.append(tweet.id)
            row_url = []
            row_type = []
            if tweets.includes and tweet.attachments:
                if 'media' in tweets.includes and 'media_keys' in tweet.attachments:
                    for media in tweets.includes['media']:
                        if media.media_key in tweet.attachments['media_keys']:
                            row_url.append(media.url)
                            row_type.append(media.type)
            column_links.append(row_url)
            column_type.append(row_type)
        
        # print(idx)
        # print(ridx)
        # print()
        for i in range(len(idx)):
            if ridx[i] != idx[i]:
                column_links.insert(-1,i)
                column_type.insert(-1,i)
                ridx.insert(i,idx[i])
        index += 100
            
    df['media_link'] = column_links
    df['media_type'] = column_type
    return df

def get_count(lst, word):
    cnt = 0
    for x in [s.lower() for s in lst]:
        if x == word.lower():
            cnt+=1
    return cnt


def get_f_ratio(data, followers, following):
    data['f_ratio'] = data[followers]/ np.clip(data[following], 1e-7, 1e10)
    return data

def clean_text(text):
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', text)
    return text

def clean_stopWords(text):
    return " ".join([w.lower() for w in text.split() if w.lower() not in stop_words and len(w) > 1])

def tokenize(text):
    return tt.tokenize(text)

def lemmatize(text):
    text = [lemmatizer.lemmatize(token) for token in text]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

def get_translate(text, source):
    translation = GoogleTranslator(source=source, target='en').translate(text=text)
    return translation

def get_ppm_count(lst):
    cnt = 0
    l = [s.lower() for s in lst]
    for x in range(len(l)-1):
        if l[x] == 'ppm':
            cnt+=1

    return cnt

def get_sentiment(text):
    return sentiment_task(text)[0]['label']

def imageResizeTrain(image):
    maxD = 1024
    height,width = image.shape
    aspectRatio = width/height
    if aspectRatio < 1:
        newSize = (int(maxD*aspectRatio),maxD)
    else:
        newSize = (maxD,int(maxD/aspectRatio))
    image = cv2.resize(image,newSize)
    return image

sift = cv2.SIFT_create()

def computeSIFT(image):
    return sift.detectAndCompute(image, None)
bf = cv2.BFMatcher()

def calculateMatches(des1,des2):
    matches = bf.knnMatch(des1,des2,k=2)
    topResults1 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            topResults1.append([m])
            
    matches = bf.knnMatch(des2,des1,k=2)
    topResults2 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            topResults2.append([m])
    
    topResults = []
    for match1 in topResults1:
        match1QueryIndex = match1[0].queryIdx
        match1TrainIndex = match1[0].trainIdx

        for match2 in topResults2:
            match2QueryIndex = match2[0].queryIdx
            match2TrainIndex = match2[0].trainIdx

            if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                topResults.append(match1)
    return topResults

def calculateScore(matches,keypoint1,keypoint2):
    return 100 * (matches/min(keypoint1,keypoint2))

def getPlot(image1,image2,keypoint1,keypoint2,matches):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    matchPlot = cv2.drawMatchesKnn(image1,keypoint1,image2,keypoint2,matches,None,[255,255,255],flags=2)
    return matchPlot

def calculateResultsFor(imageA,imageB):
    img1 = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    img1 = imageResizeTrain(img1)
    img2 = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    img2 = imageResizeTrain(img2)
    keypoint1 , descriptor1 = computeSIFT(img1)
    keypoint2, descriptor2 = computeSIFT(img2)
    matches = calculateMatches(descriptor1, descriptor2)
    score = calculateScore(len(matches),len(keypoint1),len(keypoint2))
    if score < 10 or score == 100:
        return 
    fx, ax = plt.subplots(1,2, figsize=(16,10))
    # plt.figure(figsize=(16,10))
    plt.title(str(score)+'%'+' Similar')
    ax[0].imshow(cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB))
    ax[1].imshow(cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB))
    plt.show()

auth = tweepy.OAuth2BearerHandler(my_tokens.BEARER_TOKEN)
api = tweepy.API(auth)

def get_followers(id):
    rows = []
    for page in tweepy.Cursor(api.get_follower_ids, screen_name=id,count=5000).pages(1):
        if len(rows) == 0:
            rows = page
        else:
            rows += page
    return rows

def get_following(id):
    rows = []
    for page in tweepy.Cursor(api.get_friend_ids, screen_name=id,count=5000).pages(1):
        if len(rows) == 0:
            rows = page
        else:
            rows += page
    return rows