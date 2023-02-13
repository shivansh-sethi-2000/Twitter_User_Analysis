from nltk.corpus import stopwords
from deep_translator import GoogleTranslator
import my_tokens
from nltk.stem import WordNetLemmatizer
import csv
import regex as re
import tweepy
import numpy as np
from nltk.tokenize import TweetTokenizer
from transformers import pipeline
import numpy as np
import tensorflow_hub as hub
import warnings
from twitterUsernameviaUserID import getHandles as gH
import snscrape.modules.twitter as sntwitter
warnings.filterwarnings('ignore')
import cv2 
import time


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
    
    columns = ['id', 'author_id', 'source', 'created_at', 'text', 'lang', 'ex_links', 'mentions', 'hashtags']
    start = time.time()
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

    print("got tweets time taken :- " + str(time.time() - start))
    start = time.time()
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
        writer.writerow(columns)
        writer.writerows(tweets_for_csv)
    print('time taken to write create dataset search :- '+ str(time.time() - start))


def get_username_from_id(user_ids):
    usernames = []
    users = client.get_users(ids=user_ids, user_fields=None)
    for user in users.data:
        usernames.append(user.username)
    
    return usernames


def get_user_info(filename, user_names=None, user_ids=None,user_field=None):

    index = 0
    users_for_csv = []
    columns = ['id','created_at', 'name', 'username', 'verified', 'protected', 'followers', 'following', 'tweets_count', 'description', 'profile_image_url']
    if user_names:
        while index < len(user_names):
            uids = user_names[index : min(index+100, len(user_names))]
            users = client.get_users(usernames=uids, user_fields=user_field)
            for user in users.data:
                users_for_csv.append([user.id, user.created_at ,user.name, user.username, user.verified, user.protected, user.public_metrics['followers_count'], user.public_metrics['following_count'], user.public_metrics['tweet_count'], user.description, user.profile_image_url])
            index += 100
    index = 0
    if user_ids:
         while index < len(user_ids):
            uids = user_ids[index : min(index+100, len(user_ids))]
            users = client.get_users(ids=uids, user_fields=user_field)
            for user in users.data:
                users_for_csv.append([user.id, user.created_at ,user.name, user.username, user.verified, user.protected, user.public_metrics['followers_count'], user.public_metrics['following_count'], user.public_metrics['tweet_count'], user.description])
            index += 100

    outfile = filename + ".csv"
    print("writing to " + outfile)
    with open(outfile, 'w+', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(columns)
        writer.writerows(users_for_csv)

def get_timeline(filename, ids, start_date=None, end_date=None, tweet_field=None, lim=200):

    columns = ['author_id', 'tweet_id','source', 'tweet_time', 'text', 'lang', 'likes', 'retwets']
    start = time.time()
    user_timeline = []
    for idx in ids:
        tweets = tweepy.Paginator(
            client.get_users_tweets,
            id=idx, 
            max_results=100, 
            tweet_fields=tweet_field,
            expansions=['referenced_tweets.id'],
            exclude=['retweets'],
            start_time=start_date,
            end_time = end_date).flatten(limit=lim)
        # for t in tweets:
        print('got timeline time taken :- '+str(time.time() - start))
        start = time.time()
        for tweet in tweets:
            user_timeline.append([idx, tweet.id, tweet.source, tweet.created_at, tweet.text, tweet.lang,tweet.public_metrics['like_count'], tweet.public_metrics['retweet_count']])
            
    outfile = filename + ".csv"
    print("writing to " + outfile)
    with open(outfile, 'w+', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(columns)
        writer.writerows(user_timeline)
    print("dataset ready of timelines time taken :- "+str(time.time() - start))

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
    text = text.lower()
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

def get_translate(text, source=None):
    translation = GoogleTranslator(target='en').translate(text=text)
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
    if descriptor1 is not None and descriptor2 is not None:
        matches = calculateMatches(descriptor1, descriptor2)
        score = calculateScore(len(matches),len(keypoint1),len(keypoint2))
        return score
    elif descriptor1 is None and descriptor2 is None:
        return 100
    else :
        return 0
    # fx, ax = plt.subplots(1,2, figsize=(16,10))
    # # plt.figure(figsize=(16,10))
    # plt.title(str(score)+'%'+' Similar')
    # ax[0].imshow(cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB))
    # ax[1].imshow(cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB))
    # plt.show()

auth = tweepy.OAuth2BearerHandler(my_tokens.BEARER_TOKEN)
api = tweepy.API(auth)

def get_followers(id, followers_path):
    rows = []
    for item in tweepy.Cursor(api.get_follower_ids, screen_name=id,count=5000).items(25000):
        # user = sntwitter.TwitterUserScraper(user = item)._get_entity()
        # rows.append([user.username, user.id, user.displayname, user.rawDescription, user.verified, user.created, user.followersCount, user.friendsCount, user.location, user.protected, user.profileImageUrl])
        rows.append(item)
    
    print("writing to " + followers_path)
    with open(followers_path, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(rows)

def get_following(id, following_path):
    # columns = ['username', 'userid', 'name', 'description', 'verified', 'created_at', 'followers', 'followings','location' , 'protected', 'profile_image_url']
    rows = []
    followings = tweepy.Cursor(api.get_friend_ids, screen_name=id,count=5000).items(25000)
    for item in followings:
        # user = sntwitter.TwitterUserScraper(user = item)._get_entity()
        # rows.append([user.username, user.id, user.displayname, user.rawDescription, user.verified, user.created, user.followersCount, user.friendsCount, user.location, user.protected, user.profileImageUrl])
        rows.append(item)
    
    print("writing to " + following_path)
    with open(following_path, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(rows)