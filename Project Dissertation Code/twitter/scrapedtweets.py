import twint
import pandas as pd
import re

t = twint.Config() 

t.Username = 'CVEnew'
t.Pandas = True
t.Limit = 2000
twint.run.Search(t)

tweets = twint.storage.panda.Tweets_df

Title_list = []
Target_list = []

for tweet in tweets['tweet']:
    Title_list.append(tweet)
    Target_list.append('Critical')

CVE_df = pd.DataFrame({"Title": Title_list, "Target": Target_list})

CVE_df['Title'] = CVE_df['Title'].replace(r'http\S+', '', regex=True)

print(CVE_df['Title'].head(10))

CVE_df.to_csv('CVE_tweets.csv')