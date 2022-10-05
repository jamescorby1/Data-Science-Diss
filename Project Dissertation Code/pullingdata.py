import pandas as pd 

data = pd.read_csv('Reddit-scraped-data.csv')

noncritical = data[data['Target'] == 'Non Critical']

noncritical.to_csv('noncritical.csv')

with open('CVE_tweets.csv', 'r') as f1:
    original = f1.read()

with open('noncritical.csv', 'a') as f2:
    f2.write('\n')
    f2.write(original)

