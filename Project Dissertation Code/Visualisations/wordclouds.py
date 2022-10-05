import matplotlib.pyplot as plt 
from wordcloud import WordCloud, STOPWORDS
import pandas as pd 

#Reddit Visualisation Wordcloud 

reddit_data = pd.read_csv('/Users/jamescorby/Documents/MSc Data Science/Dissertation Files/code/csvdata/Reddit-scraped-data.csv')
reddit_data = reddit_data.drop(reddit_data[reddit_data.Target == 'Non Critical'].index)
reddit_data = reddit_data.applymap(str)
stop_words = set(STOPWORDS)

text = " ".join(title for title in reddit_data.Title)
print(" There are {} words in full body of text".format(len(text)))

reddit_wordcloud = WordCloud(background_color='white', stopwords = stop_words).generate(text)
plt.imshow(reddit_wordcloud, interpolation='bilinear')
plt.title("Reddit Dataset")
plt.axis('off')
plt.show()

#Twitter Visualisation Wordcloud 

twitter_data = pd.read_csv('/Users/jamescorby/Documents/MSc Data Science/Dissertation Files/code/csvdata/Imbalanced_twitter_data.csv')
twitter_data = twitter_data.drop(twitter_data[twitter_data.Target == 'Non Critical'].index)
twitter_data = twitter_data.applymap(str)
stop_words = set(STOPWORDS)

text = " ".join(title for title in twitter_data.Title)
print(" There are {} words in full body of text".format(len(text)))

twitter_wordcloud = WordCloud(background_color='white', stopwords = stop_words).generate(text)
plt.imshow(twitter_wordcloud, interpolation='bilinear')
plt.title("Twitter Dataset")
plt.axis('off')
plt.show()