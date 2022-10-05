import praw
import pandas as pd 

reddit = praw.Reddit(client_id = "V99_PBGAbDocGA", client_secret = "GNNER7TrsFAkWVWIaZ0nFM5-On2ucA", 
    user_agent = "CTI Data", user_name = "jamescti", password = "Hernandez8")

subreddit_list = ['formula1', 'soccer', 'nba', 'gtaonline']

title_list = []
target = []

for sub in subreddit_list:
    subreddit = reddit.subreddit(sub)
    hot_posts = subreddit.hot(limit = 500)

    for post in hot_posts:
        title_list.append(post.title)
        target.append("Non Critical")

    print(sub, "Completed: ", end=" ")
    print("Total", len(title_list), " non critical posts have been scraped ")

reddit_noncritical_df = pd.DataFrame({"Title": title_list, "Target": target})

reddit_noncritical_df.to_csv("Reddit-noncritical-data.csv")