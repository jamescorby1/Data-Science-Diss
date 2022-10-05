import praw 
import pandas as pd 

reddit = praw.Reddit(client_id = "V99_PBGAbDocGA", client_secret = "GNNER7TrsFAkWVWIaZ0nFM5-On2ucA", 
    user_agent = "CTI Data", user_name = "jamescti", password = "Hernandez8") 

subreddit_list = ['HowToHack', 'hacking', 'onions', 'netsec', 'Hacking_Tutorials', 'blackhat','zeroday', 'bugbounty', 'Scams']

title_list = []
target = []

for sub in subreddit_list:
    subreddit = reddit.subreddit(sub)
    hot_posts = subreddit.hot(limit = 225)

    for post in hot_posts:
        title_list.append(post.title)
        target.append("Critical")

    print(sub, "Completed: ", end=" ")
    print("Total", len(title_list), " critical posts have been scraped ")

reddit_critical_df = pd.DataFrame({"Title": title_list, "Target": target})

reddit_critical_df.to_csv("Reddit-critical-data.csv")