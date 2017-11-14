import pandas as pd
import time


# Make display smaller
pd.options.display.max_rows = 10

unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table('data/ml-1m/users.dat', sep='::', header=None,
                      names=unames)

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('data/ml-1m/ratings.dat', sep='::', header=None,
                        names=rnames)

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('data/ml-1m/movies.dat', sep='::', header=None,
                       names=mnames)

start = time.time()
data = pd.merge(pd.merge(ratings, users), movies)
print data
print "Time to merge:", (time.time() - start)
# TODO: Change this back to mean
mean_ratings = data.pivot_table('rating', index='title', columns='gender',
                                aggfunc='mean')
print "Time to merge and create pivot table:", time.time() - start

ratings_by_title = data.groupby('title').size()
active_titles = ratings_by_title.index[ratings_by_title >= 250]

mean_ratings = mean_ratings.loc[active_titles]
mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_values(by='diff')
rating_std_by_title = data.groupby('title')['rating'].std()
rating_std_by_title = rating_std_by_title.loc[active_titles]
print rating_std_by_title.sort_values(ascending=False)[:10]
end = time.time()

print "Total time taken:", (end - start)
