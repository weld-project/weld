#!/usr/bin/python

# The usual preamble
import numpy as np
import pandas as pd
import time

# Get data (NYC 311 service request dataset) and start cleanup
data = pd.read_csv('data/us_cities_states_counties.csv', delimiter='|')
data.dropna(inplace=True)
print "Done reading input file..."

start = time.time()

# Get all city information with total population greater than 500,000
data_big_cities = data[data["Total population"] > 500000]
data_big_cities_new_df = data_big_cities[["State short"]]

# Compute "crime index" proportional to
# exp((Total population + 2*(Total adult population) - 2000*(Number of
# robberies)) / 100000)
data_big_cities_stats = data_big_cities[
    ["Total population", "Total adult population", "Number of robberies"]].values
predictions = np.exp(np.dot(data_big_cities_stats,
                            np.array([1.0, 2.0, -2000.0])) / 100000.0)
predictions = predictions / predictions.sum()
data_big_cities_new_df["Crime index"] = predictions

# Aggregate "crime index" scores by state
data_big_cities_grouped_df = data_big_cities_new_df.groupby(
    "State short").sum()
print sorted(["%.4f" % ele for ele in data_big_cities_grouped_df["Crime index"]])
end = time.time()

print "Total end-to-end time: %.2f" % (end - start)
