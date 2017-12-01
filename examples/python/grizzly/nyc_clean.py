#!/usr/bin/env python

"""
This workload is adapted from http://rawgit.com/vberaudi/utwt/master/nyc_taxis.html.
Data: `wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2014-01.csv`
(Newer data may have different format).
"""

import tempfile
import os

import argparse
import time

import pandas as pd
import grizzly.grizzly as gr

# Default CSV
CSV_DEFAULT = "data/yellow_tripdata_2014-01.csv"
# Use Pandas by default.
GRIZZLY_DEFAULT = False

def run(filename, use_grizzly, passes=None, threads=1):
    """
    Loads data, prints the number of initial rows, and wraps the DataFrame
    as a DataFrameWeld if use_grizzly is True.
    Returns the DataFrame and the initial size of the DataFrame.
    """
    ft = pd.read_csv(filename)
    databegin = len(ft)
    print("We have " + str(databegin) + " trips in New York in January")

    # Clean up the columns.
    cols = [c.strip() for c in list(ft.columns.values)]
    ft.columns = cols
    mycols = {'pickup_datetime', 'dropoff_datetime', 'passenger_count',
            'trip_time_in_secs', 'trip_distance', 'pickup_longitude',
            'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'}

    for c in cols:
        if c not in mycols:
            ft.drop(c, 1)

    if use_grizzly:
        ft = gr.DataFrameWeld(ft)

    print "Starting timed run..."
    start = time.time()

    if 'trip_distance' in cols:
        ft = ft[ft['trip_distance'] != 0]

    if 'total_amount' in cols:
        ft = ft[ft['total_amount'] != 0]

    ft = ft[ft['pickup_longitude'] < -73]
    ft = ft[ft['pickup_longitude'] > -76]
    ft = ft[ft['pickup_latitude'] > 39]
    ft = ft[ft['pickup_latitude'] < 43]

    ft = ft[ft['dropoff_longitude'] < -73]
    ft = ft[ft['dropoff_longitude'] > -76]
    ft = ft[ft['dropoff_latitude'] > 39]
    ft = ft[ft['dropoff_latitude'] < 43]

    if 'passenger_count' in cols:
        ft = ft[ft['passenger_count'] <= 3]

    print "Starting timed run..."

    if use_grizzly:
        print "Evaluating Length with Grizzly..."
        length = ft.len(passes=passes, threads=threads)
    else:
        length = len(ft)

    end = time.time()

    print "Length:", length

    print "Removed " + str(databegin - length) + " bad records."

    print "{} seconds".format(end - start)

    return end - start

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean the NYC Taxi Cab dataset.")

    parser.add_argument('-g', '--grizzly', action='store_true')
    parser.add_argument('-f', '--filename', type=str, default=CSV_DEFAULT)
    parser.add_argument('-p', '--passes', type=str, default=None)
    parser.add_argument('-t', '--threads', type=int, default=1)

    args = parser.parse_args()

    print "File={} Grizzly={} Threads={}".format(args.filename, args.grizzly, args.threads)

    if args.passes is not None:
        passes = [p.strip().lower() for p in args.passes.split(",")]
        print "Passes:", passes
    else:
        passes = None

    time = run(args.filename, args.grizzly, passes=passes, threads=str(args.threads))


