# import pandas as pd
import numpy as np
from math import *
import time
from weldnumpy import weldarray
import weldnumpy as wn
import argparse
import csv
import ast
import json
import pandas as pd
import os

# for group 
import grizzly.grizzly as gr
from grizzly.lazy_op import LazyOpResult

# ### Read in the data

# In[52]:

df = pd.read_csv('data', encoding='cp1252')
LATS_NAME = 'lats'
LONS_NAME = 'lons'

# Haversine definition
def haversine(lat1, lon1, lat2, lon2):
    
    miles_constant = 3959.0
    start2 = time.time()
    
    # tracking number of operators as a comment above each line
    
    # 2*2 = 4 (div, multiply in impl) --> just add 3 on final count
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2]) 
    print("deg2rad took: ", time.time() - start2)
    # 2
    dlat = lat2 - lat1 
    # 3
    dlon = lon2 - lon1 
    #a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    # 4
    a0 = np.sin(dlat/2.0)  
    # 5
    a1 = np.sin(dlon/2.0)
    # 11
    a = a0*a0 + np.cos(lat1) * np.cos(lat2) * a1*a1
    # 14
    c = 2.0 * np.arcsin(np.sqrt(a)) 
    # 15 + 3 = 18
    mi = miles_constant * c
    return mi

def write_to_file(arr, name):
    # f = open(name, 'w')
    arr.tofile(name)

def gen_data(lat, lon, scale=10):
    '''
    Generates the array replicated X times.
    '''
    np.random.seed(1)
    new_lat = []
    new_lon = []
    for i in xrange(len(lat)*scale):
        index = i % len(lat) 
        l1 = lat[index]
        l2 = lon[index]
        new_lat.append(l1 + np.random.rand())
        new_lon.append(l2 + np.random.rand())

    new_lat = np.array(new_lat)
    new_lon = np.array(new_lon)
    write_to_file(new_lat, LATS_NAME)
    write_to_file(new_lon, LONS_NAME)

    return new_lat, new_lon

def gen_data2(lat, lon, scale=10):
    '''
    generates 2 pairs of arrays so can pass two arrays to havesrine. 
    '''
    pass

def generate_lazy_op_list(arrays):
    ret = []
    for a in arrays:
        lazy_arr = LazyOpResult(a.weldobj, a._weld_type, 1)
        ret.append(lazy_arr)
    return ret

def compare(R, R2):
    
    if isinstance(R2, weldarray):
        R2 = R2.evaluate()

    mistakes = 0
    R = R.flatten()
    R2 = R2.view(np.ndarray).flatten()
    
    assert R.dtype == R2.dtype, 'dtypes must match!'
    
    assert np.allclose(R, R2)
    # assert np.array_equal(R, R2)

def print_args(args):
    d = vars(args)
    print('params: ', str(d))

def read_data():
    start = time.time()
    new_lat = np.fromfile(LATS_NAME)
    new_lon = np.fromfile(LONS_NAME)
    end = time.time()
    print('reading in data took ', end-start)
    return new_lat, new_lon

def run_haversine_with_scalar(args):
    orig_lat = df['latitude'].values
    orig_lon = df['longitude'].values

    if args.use_numpy:
        ########### Numpy stuff ############
        if not os.path.isfile(LATS_NAME):
            lat, lon = gen_data(orig_lat, orig_lon, scale=args.scale)
        else:
            lat, lon = read_data()
        print('num rows in lattitudes: ', len(lat))

        start = time.time()
        dist1 = haversine(40.671, -73.985, lat, lon)
        end = time.time()
        print('****************************')
        print('numpy took {} seconds'.format(end-start))
        print('****************************')
	del(lat) 
	del(lon)
    else:
        print('Not running numpy')
    # just in case let us free memory

    if args.use_weld:
        ####### Weld stuff ############
        if not os.path.isfile(LATS_NAME):
            lat2, lon2 = gen_data(orig_lat, orig_lon, scale=args.scale)
        else:
            lat2, lon2 = read_data()
        print('num rows in lattitudes: ', len(lat2))
        lat2 = weldarray(lat2)
        lon2 = weldarray(lon2)
        start = time.time()
        dist2 = haversine(40.671, -73.985, lat2, lon2) 
        if args.use_group:
            lazy_ops = generate_lazy_op_list([dist2])
            dist2 = gr.group(lazy_ops).evaluate(True, passes=wn.CUR_PASSES)[0]
        else:
            dist2 = dist2.evaluate()
	
	#print(np.sum(dist2))
        end = time.time()
        print('****************************')
        print('weld took {} seconds'.format(end-start))
        print('****************************')
        print('END')
    else:
        print('Not running weld')
    
    if args.use_numpy and args.use_weld:
        compare(dist1, dist2)

parser = argparse.ArgumentParser(
    description="give num_els of arrays used for nbody"
)
parser.add_argument('-s', "--scale", type=int, required=True,
                    help=("how much to scale up the orig dataset? Used so we",
                    "can run it on larger data sizes"))
parser.add_argument('-g', "--use_group", type=int, default=0,
                    help="use group or not")
parser.add_argument('-p', "--remove_pass", type=str, 
                    default="whatever_string", help="will remove the pass containing this str")
parser.add_argument('-numpy', "--use_numpy", type=int, required=False, default=0,
                    help="use numpy or not in this run")
parser.add_argument('-weld', "--use_weld", type=int, required=False, default=0,
                    help="use weld or not in this run")

args = parser.parse_args()

LATS_NAME = LATS_NAME + str(args.scale)
LONS_NAME = LONS_NAME + str(args.scale)
run_haversine_with_scalar(args)
