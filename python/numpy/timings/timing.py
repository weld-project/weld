import numpy as np
import sys
sys.path.append('..')
import weldnumpy as wn

import py.test
import random
import timeit
import time
import sys
import argparse
import scipy.special as ss
'''
TODO: Adjust the number of arrays being added as an arg etc.
'''

invsqrt2 = 1.0 #0.707

def str2bool(v):
  '''
  used to create special type for argparse
  '''
  return v.lower() in ('yes', 'true', 't', '1')

UNARY_OPS = {}
UNARY_OPS['sqrt'] = np.sqrt
UNARY_OPS['exp'] = np.exp
UNARY_OPS['log'] = np.log

parser = argparse.ArgumentParser()
parser.register('type', 'Bool', str2bool)

parser.add_argument('--binary_loop', help='',default=False,type='Bool')
parser.add_argument('--unary_loop', help='',default=False,type='Bool')
parser.add_argument('--inplace_loop', help='',default=False,type='Bool')
parser.add_argument('--random_computation1', help='',default=False,type='Bool')
parser.add_argument('--blackscholes', help='',default=False,type='Bool')

# parser.add_argument('--mix_loop', help='',default=False,type='Bool')

parser.add_argument('--num_reps', help='',default=1,type=int)
parser.add_argument('--num_els', help='',default=1000,type=int)
parser.add_argument('--num_operands', help='',default=2,type=int)
parser.add_argument('--binary_op', help='',default='+',type=str)
parser.add_argument('--unary_op', help='',default='sqrt',type=str)

args = parser.parse_args()

def binary_loop(lib):
    '''
    lib = np or wn.
    '''
    np.random.seed(1)
    # construct the expr:
    arrays = []
    expr = ''
    for i in range(args.num_operands):
        arrays.append(lib.array(np.random.rand(args.num_els)))
        expr += ' arrays[{i}] '.format(i=str(i))
        if i != args.num_operands-1:
            expr += args.binary_op
    # print(expr)

    start = time.time()
    for i in range(args.num_reps):
        arrays[0] = eval(expr)

    if isinstance(arrays[0], wn.weldarray):
        arrays[0].evaluate()

    print("{} took {} seconds".format(type(arrays[0]), time.time() - start))

def unary_loop(lib, inplace=False):
    '''
    lib = np or wn.
    TODO: Does it make sense to construct cases like np.sqrt(np.sqrt(x)) etc?
    '''
    np.random.seed(1)
    arr = lib.array(np.random.rand(args.num_els))

    start = time.time()

    op = UNARY_OPS[args.unary_op]
    for i in range(args.num_reps):
        if inplace:
            arr = op(arr, out=arr)
        else:
            arr = op(arr)

    if isinstance(arr, wn.weldarray):
        arr.evaluate()

    print("{} took {} seconds".format(type(arr), time.time() - start))

# Taken from:
# https://stackoverflow.com/questions/25950943/why-is-numba-faster-than-numpy-here
def random_computation1(lib):

    x = lib.array(np.random.rand(args.num_els))
    y = lib.array(np.random.rand(args.num_els))
    z = lib.array(np.random.rand(args.num_els))

    start = time.time()
    for i in range(args.num_reps):
        x = x*2.0 - ( y * 55.0 )      # these 4 lines represent use cases
        y = x + y*2.0               # where the processing time is mostly
        z = x + y + 99.0           # a function of, say, 50 to 200 lines
        z = z * ( z - .88 )       # of fairly simple numerical operations

    if isinstance(z, wn.weldarray):
        # print('len of code: ', len(z.weldobj.weld_code))
        z.evaluate()

    end = time.time()
    print("{} took {} seconds".format(type(z), time.time() - start))
    return z

if args.binary_loop:
    print("***********binary loop***********")
    binary_loop(wn)
    binary_loop(np)
    print("***********end***********")

if args.unary_loop:
    print("***********unary loop***********")
    unary_loop(wn)
    unary_loop(np)
    print("***********end***********")

def get_bs_data(use_weld):
    # SIZE = (2 << 20)
    SIZE = args.num_els
    # random prices between 1 and 101
    price = np.float32(np.random.rand(SIZE) * 100)
    # random prices between 0 and 101
    strike = np.float32(np.random.rand(SIZE) * 100)
    # random maturity between 0 and 4
    t = np.float32(1 + np.random.rand(SIZE) * 6)
    # random rate between 0 and 1
    rate = np.float32(0.01 + np.random.rand(SIZE))
    # random volatility between 0 and 1
    vol = np.float32(0.01 + np.random.rand(SIZE))

    if use_weld:
        return wn.weldarray(price), wn.weldarray(strike), wn.weldarray(t), wn.weldarray(rate),wn.weldarray(vol)
    else:
        return price, strike, t, rate, vol

def blackscholes(use_weld):
    """
    All inputs are float32 NumPy arrays.

    price: stock price at time 0.
    strike: strike price
    t: time to maturity in trading years.
    vol: volatility of the stock.
    rate: continuously compounded risk-free rate

    Returns: call and put of each stock as a vector

    Order of operations in terms of usage:

    +, *, sqrt, log, /, -, erf, exp,
    """
    price, strike, t, rate, vol = get_bs_data(use_weld)
    start = time.time()
    for i in range(args.num_reps):
        c05 = 0.5
        c10 = 0.0

        rsig = rate + vol * vol * c05
        vol_sqrt = vol * np.sqrt(t)

        d1 = (np.log(price / strike) + rsig * t) / vol_sqrt
        d2 = d1 - vol_sqrt

        # TODO: Need to fix these.
        # d1 = c05 + c05 * ss.erf(d1 * invsqrt2)
        # d2 = c05 + c05 * ss.erf(d2 * invsqrt2)

        e_rt = np.exp((-rate) * t)

        call = price * d1 - e_rt * strike * d2
        put = e_rt * strike * (c10 - d2) - price * (c10 - d1)

        if isinstance(call, wn.weldarray):
            print('going to call EVALUATE')
            call.evaluate()
            put.evaluate()

    print("{} took {} seconds".format(type(call), time.time() - start))

    return call, put

if args.inplace_loop:
    print("*********inplace loop***********")
    unary_loop(wn, inplace=True)
    unary_loop(np, inplace=True)
    print("***********end***********")

if args.random_computation1:
    print("*********random computation1 ***********")
    random_computation1(wn)
    random_computation1(np)
    print("***********end***********")

if args.blackscholes:
    print("*********black scholes***********")
    blackscholes(True)
    blackscholes(False)
    print("***********end***********")
