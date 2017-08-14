import numpy as np
import sys
sys.path.append('..')
import weldnumpy as wa

import py.test
import random
import timeit
import time
import sys
import argparse

'''
TODO: Adjust the number of arrays being added as an arg etc.
'''

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
parser.add_argument('--mix_loop', help='',default=False,type='Bool')

parser.add_argument('--num_reps', help='',default=1,type=int)
parser.add_argument('--num_els', help='',default=1000,type=int)
parser.add_argument('--num_operands', help='',default=2,type=int)
parser.add_argument('--binary_op', help='',default='+',type=str)
parser.add_argument('--unary_op', help='',default='sqrt',type=str)

args = parser.parse_args()
# OPS =

def binary_loop(lib):
    '''
    lib = np or wa.
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

    if isinstance(arrays[0], wa.weldarray):
        arrays[0].evaluate()

    print("{} took {} seconds".format(type(arrays[0]), time.time() - start))

def unary_loop(lib):
    '''
    lib = np or wa.
    TODO: Does it make sense to construct cases like np.sqrt(np.sqrt(x)) etc?
    '''
    np.random.seed(1)
    arr = lib.array(np.random.rand(args.num_els))

    start = time.time()

    op = UNARY_OPS[args.unary_op]
    for i in range(args.num_reps):
        arr = op(arr)

    if isinstance(arr, wa.weldarray):
        arr.evaluate()

    print("{} took {} seconds".format(type(arr), time.time() - start))

if args.binary_loop:
    print("***********binary loop***********")
    binary_loop(wa)
    binary_loop(np)
    print("***********end***********")

if args.unary_loop:
    print("***********unary loop***********")
    unary_loop(wa)
    unary_loop(np)
    print("***********end***********")
