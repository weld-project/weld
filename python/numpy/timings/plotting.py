import subprocess as sp
import math
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

import glob, os
import json
import csv
import argparse

# TODO: create common utils file for plotting/timing.
def str2bool(v):
  '''
  used to create special type for argparse
  '''
  return v.lower() in ('yes', 'true', 't', '1')

template = 'python timing.py --num_reps {reps} --num_els {els} --num_operands {operands} '

# max power of 10 that we increase the number of elements to.
parser = argparse.ArgumentParser()
parser.register('type', 'Bool', str2bool)

parser.add_argument('--plot', help='',default=False,type='Bool')
parser.add_argument('--csv', help='',default=True,type='Bool')
parser.add_argument('--max_power', help='',default=1,type=int)
parser.add_argument('--base_name', help='',default='info',type=str)
parser.add_argument('--unary', help='', default=False, type='Bool')
parser.add_argument('--binary', help='', default=False, type='Bool')
parser.add_argument('--inplace', help='', default=False, type='Bool')
parser.add_argument('--random_func', help='', default=False, type='Bool')
parser.add_argument('--blackscholes', help='', default=False, type='Bool')

args = parser.parse_args()

def weld_numpy_plot(X, Y, x_label, y_label, colors, name, title):
        labels = ['Numpy', 'Weld']
        for i, y in enumerate(Y):
            print('y: ', len(y))
            print('X: ', len(X))
            assert len(y) == len(X), 'match'
            plt.plot(X, y, colors[i], label=labels[i], marker='o',
            linestyle='--')

	plt.xlabel(x_label, size=15)
	plt.ylabel(y_label, size=15)
        plt.legend(loc='best')
        plt.title(title)
        print('saving image...: ', name)
        plt.savefig(name)
        plt.close()

def bar_plot(np_times, breakdown, name, title):

    # plot N: bar graph: breakdown of the time spent in weld execution
    fig, ax = plt.subplots()
    index = np.arange(len(np_times))
    bar_width = 0.2
    opacity = 1.0

    # bar: numpy
    plt.bar(index-2*bar_width, np_times, bar_width,
             alpha=opacity,
             color='m',
             label='Numpy')

    # bar: encoding + decoding
    Y = list(np.array(breakdown['encoding']+np.array(breakdown['decoding'])))
    print(len(Y))
    print('Y: ', Y)
    plt.bar(index-bar_width, Y,bar_width,
             alpha=opacity,
             color='b',
             label='Enc + Dec')

    # second bar: running. This should be a stacked bar.
    breakdown['running'] = breakdown['running'] - breakdown['compile_module']
    breakdown['running'] = breakdown['running'] - breakdown['optimization_passes']
    breakdown['running'] = breakdown['running'] - breakdown['run_module']

    # stack 1
    plt.bar(index, breakdown['running'], bar_width,
             alpha=opacity,
             color='r',
             label='Running-Other')

    # stack 2
    plt.bar(index, breakdown['optimization_passes'], bar_width,
             alpha=opacity,
             color='c',
             label='Optimization passes',
             bottom=breakdown['running'])

    # stack 3
    bottom2 = breakdown['running'] + breakdown['optimization_passes']
    plt.bar(index, breakdown['compile_module'], bar_width,
             alpha=opacity,
             color='y',
             label='Compile Module',
             bottom=bottom2)

    # stack 4
    print("adding 4th stack!")
    bottom3 = breakdown['running'] + breakdown['optimization_passes'] \
                + breakdown['compile_module']
    plt.bar(index, breakdown['run_module'], bar_width,
             alpha=opacity,
             color='g',
             label='Run Module',
             bottom=bottom3)

    plt.xlabel('Number of elements as powers of 10')
    plt.ylabel('Seconds')
    plt.title('Bar graphs')
    plt.legend(loc='best')
    plt.title(title)

    print('saving image...: ', name)
    plt.savefig(name)
    plt.close()

def get_time_from_string(string):
    '''
    '''
    # matches scientific notation and stuff.
    numbers = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",
            string)
    # print(string)
    # print(numbers)

    # all the things we are printing so far has just 1 num.
    if len(numbers) == 1:
        return float(numbers[0])
    else:
        return None

def add_n_els(lst, n, start):
    assert n >= 1
    new_l = []
    # assert n == 2, 'only supp'
    for i in range(start, len(lst), n):
        s = 0
        for j in range(start, n, 1):
            s += lst[j]
        new_l.append(s)

    return new_l

def extract_times(output, np_times, weld_times, breakdown, start=0):
    # now let's parse it to get values.
    for o in output:
        t = get_time_from_string(o)
        if t is None:
            continue
        t = round(t, 4)

        if 'ndarray' in o:
            np_times.append(t)
        elif 'weldarray' in o:
            weld_times.append(t)
        elif 'encoding' in o:
            breakdown['encoding'].append(t)
        elif 'decoding' in o:
            breakdown['decoding'].append(t)
        elif 'running' in o:
            breakdown['running'].append(t)
        elif 'uniquify' in o:
            breakdown['uniquify'].append(t)
        elif 'ast' in o:
            breakdown['ast'].append(t)
        elif 'inference' in o:
            breakdown['inference_typed'].append(t)
        elif 'opt passes' in o:
            breakdown['optimization_passes'].append(t)
        elif 'llvm gen' in o:
            breakdown['llvm_gen'].append(t)
        elif 'compile_module' in o:
            breakdown['compile_module'].append(t)
        elif 'module.run' in o:
            breakdown['run_module'].append(t)

    if len(breakdown['encoding']) > len(np_times):
        num_evals = 2
        print('num evals = ', num_evals)
        # assert num_evals <= 2
        for k, val in breakdown.iteritems():
            breakdown[k] = []
            for i in range(start):
                breakdown[k].append(val[i])

            breakdown[k].append(add_n_els(val, num_evals, start))
            assert len(breakdown[k]) == len(np_times)

def plot_stuff(np_times, weld_times, breakdown, name='', title=''):
    '''
    '''
    # convert to np arrays because it makes future life easier.
    np_times = np.array(np_times)
    weld_times = np.array(weld_times)
    for k,v in breakdown.iteritems():
        breakdown[k] = np.array(v)

    assert len(np_times) == len(weld_times)
    for k,v in breakdown.iteritems():
        len(v) == len(weld_times)

    # time to start plotting stuff.

    name1 = 'weld_numpy' + name

    # plot 1: x-elements, y = time. numpy vs weld.
    Y = [np_times, weld_times]
    x_axis = "Number of elements as a power of 10"
    y_axis = "Seconds"
    colors = ["r", "b"]

    weld_numpy_plot(range(len(Y[0])), Y, x_axis, y_axis, colors, name1,
            title)

    name2 = 'bar_graph' + name

    bar_plot(np_times, breakdown, name2, title)

def dump_csv(np_times, weld_times, breakdown, reps, func_name):

    name = args.base_name + '.csv'
    file_exists = False

    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, name)

    if os.path.exists(filename):
        file_exists = True

    # TODO: improve the ordering of these elements.
    header = ['Function Name', 'Reps', 'log(10) elements', 'Numpy Time', 'Weld Time']
    for k in breakdown:
        header.append(k)

    assert len(np_times) == len(weld_times) == len(breakdown['encoding'])

    with open(name, 'a') as f:
        writer = csv.writer(f)
        if not file_exists:
            print('writing row')
            writer.writerow(header)

        for i, np_time in enumerate(np_times):
            row = [func_name, reps, i, np_time, weld_times[i]]
            for k,v in breakdown.iteritems():
                row.append(v[i])

            writer.writerow(row)

def simple_loop(binary=True, inplace=False):
    '''
    '''

    # default values.
    # TODO: Run loop over each of these.
    if binary:
        op_list = ['+']
        # op_list = ['+', '*']
    else:
        # op_list = ['sqrt', 'exp']
        op_list = ['sqrt']

    reps_list = [1]
    operands_list = [2]

    els_list = [int(math.pow(10,i)) for i in range(args.max_power)]

    for op in op_list:
        for reps in reps_list:
            for operands in operands_list:
                np_times = []
                weld_times = []
                # each key will be one of the components of the weld time.
                # each val will be a list like weld_times.
                breakdown = defaultdict(list)
                
                # will plot/dump row to csv after this loop.
                for els in els_list:
                    print("num els = ", els)
                    # call the process and get back stdout.
                    cmd = template.format(reps=str(reps), els=str(els),
                            operands=str(operands), op=op)

                    if inplace:
                        cmd += '--inplace_loop 1 '
                        cmd += '--unary_op ' + op

                    elif binary:
                        cmd += '--binary_loop 1 '
                        cmd += '--binary_op ' + op
                    else:
                        cmd += '--unary_loop 1 '
                        cmd += '--unary_op ' + op

                    cmd = cmd.split(' ')
                    output = sp.check_output(cmd)
                    print(output)
                    output = output.split('\n')
                    extract_times(output, np_times, weld_times, breakdown)

                name = str(args.max_power) + str(reps) + op + str(inplace) + '.png'
                title_template = "Numpy vs Weld, Reps = {reps}, Op = {op}, Inplace = {ip}"
                title = title_template.format(reps=reps, op=op, ip=inplace)
                
                if args.plot:
                    plot_stuff(np_times, weld_times, breakdown, name=name, title=title)

                if args.csv:
                    func_name = op
                    if inplace:
                        func_name + '-inplace'
                    dump_csv(np_times, weld_times, breakdown, reps, func_name)


def random_func(func_flag):
    '''
    '''
    reps_list = []
    for i in range(1):
        reps_list.append(i+1)

    # els_list = [int(math.pow(10,i)) for i in range(5,8,1)]
    els_list = [int(math.pow(10,i)) for i in range(args.max_power)]

    op = func_flag.replace('--', '')
    op = op.replace(' 1', '')
    for reps in reps_list:
        print('reps: ', reps)
        np_times = []
        weld_times = []
        # each key will be one of the components of the weld time.
        # each val will be a list like weld_times.
        breakdown = defaultdict(list)
        for i, els in enumerate(els_list):
            print("num els = ", els)
            # call the process and get back stdout.
            cmd = template.format(reps=str(reps), els=str(els),
                    operands=str(2), op='whatever')

            cmd += func_flag
            print(cmd)

            cmd = cmd.split(' ')
            output = sp.check_output(cmd)
            print(output)
            output = output.split('\n')
            extract_times(output, np_times, weld_times, breakdown, start=i)

        # decompose plotting stuff and add it here.
        name = str(args.max_power) + str(reps) + op
        title_template = "Numpy vs Weld, Reps = {reps}, func = {op}"
        title = title_template.format(reps=reps, op=op)
        
        # TODO: add args here.
        if args.plot:
            name += '.png'
            plot_stuff(np_times, weld_times, breakdown, name=name, title=title)

        if args.csv:
            func_name = 'random'
            dump_csv(np_times, weld_times, breakdown, reps, func_name)


def plot_jsons():
    '''
    '''
    # load all .txt files in directory
    os.chdir("jsons")
    for f in glob.glob("*.txt"):
        print(f)

        with open(f) as json_data:
            d = json.load(json_data)
            # title, np_times, weld_times, breakdown.
            title = d[0]
            assert len(d[1]) == len(d[2])
            plot_stuff(d[1], d[2], d[3], title=title, name=f.replace('.txt', '.png'))

if __name__ == '__main__':
    if args.unary:
        simple_loop(binary=False)
    if args.binary:
        simple_loop(binary=True)
    if args.inplace:
        simple_loop(binary=False, inplace=True)
    
    if args.random_func:
        func_flag = '--random_computation1 1'
        random_func(func_flag)
    
    if args.blackscholes:
        func_flag = '--blackscholes 1'
        random_func(func_flag)
    
    # if we have json files
    # plot_jsons()
