import subprocess as sp
import math
import re
from collections import defaultdict
import hashlib
import matplotlib.pyplot as plt
import numpy as np

template = 'python timing.py --num_reps {reps} --num_els {els} --num_operands {operands} '

# max power of 10 that we increase the number of elements to.
MAX_POWER = 9

def weld_numpy_plot(X, Y, x_label, y_label, colors, name, title):
        labels = ['Numpy', 'Weld']
        for i, y in enumerate(Y):
            # print('y: ', len(y))
            # print('X: ', len(X))
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

def extract_times(output, np_times, weld_times, breakdown):

    # now let's parse it to get values.
    for o in output:
        t = get_time_from_string(o)
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

def simple_loop(binary=True):
    '''
    '''

    # default values.
    # TODO: Run loop over each of these.

    if binary:
        op_list = ['+', '*']
    else:
        op_list = ['sqrt', 'exp']

    reps_list = [1, 10, 20, 50]
    operands_list = [2]

    els_list = [int(math.pow(10,i)) for i in range(MAX_POWER)]

    for op in op_list:
        for reps in reps_list:
            for operands in operands_list:
                np_times = []
                weld_times = []
                # each key will be one of the components of the weld time.
                # each val will be a list like weld_times.
                breakdown = defaultdict(list)

                for els in els_list:
                    print("num els = ", els)
                    # call the process and get back stdout.
                    cmd = template.format(reps=str(reps), els=str(els),
                            operands=str(operands), op=op)
                    if binary:
                        cmd += '--binary_loop 1 '
                        cmd += '--binary_op ' + op
                    else:
                        cmd += '--unary_loop 1 '
                        cmd += '--unary_op ' + 'sqrt'

                    cmd = cmd.split(' ')
                    output = sp.check_output(cmd)
                    print(output)
                    output = output.split('\n')
                    extract_times(output, np_times, weld_times, breakdown)

                # convert to np arrays because it makes future life easier.
                np_times = np.array(np_times)
                weld_times = np.array(weld_times)
                for k,v in breakdown.iteritems():
                    breakdown[k] = np.array(v)

                assert len(np_times) == len(weld_times)
                for k,v in breakdown.iteritems():
                    len(v) == len(weld_times)

                # time to start plotting stuff.

                # hcmd = hashlib.sha1(str(cmd)).hexdigest()
                # name = 'weld_numpy' + hcmd[0:4] + '.png'
                name = 'weld_numpy' + str(MAX_POWER) + str(reps) + op + str(operands) + '.png'

                # plot 1: x-elements, y = time. numpy vs weld.
                Y = [np_times, weld_times]
                x_axis = "Number of elements as a power of 10"
                y_axis = "Seconds"
                colors = ["r", "b"]

                title_template = "Numpy vs Weld, Reps = {reps}, Op = {op}, operands = {operands}"
                title = title_template.format(reps=reps, op=op, operands=operands)
                weld_numpy_plot(range(MAX_POWER), Y, x_axis, y_axis, colors, name,
                        title)

                name = 'bar_graph' + str(MAX_POWER) + str(reps) + op + str(operands) + '.png'
                bar_plot(np_times, breakdown, name, title)

if __name__ == '__main__':
    simple_loop(binary=False)
    simple_loop(binary=True)

