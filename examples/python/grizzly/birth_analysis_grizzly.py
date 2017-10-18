import pandas as pd
import grizzly.grizzly as gr
import numpy as np
import time
import sys

years = range(1880, 2011)
pieces = []
columns = ['name', 'sex', 'births']
for year in years:
    path = 'data/names/yob%d.txt' % year
    frame = pd.read_csv(path, names=columns)
    frame['year'] = year
    pieces.append(frame)

# Concatenate everything into a single DataFram
names = pd.concat(pieces, ignore_index=True)
print "Size of names: %d" % len(names)

def get_top1000(group):
    return group.sort_values(by='births', ascending=False).slice(0,1000)

#Time preprocessing step
start0 = time.time()
names = gr.DataFrameWeld(names)
grouped = names.groupby(['year', 'sex'])
top1000 = grouped.apply(get_top1000).evaluate(True)
end0 = time.time()

# Drop the group index, not needed
# Does nothing in NVL
top1000.reset_index(inplace=True, drop=True)
start1 = time.time()
top1000 = gr.DataFrameWeld(top1000)
top1000names = top1000.name

all_names = top1000names.unique()

lesley_like = all_names[all_names.lower().contains('lesl')]

filtered = top1000[top1000names.isin(lesley_like)]

table = filtered.pivot_table('births', index='year',
                             columns='sex', aggfunc='sum')

table = table.div(table.sum(1), axis=0)
end1= time.time()

print "Time taken by preprocess portion: %.5f" %(end0 - start0)
print "Time taken by analysis portion  : %.5f" % (end1- start1)
