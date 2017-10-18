import pandas as pd
import numpy as np
import sys
import time

years = range(1880, 2011)
pieces = []
columns = ['name', 'sex', 'births']
for year in years:
    path = 'data/names/yob%d.txt' % year
    frame = pd.read_csv(path, names=columns)
    frame['year'] = year
    pieces.append(frame)

# Concatenate everything into a single DataFrame
names = pd.concat(pieces, ignore_index=True)
print "Size of names: %d" % len(names)

def get_top1000(group):
    return group.sort_values(by='births', ascending=False)[0:1000]

#Time preprocessing step
start0 = time.time()
grouped = names.groupby(['year', 'sex'])
top1000 = grouped.apply(get_top1000)
end0 = time.time()

# Drop the group index, not needed
top1000.reset_index(inplace=True, drop=True)
start1 = time.time()
all_names = pd.Series(top1000.name.unique())

lesley_like = all_names[all_names.str.lower().str.contains('lesl')]

filtered = top1000[top1000.name.isin(lesley_like)]

table = filtered.pivot_table('births', index='year',
                             columns='sex', aggfunc='sum')

table = table.div(table.sum(1), axis=0)
end1 = time.time()


print "Time taken by preprocess portion:   %.5f" % (end0 - start0)
print "Time taken by analysis portion  :   %.5f" % (end1 - start1)
