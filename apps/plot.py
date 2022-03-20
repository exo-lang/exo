import json
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

matcher = re.compile(r'^sgemm_(?P<name>\w+)/(?P<m>\d+)/(?P<n>\d+)/(?P<k>\d+)$')

square_plots = defaultdict(lambda: defaultdict(list))
aspect_plots = defaultdict(lambda: defaultdict(list))

for filename in sys.argv[1:]:
    with open(filename) as f:
        data = json.load(f)

    for point in data['benchmarks']:
        if m := matcher.match(point['name']):
            groups = m.groupdict()
            series = groups.get('name')

            if groups['m'] == groups['n'] == groups['k']:
                square_plots[series]['n'].append(float(groups['n']))
                square_plots[series]['flops'].append(float(point['flops']))

            if groups['k'] == '512':
                aspect_plots[series]['ratio'].append(
                    float(groups['m']) / float(groups['n']))
                aspect_plots[series]['flops'].append(float(point['flops']))

for series, points in square_plots.items():
    points['n'], points['flops'] = zip(
        *sorted(zip(points['n'], points['flops'])))

for series, points in aspect_plots.items():
    points['ratio'], points['flops'] = zip(
        *sorted(zip(points['ratio'], points['flops'])))

##
# Get peak flops

if flops := os.getenv('MAX_GFLOPS'):
    flops = float(flops) * 1e9

##
# Create aspect ratio plots

fig, ax = plt.subplots(figsize=(12, 8))

for series, points in aspect_plots.items():
    ax.plot(points['ratio'], points['flops'], label=series)

ax.set(xlabel='aspect ratio (m/n)', ylabel='flops',
       title='SGEMM performance with fixed workload and\nvariable output aspect ratio\n')
ax.set_xscale('log')
ax.set_ybound(lower=0, upper=flops)
ax.grid()
ax.legend()

if flops:
    ax_right = ax.twinx()
    ax_right.set_ylabel('% of peak')
    ax_right.set_ylim(0, 100)

plt.savefig('sgemm_aspect_ratio.png')

##
# Create square ratio plots

fig, ax = plt.subplots(figsize=(12, 8))

for series, points in square_plots.items():
    ax.plot(points['n'], points['flops'], label=series)

ax.set(xlabel='dimension (n)', ylabel='flops',
       title='SGEMM performance with square matrices')
ax.set_ybound(lower=0, upper=flops)
ax.grid()
ax.legend()

if flops:
    ax_right = ax.twinx()
    ax_right.set_ylabel('% of peak')
    ax_right.set_ylim(0, 100)

plt.savefig('sgemm_square.png')
