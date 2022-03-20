import json
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

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
# Plotting function

def plot_perf(data, filename, title, xkey, xlabel, xscale, ykey, ylabel):
    fig, ax1 = plt.subplots(figsize=(12, 8))

    for series, points in data.items():
        ax1.plot(points[xkey], points[ykey], label=series)

    ax1.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax1.set_xscale(xscale)
    ax1.set_ybound(lower=0, upper=flops)
    ax1.grid()
    ax1.legend()

    ax1.yaxis.set_major_formatter(lambda x, _: f'{x / 1e9:.2f}')

    if flops:
        ax2 = ax1.twinx()
        ax2.set_ylabel('% of peak')
        ax2.yaxis.set_major_formatter(lambda x, _: f'{x:.0%}')

        ax1.set_yticks(np.linspace(ax1.get_ybound()[0], ax1.get_ybound()[1], 7))
        ax2.set_yticks(np.linspace(ax2.get_ybound()[0], ax2.get_ybound()[1], 7))

    plt.savefig(filename)


##
# Create aspect ratio plots

plot_perf(
    data=aspect_plots,
    filename='sgemm_aspect_ratio.png',
    title='SGEMM performance with fixed workload and\nvariable output aspect ratio\n',
    xkey='ratio',
    xlabel='aspect ratio (m/n)',
    xscale='log',
    ykey='flops',
    ylabel='GFLOP/s',
)

##
# Create square ratio plots

plot_perf(
    data=square_plots,
    filename='sgemm_square.png',
    title='SGEMM performance with square matrices',
    xkey='n',
    xlabel='dimension (n)',
    xscale='linear',
    ykey='flops',
    ylabel='GFLOP/s',
)
