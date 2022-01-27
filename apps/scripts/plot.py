import json
import re
from pathlib import Path
from typing import Tuple, List

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt

extract_n = re.compile(r'^[^/]+/(\d+)')

sgemm = json.loads(Path('../mkl_sgemm.json').read_text())['benchmarks']
ssyrk = json.loads(Path('../mkl_ssyrk.json').read_text())['benchmarks']


def get_data(series, key) -> Tuple[List[int], List[float]]:
    return tuple(zip(*(
        (int(extract_n.match(point['name']).group(1)), float(point[key]))
        for point in series
    )))


def plot_data(key, **kwargs):
    ax: matplotlib.axes.Axes
    fig, ax = plt.subplots()

    ax.plot(*get_data(sgemm, key), label='sgemm')
    ax.plot(*get_data(ssyrk, key), label='ssyrk')

    ax.set(xlabel='matrix dimension (n)', ylabel=key, **kwargs)
    ax.grid()
    ax.legend()

    return fig


plot_data('flops', title='MKL throughput comparison')
plot_data('real_time', title='MKL wall-clock comparison')

plt.show()
