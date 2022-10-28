'''
Code to visualize the composite plots from multiple runs with linked pan and zoom.
'''

import os
import math
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

parser = argparse.ArgumentParser(description='Enter runs to compare.')
parser.add_argument('--runs', metavar='runs', type=str, nargs='+', help='runs to compare')
args = parser.parse_args()
runs = args.runs
if runs == None:
    runs = [
        'analysis/2022-10-20_19-11-51_analysis.png', 
        'analysis/2022-10-20_19-34-20_analysis.png',
        'analysis/2022-10-20_20-35-14_analysis.png',
        'analysis/2022-10-20_21-09-58_analysis.png',
        'analysis/2022-10-21_00-40-54_analysis.png',
        'analysis/2022-10-21_01-17-08_analysis.png',
        'analysis/2022-10-21_01-26-50_analysis.png',
        'analysis/2022-10-21_01-37-13_analysis.png',
    ]
elif runs[0] == 'all':
    runs = ["analysis/" + run for run in os.listdir('analysis')]

imgs = [mpimg.imread(run) for run in runs]
num_imgs = len(imgs)
ipr = 2  # Images per row
rows = math.ceil(num_imgs / ipr) if num_imgs >= ipr else 1
cols = ipr if num_imgs > ipr else num_imgs

fig, axes = plt.subplots(rows, cols, figsize=[25, 60], sharex=True, sharey=True)
plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98, hspace=-0.05, wspace=-0.25)
fig.tight_layout()

for (i, img) in enumerate(imgs):
    if rows == 1:
        axes[i].imshow(img, aspect='auto')
    else:
        axes[int(i / ipr)][i % ipr].imshow(img, aspect='auto')
[ax.set_axis_off() for ax in axes.ravel()]

plt.show()