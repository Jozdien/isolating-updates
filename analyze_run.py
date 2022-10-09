import argparse

import utils

parser = argparse.ArgumentParser(description='Enter path to folder containing run files.')
parser.add_argument('--path', metavar='path', type=str, help='path to run')
args = parser.parse_args()
path = args.path

rewards = utils.load_rewards(path)

utils.print_stats(rewards)