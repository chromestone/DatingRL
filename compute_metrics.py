"""
compute_metrics.py

TODO
"""

from argparse import ArgumentParser
from collections import defaultdict
import json

import pandas as pd
import scipy

parser = ArgumentParser(description='TODO')

parser.add_argument(
	'input_file',
	help='Input CSV filename'
)
parser.add_argument(
	'output_file',
	help='Output JSON filename'
)

args = parser.parse_args()

df = pd.read_csv(args.input_file)
num_rows = len(df.index)

output_dict = {}

output_dict['100_percentile'] = (df['rank'] == 1000).sum().item() * 100 / num_rows
output_dict['95_percentile'] = (df['rank'] >= 950).sum().item() * 100 / num_rows
output_dict['90_percentile'] = (df['rank'] >= 900).sum().item() * 100 / num_rows
output_dict['forced_choice'] = (df['index'] <= 0.01).sum().item() * 100 / num_rows
output_dict['avg_rank'] = df['rank'].mean().item()
output_dict['avg_reward'] = df['reward'].mean().item()

sample_estimates = defaultdict(list)
for i in range(0, num_rows, 1000):

	chunk = df.iloc[i : i + 1000]
	sample_estimates['100_percentile'].append((chunk['rank'] == 1000).sum().item() / 10)
	sample_estimates['95_percentile'].append((chunk['rank'] >= 950).sum().item() / 10)
	sample_estimates['90_percentile'].append((chunk['rank'] >= 900).sum().item() / 10)
	sample_estimates['forced_choice'].append((chunk['index'] <= 0.01).sum().item() / 10)

# standard error estimate
print(sample_estimates)
output_dict['100_percentile_se'] = scipy.stats.sem(sample_estimates['100_percentile'])
output_dict['95_percentile_se'] = scipy.stats.sem(sample_estimates['95_percentile'])
output_dict['90_percentile_se'] = scipy.stats.sem(sample_estimates['90_percentile'])
output_dict['forced_choice_se'] = scipy.stats.sem(sample_estimates['forced_choice'])

with open(args.output_file, 'w', encoding='utf-8') as fp:

	json.dump(output_dict, fp, indent=4)
