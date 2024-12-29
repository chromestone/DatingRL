"""
visualize_actions.py

This script visualizes actions over the observation space.

Usage:
TODO
"""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import seaborn as sns

from datingrl.agents.stateless import STR2AGENT

parser = ArgumentParser(description='Visualize actions over the observation space')

parser.add_argument('agent', choices=STR2AGENT.keys(), help='Name of an agent in datingrl.agents.stateless (without any prefix)')

args = parser.parse_args()

def create_visual(matrix, agent_name):

	mask = np.tril(np.ones_like(matrix, dtype=bool), k=-1)
	ax = sns.heatmap(matrix, vmin=0, vmax=1, cmap="viridis", cbar=True, mask=mask)

	ax.xaxis.set_major_locator(MultipleLocator(10))
	ax.yaxis.set_major_locator(MultipleLocator(10))

	ax.xaxis.set_major_formatter(lambda x, _: f'{x:.0f}' if x != 0 else '')
	ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f}' if x != 0 else '')

	ax.xaxis.set_minor_locator(MultipleLocator(5))
	ax.yaxis.set_minor_locator(MultipleLocator(5))

	plt.xticks(rotation=0)
	# ax.xaxis.set_ticks_position('top')

	plt.title(f'Actions By Observation Space ({agent_name})')
	plt.xlabel('Candidates Seen')
	plt.ylabel('Candidate (Running) Rank')

	# Save the figure to a PNG file without displaying it
	output_filename = "random_matrix_plot.png"
	plt.savefig(output_filename, dpi=300, bbox_inches='tight')

	print(f"Figure saved as {output_filename}")

if args.agent == 'optimal':

	agent = STR2AGENT[args.agent](100)

	rows, cols = 100, 100
	actions_matrix = np.full((rows, cols), 0.5, dtype=np.float32)

	for col in range(0, 100):

		candidates_remaining = np.full((col + 1, ), (100 - col) / 100, dtype=np.float32)

		# Compute running rank for the ith column
		running_rank = (np.arange(col + 1) + 1) / (col + 1)

		actions, _ = agent.compute_actions(np.column_stack((candidates_remaining, running_rank)))
		actions_matrix[: (col + 1), col] = actions

	create_visual(actions_matrix, 'Optimal')

plt.close('all')
