"""
visualize_actions.py

This script visualizes actions over the observation space.
This script assumes that the agent is stateless and has been trained using "running rank" inputs.

Usage:
TODO
"""

from argparse import ArgumentParser
import os

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

def visualize_actions(matrix, agent_name):

	create_visual(matrix, agent_name)

	plt.title(f'Actions By Observation Space ({agent_name})')
	plt.xlabel('Candidates Seen')
	plt.ylabel('Candidate (Running) Rank')

	# Save the figure to a PNG file without displaying it
	output_filename = f'actions_{agent_name.lower()}.png'
	plt.savefig(output_filename, dpi=300, bbox_inches='tight')

	plt.close()

def visualize_probs(matrix, agent_name):

	create_visual(matrix, agent_name)

	plt.title(f'Probabilities By Observation Space ({agent_name})')
	plt.xlabel('Candidates Seen')
	plt.ylabel('Candidate (Running) Rank')

	# Save the figure to a PNG file without displaying it
	output_filename = f'probs_{agent_name.lower()}.png'
	plt.savefig(output_filename, dpi=300, bbox_inches='tight')

	plt.close()

agent = None
env_type = None
agent_name = None

if args.agent == 'optimal':

	agent = STR2AGENT[args.agent](100)
	env_type = 'running_rank'
	agent_name = 'Optimal'

elif args.agent == 'drl':

	agent = STR2AGENT[args.agent](os.path.join('checkpoints', 'running_rank_10'))
	env_type = 'running_rank'
	agent_name = 'DRL'

assert agent is not None and env_type is not None and agent_name is not None

if env_type == 'running_rank':

	rows, cols = 100, 100
	actions_matrix = np.zeros((rows, cols), dtype=np.float32)
	probs_matrix = None

	for col in range(0, 100):

		candidates_remaining = np.full((col + 1, ), (100 - col) / 100, dtype=np.float32)

		# Compute running rank for the ith column
		running_rank = (np.arange(col + 1, dtype=np.float32) + 1) / (col + 1)

		actions, probs = agent.compute_actions(np.column_stack((candidates_remaining, running_rank)))

		actions_matrix[: (col + 1), col] = actions

		if probs is None:

			assert probs_matrix is None

		else:

			if probs_matrix is None:

				probs_matrix = np.zeros((rows, cols), dtype=np.float32)

			probs_matrix[: (col + 1), col] = probs

	visualize_actions(actions_matrix, agent_name)

	if probs_matrix is not None:

		visualize_probs(probs_matrix, agent_name)

plt.close('all')
