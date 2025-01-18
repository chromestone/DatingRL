"""
visualize_actions.py

This script visualizes actions over the observation space.

Usage:
python3 visualize_actions.py -a optimal
python3 visualize_actions.py -a ppo -c checkpoints/ppo_running_rank_10
python3 visualize_actions.py -a dqn -c checkpoints/dqn_running_rank_180
"""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import seaborn as sns

from datingrl.agents.stateless import STR2AGENT

parser = ArgumentParser(description='Visualize actions over the observation space')

parser.add_argument(
	'-a',
	'--agent',
	choices=STR2AGENT.keys(),
	required=True,
	help='Name of an agent in datingrl.agents.stateless (without any prefix)'
)
parser.add_argument(
	'-i',
	'--input_type',
	default='running_rank',
	choices=['running_rank'],
	help='Type of inputs the agent expects'
)
parser.add_argument(
	'-c',
	'--ckpt_path',
	help='Path to checkpoint (if agent is trained)'
)

args = parser.parse_args()

def create_visual(matrix: np.ndarray[np.float32], agent_name: str):

	mask = np.tril(np.ones_like(matrix, dtype=bool), k=-1)
	ax = sns.heatmap(matrix, vmin=0, vmax=1, cmap="viridis", cbar=True, mask=mask)

	ax.xaxis.set_major_locator(MultipleLocator(10))
	ax.yaxis.set_major_locator(MultipleLocator(10))

	ax.xaxis.set_major_formatter(lambda x, _: f'{x:.0f}' if x != 0 else '')
	ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f}' if x != 0 else '')

	ax.xaxis.set_minor_locator(MultipleLocator(5))
	ax.yaxis.set_minor_locator(MultipleLocator(5))

	plt.xticks(rotation=0)

def visualize_actions(matrix: np.ndarray[np.float32], agent_name: str):

	create_visual(matrix, agent_name)

	plt.title(f'Actions By Observation Space ({agent_name})')
	plt.xlabel('Candidates Seen')
	plt.ylabel('Candidate (Running) Rank')

	# Save the figure to a PNG file without displaying it
	output_filename = f'actions_{agent_name.lower()}.png'
	plt.savefig(output_filename, dpi=300, bbox_inches='tight')

	plt.close()

def visualize_probs(matrix: np.ndarray[np.float32], agent_name: str):

	create_visual(matrix, agent_name)

	plt.title(f'Probabilities By Observation Space ({agent_name})')
	plt.xlabel('Candidates Seen')
	plt.ylabel('Candidate (Running) Rank')

	# Save the figure to a PNG file without displaying it
	output_filename = f'probs_{agent_name.lower()}.png'
	plt.savefig(output_filename, dpi=300, bbox_inches='tight')

	plt.close()

agent = None
agent_name = None

if args.agent == 'optimal':

	assert args.input_type == 'running_rank'

	agent = STR2AGENT[args.agent](100)
	agent_name = 'Optimal'

elif args.agent == 'ppo':

	assert args.input_type == 'running_rank'
	assert args.ckpt_path

	agent = STR2AGENT[args.agent](100, args.ckpt_path)
	agent_name = 'PPO'

elif args.agent == 'dqn':

	assert args.input_type == 'running_rank'
	assert args.ckpt_path

	agent = STR2AGENT[args.agent](100, args.ckpt_path)
	agent_name = 'DQN'

else:

	raise NotImplementedError(f'Agent "{args.agent}" is not supported!')

assert agent is not None and agent_name is not None

if args.input_type == 'running_rank':

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

else:

	raise NotImplementedError(f'Input type "{args.input_type}" is not supported!')

plt.close('all')
