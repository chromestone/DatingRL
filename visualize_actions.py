"""
visualize_actions.py

This script visualizes actions over the observation space.

Usage:
TODO
"""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from datingrl.agents.stateless import STR2AGENT

parser = ArgumentParser(description='Visualize actions over the observation space')

parser.add_argument('agent', choices=STR2AGENT.keys(), help='Name of an agent in datingrl.agents.stateless (without any prefix)')

args = parser.parse_args()

def create_visual(matrix):

	# Create a figure
	plt.figure(figsize=(6, 6))

	# Create a matshow plot
	matshow_plot = plt.matshow(matrix)#, fignum=1)

	# Add a color bar and scale it between 0 and 1
	color_bar = plt.colorbar(matshow_plot)
	color_bar.mappable.set_clim(0, 1)

	# Save the figure to a PNG file without displaying it
	output_filename = "random_matrix_plot.png"
	plt.savefig(output_filename, bbox_inches='tight')

	# Close the figure to free resources
	plt.close()

	print(f"Figure saved as {output_filename}")

if args.agent == 'optimal':

	agent = STR2AGENT[args.agent](100)

	rows, cols = 100, 100
	actions_matrix = np.full((rows, cols), 0.5, dtype=np.float32)

	for col in range(0, 100):

		candidates_remaining = np.full((col + 1, ), (100 - col) / 100, dtype=np.float32)

		# Compute running rank for the ith column
		running_rank = (np.arange(col + 1) + 1) / (col + 1)

		# print(np.column_stack((candidates_remaining, running_rank)))

		actions, _ = agent.compute_actions(np.column_stack((candidates_remaining, running_rank)))
		# print(actions)
		actions_matrix[: (col + 1), col] = actions

	print(actions_matrix)
	create_visual(actions_matrix)
