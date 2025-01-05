"""
predict.py

TODO
"""

from csv import DictWriter
from argparse import ArgumentParser

import tqdm

from datingrl.agents import STR2AGENT
from datingrl.envs import STR2ENV

parser = ArgumentParser(description='TODO')

parser.add_argument(
	'-a',
	'--agent',
	choices=STR2AGENT.keys(),
	required=True,
	help='Name of an agent in datingrl.agents (with stateful/stateless prefix)'
)
parser.add_argument(
	'-o',
	'--output_file',
	required=True,
	help='Output CSV filename'
)
parser.add_argument(
	'--trials',
	default=1000,
	type=int,
	help='Number trials. This is the number of times the env is reset'
)

args = parser.parse_args()

agent = None
env = None

if args.agent == 'stateful.optimal':

	agent = STR2AGENT[args.agent](1000)
	env = STR2ENV['real_score']({})

elif args.agent == 'stateless.optimal':

	agent = STR2AGENT[args.agent](1000)
	env = STR2ENV['running_rank']({})

else:

	raise NotImplementedError(f'Agent "{args.agent}" is not supported!')

assert agent is not None and env is not None

prev_observation = None

with open(args.output_file, 'w', encoding='utf-8', newline='') as fp:

	fieldnames = ['index', 'score', 'rank', 'reward']
	writer = DictWriter(fp, fieldnames=fieldnames)

	writer.writeheader()

	for _ in tqdm.tqdm(range(args.trials)):

		# seed the very first reset
		if prev_observation is None:

			observation, info = env.reset(0)

		else:

			observation, info = env.reset()

		terminated = False
		while not terminated:

			prev_observation = observation
			action = agent.compute_single_action(observation)
			observation, reward, terminated, _, info = env.step(action)

		info['index'] = prev_observation[0]
		info['reward'] = reward

		writer.writerow(info)

		# reinitialize stateful agents below

		if args.agent == 'stateful.optimal':
			
			agent = STR2AGENT[args.agent](1000)
