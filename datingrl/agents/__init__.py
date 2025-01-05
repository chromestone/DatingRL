from . import stateless
from . import stateful

STR2AGENT = {
	**{f'stateless.{k}' : v for k, v in stateless.STR2AGENT.items()},
	**{f'stateful.{k}'  : v for k, v in stateful .STR2AGENT.items()}
}
