
from gym.envs.registration import registry, register, make, spec

register(
  id='Toy-v0',
  entry_point='toy_env:Toy',
  timestep_limit=3,
  nondeterministic=False,
)
