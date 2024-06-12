from envs.pvtol_env import PvtolEnv


"""
This file includes a function that simply returns one of the two supported environments. 
"""

def build_env(args):
    """Build our custom gym environment."""

    if args.env_name == 'Pvtol':
        return PvtolEnv(args.seed)

    else:
        raise Exception('Env {} not supported!'.format(args.env_name))
