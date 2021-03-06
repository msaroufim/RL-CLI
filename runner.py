# Based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/experiments/train_cartpole.py
# https://github.com/openai/retro/blob/master/examples/random_agent.py

import argparse
import retro
from baselines import deepq


parser = argparse.ArgumentParser()
parser.add_argument('game', help='the name or path for the game to run')
parser.add_argument('state', nargs='?', help='the initial state file to load, minus the extension')
#parser.add_argument('algorithm', help='the name of the reinforcement learning trainer')
parser.add_argument('--scenario', '-s', default='scenario', help='the scenario file to load, minus the extension')
parser.add_argument('--record', '-r', action='store_true', help='record bk2 movies')
parser.add_argument('--verbose', '-v', action='count', default=1, help='increase verbosity (can be specified multiple times)')
parser.add_argument('--quiet', '-q', action='count', default=0, help='decrease verbosity (can be specified multiple times)')
args = parser.parse_args()

env = retro.make(args.game, args.state or retro.STATE_DEFAULT, scenario=args.scenario, record=args.record)
verbosity = args.verbose - args.quiet

# if training proceeds slowly then try varying parameters here
act = deepq.learn(
    env,
    q_func=model,
    lr=1e-3,
    max_timesteps=100000,
    buffer_size=50000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    print_freq=10,
    callback=callback
)

def callback(lcl, _glb):
    # stop training if reward exceeds 1000000
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 1000000
    return is_solved

output_string = "Saving model to {0} {1}".format(args.game, args.state)

print("Saving model to {0} {1}".format(args.game, args.state))
act.save("Saving model to {0} {1}".format(args.game, args.state))

act = deepq.load("cartpole_model.pkl")

while True:
    obs, done = env.reset(), False
    episode_rew = 0
    while not done:
        env.render()
        obs, rew, done, _ = env.step(act(obs[None])[0])
        episode_rew += rew
    print("Episode reward", episode_rew)