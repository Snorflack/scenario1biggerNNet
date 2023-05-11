import logging

import coloredlogs

from Coach import Coach
from alphazero_game import AtlatlGame as Game
from alphazero_nnet import NNetWrapper as nn
from utils import *
import torch
import util

log = logging.getLogger(__name__)

coloredlogs.install(level='DEBUG')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 10,           # Was 1000
    'numEps': 5,              # Was 100. Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.55,     # During arena playoff, new neural net will be accepted if threshold or more of games are won. made .55 like in paper
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 20,          # Was 25. Number of games moves for MCTS to simulate. made 30k for testing now 50 for time
    'arenaCompare': 20,         # Was 40. Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 4,   # cjd Scale to max score? Their C constant that tweeks exploration. described in their paper
    'run_name': 'Scenario1TBtest',
    'checkpoint': './temptest/',
    'load_model': False,
    'load_folder_file': ('/temp','best.pth.tar'),
    'numItersForTrainExamplesHistory': 50,
    'redAI': "pass-agg",
    'blueAI': "pass-agg",
    'heuristicEvalFn': None, # Either None or util.playAndScore
    'scenario_name':"Scenario1middleIsland.scn",
    'verbose':False
})


def main():
    print(args)
    print('Using GPU?',torch.cuda.is_available())
    print('scenario name',args.scenario_name)
    log.info('Loading %s...', Game.__name__)
    g = Game(args.scenario_name)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
