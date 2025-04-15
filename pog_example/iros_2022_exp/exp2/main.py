import logging, argparse
from pog.graph.graph import Graph
from pog.planning.planner import test, Searcher
from pog.planning.problem import PlanningOnGraphProblem
from pog.planning.utils import *
from pog.algorithm.params import CHECK_COLLISION_IK
import os
import numpy as np
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-viewer',
                        action='store_true',
                        help='Enable the viewer and visualizes the plan')
    parser.add_argument('-filter',
                        action='store_true',
                        help='Enable to filter redundant actions during searching')
    parser.add_argument('--max_iter',
                        type=int,
                        default=0,
                        help='Maximum number of iterations for the planner')
    parser.add_argument('--max_opt_time',
                        type=int,
                        default=200,
                        help='Maximum time (in seconds) for the planner')
    
    args = parser.parse_args()

    data_folder = "result/data/exp2"
    iter = "_once" if args.max_iter == 1 else ""
    opt_time = f"_in{int(args.max_opt_time)}"
    
    data_file = f"{data_folder}/optimize{iter}{opt_time}.npy"
    log_file = f"{data_folder}/optimize{iter}{opt_time}_log.npy"

    print("\033[93mArguments:", args, f"Check collision-aware IK: {CHECK_COLLISION_IK}\033[0m")

    logFormatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)s] [%(levelname)-5.5s]  %(message)s"
    )
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler("pog_example/iros_2022_exp/exp2/test.log")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(logging.INFO)

    # Planning
    g_start = Graph('exp2-init', file_dir='pog_example/iros_2022_exp/exp2/', file_name='init.json')
    g_goal = Graph('exp2-goal', file_dir='pog_example/iros_2022_exp/exp2/', file_name='goal.json')
    
    # Environment(g_goal)
    start_time = time.time()
    ites, path = test(Searcher, 
                      problem=PlanningOnGraphProblem(g_start, g_goal, parking_place=99), 
                      pruning=args.filter, 
                      max_iter=args.max_iter, 
                      max_opt_time=args.max_opt_time)
    
    action_seq = path_to_action_sequence(path)            
    apply_action_sequence_to_graph(g_start, g_goal, action_seq, visualize=args.viewer)

    experiment_time = time.time() - start_time
    experiment_path_length = len(action_seq) - 1

    print("\033[93mSearch time: {:.2f} seconds;".format(experiment_time),
          "Path length: {};".format(experiment_path_length),
          "Iterations: {}.\033[0m".format(ites))