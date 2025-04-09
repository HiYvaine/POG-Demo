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
    parser.add_argument('--max_iter',
                        type=int,
                        default=1,
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
        
    if os.path.exists(data_file):
        data = np.load(data_file)
        total_time, total_path_length, total_experiments = data
    else:
        total_time = 0
        total_path_length = 0
        total_experiments = 0

    # Planning
    g_start = Graph('exp2-init', file_dir='pog_example/iros_2022_exp/exp2/', file_name='init.json')
    g_goal = Graph('exp2-goal', file_dir='pog_example/iros_2022_exp/exp2/', file_name='goal.json')
    
    # Environment(g_goal)
    start_time = time.time()
    ites, path = test(Searcher, problem=PlanningOnGraphProblem(g_start, g_goal, parking_place=99), pruning=True, max_iter=args.max_iter, max_opt_time=args.max_opt_time)
    
    action_seq = path_to_action_sequence(path, removeRedundant=False)            
    apply_action_sequence_to_graph(g_start, g_goal, action_seq, visualize=args.viewer)

    experiment_time = time.time() - start_time
    experiment_path_length = len(action_seq) - 1

    total_time += experiment_time
    total_path_length += experiment_path_length
    total_experiments += 1

    print("\033[93mSearch time: {:.2f} seconds;".format(experiment_time),
          "Path length: {};".format(experiment_path_length),
          "Iterations: {}.\033[0m".format(ites))

    np.save(data_file, np.array([total_time, total_path_length, total_experiments]))

    experiment_data = np.array([total_experiments, experiment_time, experiment_path_length, ites])

    if os.path.exists(log_file):
        existing_data = np.load(log_file, allow_pickle=True)
        updated_data = np.vstack([existing_data, experiment_data])
    else:
        updated_data = np.array([experiment_data])

    np.save(log_file, updated_data)

    print(updated_data)
    print(f"\033[93mExps: {total_experiments};",
        "Ave time: {:.2f} seconds;".format(total_time/total_experiments),
        "Ave length: {}.\033[0m".format(total_path_length/total_experiments))