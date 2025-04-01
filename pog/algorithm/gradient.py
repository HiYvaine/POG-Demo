import numpy as np
from numpy.random import rand, randn

from pog.algorithm.utils import *
from pog.graph.graph import Graph

from pog.algorithm.params import STEP_SIZE, MAX_ITER, EPS_THRESH, TEMP_COEFF, NUM_CYCLES_EPS, CHECK_COLLISION_IK

def gradient_descent(
    objective,
    sg: Graph,
    node_id=None,
    random_start=False,
    verbose=False,
):

    if node_id is not None:
        pose = sg.getPose(edge_id=node_id)
    else:
        pose = sg.getPose()
    object_pose_dict, bounds = gen_bound(pose)

    # generate an initial point
    if random_start:
        best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
        sg.setPose(arr2pose(best, object_pose_dict, pose))
    else:
        best = pose2arr(pose, object_pose_dict, [])

    # evaluate the initial point
    best_eval, step_direction = objective[0](best, object_pose_dict, sg)
    # current working solution
    curr, curr_eval = best, best_eval
    # Remove the pose-optimizing object from worldcfg to reduce IK request frequency
    if CHECK_COLLISION_IK:
        worldcfg = sg.genWorldCfg(exclude_object_ids={node_id[0]})
        ik_solver = sg.CreateIKSolver(worldcfg)
    # run the algorithm
    i = 0
    best_eval_arr = []
    history = []
    t = 1
    skip_candidates = 0
    while i < MAX_ITER:
        i += 1
        # take a step
        gauss_noise = t * randn(len(curr))  # Gaussian
        t = t * TEMP_COEFF
        candidate = curr + (step_direction + gauss_noise) * STEP_SIZE

        # print(step_direction)

        # evaluate candidate point
        pose = arr2pose(candidate, object_pose_dict, pose)
        sg.setPose(pose)
        candidate_eval, temp_step_direction = objective[0](candidate,
                                                           object_pose_dict,
                                                           sg)

        # difference between candidate and current point evaluation
        diff = candidate_eval - curr_eval

        # calculate metropolis acceptance criterion

        # check for new best solution
        if candidate_eval < best_eval:

            # TODO: what if all candidates have no valid IK solution?
            if CHECK_COLLISION_IK:
                # skip if the candidate has no valid IK solution
                object_pose = sg.transform_matrix_to_list(sg.global_transform[node_id[0]])
                ik_result = sg.checkIK(object_id=node_id[0], ik_solver=ik_solver, object_pose=object_pose)
                if not ik_result:
                    skip_candidates += 1
                    continue
            
            # store new best point
            best, best_eval = candidate, candidate_eval
            # report progress
            if verbose:
                best_tmp = [float("{:.4f}".format(x)) for x in list(best)]
                print('>{} f({}) = {:.4f}'.format(i, best_tmp, best_eval))

            best_eval_arr.append(diff)
            if len(best_eval_arr) > NUM_CYCLES_EPS and (abs(
                    np.array(best_eval_arr[-NUM_CYCLES_EPS:])) <
                                                        EPS_THRESH).all():
                break

        if diff < 0:
            # store the new current point
            curr, curr_eval = candidate, candidate_eval
            step_direction = temp_step_direction

    if skip_candidates > 0:
        print(f"(skip {skip_candidates} candidate poses for object_{node_id[0]} due to IK failure)")
    return best, best_eval
