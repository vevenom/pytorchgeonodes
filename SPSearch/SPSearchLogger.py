import copy
import os
import numpy as np
import torch
import time
import json

from SPSearch.Logger import Logger

from pytorch3d.transforms import matrix_to_axis_angle

from SPSearch.SPGame import SPGame, TransProposal
from SPSearch.Target import Target

class SPSearchLogger(Logger):
    """
    SP MCTS Logger

    Attributes:
          game: SPGame instance

    """
    def __init__(self, game, target):
        """

        :param game:
        :type game: SPGame
        :param target:
        :type target: Target
        """

        self.game = game
        self.target = target

        self.scores_list = []
        self.tree_depth_list = []
        self.iters = []

        self.best_score = -np.inf
        self.best_time = 0.0
        self.best_prop_seq_str = "_inv"
        self.reset_logger()

        self.start_time = None

    def print_to_log(self, print_str):
        print(print_str)

    def reset_logger(self):
        """
        Reset logging variables

        :return:
        """

        self.scores_list = []
        self.tree_depth_list = []
        self.iters = []

        self.best_score = -np.inf
        self.best_time = 0.0
        # self.best_prop_seq_str = "_inv"
        self.start_time = time.time()

    def export_solution(self, best_props_list):
        """
        Export final solution.

        :param best_props_list: List of best proposals
        :type best_props_list:  List[Proposal]
        :return:
        """

        raise NotImplementedError()

    def log_dvs(self, log_path, iter_num, file_prefix=''):
        """
        Creates csv file with decision variables values and scores. Scores are normalized to be between 0 and 1.

        :param iter_num:
        :param file_prefix:
        :return:
        """

        decision_variables_list = self.game.decision_var_list

        # Structure should be:
        # dv_name, dv_value0, dv_value1, ..., dv_valueN, score

        with open(os.path.join(log_path, file_prefix + '{0}.csv'.format(iter_num)), 'w') as f:

            # find min and max score
            min_score = np.inf
            max_score = -np.inf
            for decision_variable in decision_variables_list:
                for value in decision_variable.values:
                    if value.visits != 0:
                        score = value.score / float(value.visits)
                        if score < min_score:
                            min_score = score
                        if score > max_score:
                            max_score = score

            for decision_variable in decision_variables_list:
                f.write('\n')
                f.write(decision_variable.name + ',')
                for value in decision_variable.values:
                    # round to 2 decimal places
                    value = value.value.detach().cpu().numpy().flatten()[0]
                    if isinstance(value.item(), float):
                        value = np.round(value, 2)
                    value = str(value)

                    f.write(value + ',')
                f.write('\n')
                # f.write(decision_variable.name + ',')
                f.write('score,')

                # sum_scores = 0.0
                # for value in decision_variable.values:
                #     if value.score > 0:
                #         sum_scores += (value.score / float(value.visits))
                #
                # scores_softmax = 0.0
                # for value in decision_variable.values:
                #     if value.visits > 0:
                #         score = value.score / float(value.visits)
                #         print(score)
                #         score = np.exp(score)
                #         print(score)
                #
                #         scores_softmax += score

                for value in decision_variable.values:
                    if value.visits == 0:
                        f.write('0,')
                    else:
                        score = value.score / float(value.visits)
                        # score = score / float(sum_scores)
                        # calculate softmax

                        # score = np.exp(score) / scores_softmax

                        # normalize to 0-1
                        eps = 1e-4
                        score = (score - min_score) / (max_score - min_score + eps)

                        score = np.round(score, 2)

                        f.write(str(score) + ',')
                f.write('\n')

                f.write('visits,')
                for value in decision_variable.values:
                    f.write(str(value.visits) + ',')
                f.write('\n')

    def log_solution_dict(self, input_dict, angle, translation_offset, iter_num, file_prefix=''):
        """
        Log solution dictionary

        :param input_dict:
        :param angle:
        :param translation_offset:
        :param iter_num:
        :param file_prefix:
        :return:
        """

        solution_dict = {}
        input_dict_np = copy.deepcopy(input_dict)
        for k, v in input_dict_np.items():
            input_dict_np[k] = v.detach().cpu().numpy()[0,0].item()
            if isinstance(input_dict_np[k], float):
                input_dict_np[k] = np.round(input_dict_np[k], 3)
        solution_dict['input_dict'] = input_dict_np
        solution_dict['rotation_angle_y'] = np.round(angle.detach().cpu().numpy()[0,1].item(), decimals=3)
        solution_dict['translation_offset'] = [np.round(t.item(), decimals=3) for
                                               t in translation_offset.detach().cpu().numpy()[0]]

        # export as json
        with open(os.path.join(self.target.log_path, file_prefix + '{0}_solution.json'.format(iter_num)), 'w') as f:
            json.dump(solution_dict, f)

    @torch.no_grad()
    def log_final(self, mc_tree):
        """
        Final log performed after the search

        :param mc_tree: final tree
        :type mc_tree: DVTree
        """

        curr_runtime = time.time() - self.start_time

        best_props, leaf_node = mc_tree.get_best_path()
        best_props = leaf_node.prop.get_proposals()

        # parse input_dict
        input_dict, rotation_matrix, translation_offset = self.game.parse_prop_seq(prop_seq=best_props)

        self.target.log_iter_from_input_dict(input_dict, rotation_matrix, translation_offset,
                                             0, file_prefix='final_')

        print('-' * 80)
        # print("Best parameters:")
        # print(input_dict)
        angle = matrix_to_axis_angle(rotation_matrix[:, :3, :3])
        # print('rotation', angle)
        # print('translation', translation_offset)

        print('Best score: %f' % self.best_score)
        print('Final runtime in minutes: %f' % np.round(curr_runtime / 60.0, decimals=3))

        self.log_solution_dict(input_dict, angle, translation_offset, 0, file_prefix='0final_')

        # input_dict_np = copy.deepcopy(input_dict)
        # for k, v in input_dict_np.items():
        #     input_dict_np[k] = v.detach().cpu().numpy()[0,0].item()
        #     if isinstance(input_dict_np[k], float):
        #         input_dict_np[k] = np.round(input_dict_np[k], 3)
        # solution_dict = {}
        # solution_dict['input_dict'] = input_dict_np
        # solution_dict['rotation_angle_y'] = np.round(angle.detach().cpu().numpy()[0,0].item(), decimals=3)
        # solution_dict['translation_offset'] = [np.round(t.item(), decimals=3) for
        #                                        t in translation_offset.detach().cpu().numpy()[0]]
        # print(solution_dict)
        #
        # # export as json
        # with open(os.path.join(self.target.log_path, '0final_solution.json'), 'w') as f:
        #     json.dump(solution_dict, f)

        meta_dict = {
            'best_score': float(self.best_score),
            'best_time': self.best_time,
            'full_time': curr_runtime
        }
        with open(os.path.join(self.target.log_path, '0final_meta.json'), 'w') as f:
            json.dump(meta_dict, f)

    @torch.no_grad()
    def log_mcts(self, iter_num, last_score, last_tree_depth, mc_tree):
        """
        Log MCTS iteration

        :param iter_num:
        :param last_score:
        :param last_tree_depth:
        :param mc_tree:
        :type mc_tree: DVTree
        :return:
        """

        self.scores_list.append(last_score)
        self.tree_depth_list.append(last_tree_depth)
        self.iters.append(iter_num)

        new_best = False
        if last_score > self.best_score:
            new_best = True
            self.best_score = last_score

            curr_runtime = time.time() - self.start_time
            self.best_time = curr_runtime
            # self.best_prop_seq_str = mc_tree.get_best_path()

            # if iter % self.log_config['log_iter'] == 0:
            #     self.print_to_log("Iteration: %d, score: %f, tree depth: %d" % (iter, last_score, last_tree_depth))
            #     self.print_to_log("Best score: %f, best prop seq: %s" % (self.best_score, self.best_prop_seq_str))

            # best_props = mc_tree.get_best_path()[0]
            best_props, leaf_node = mc_tree.get_best_path()

            best_props = leaf_node.prop.get_proposals()

            # parse input_dict
            input_dict, rotation_matrix, translation_offset = self.game.parse_prop_seq(prop_seq=best_props)

            print('-' * 80)
            print("New best parameters")
            print(input_dict)
            print('rotation', matrix_to_axis_angle(rotation_matrix[:, :3, :3]))
            print('translation_offset', translation_offset)

            self.target.log_iter_from_input_dict(input_dict,
                                                 rotation_matrix, translation_offset, iter_num, file_prefix='best_')
            self.target.log_iter_from_input_dict(input_dict,
                                                 rotation_matrix, translation_offset, 0, file_prefix='0best_')
            print('New best score: %f' % self.best_score)
            print('Current runtime in minutes: %f' % np.round(curr_runtime / 60.0, decimals=3))

            self.log_solution_dict(input_dict, matrix_to_axis_angle(rotation_matrix[:, :3, :3]),
                                   translation_offset, iter_num, file_prefix='best_')

            self.log_solution_dict(input_dict, matrix_to_axis_angle(rotation_matrix[:, :3, :3]),
                                   translation_offset, 0, file_prefix='0best_')

            meta_dict = {
                'best_score': float(self.best_score),
                'best_time': self.best_time,
                'full_time': curr_runtime
            }
            with open(os.path.join(self.target.log_path, 'best_{0}_meta.json.json'.format(iter_num)), 'w') as f:
                json.dump(meta_dict, f)

        if iter_num % 100 == 0:
            prop_seq = self.game.prop_seq
            input_dict, rotation_matrix, translation_offset = self.game.parse_prop_seq(prop_seq=prop_seq)

            assert isinstance(prop_seq[len(self.game.decision_var_list)], TransProposal)
            translation_offset = prop_seq[len(self.game.decision_var_list)].get_translation_offset()

            self.target.log_iter_from_input_dict(input_dict,
                                                 rotation_matrix, translation_offset, iter_num, file_prefix='curr_')

        # if (iter % 100 == 0 and iter > 0) or new_best:
        #     graph_path = os.path.join(self.target.log_path, 'graph_%d.png' % iter)
        #     self.drawGraph(mc_tree, graph_path, K=2)
