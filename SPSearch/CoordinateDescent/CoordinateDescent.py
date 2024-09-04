import torch
import numpy as np
import random
import time
import os
import json
import copy

from pytorch3d.transforms import matrix_to_axis_angle

from SPSearch.constants.constants import NodesTypes

from SPSearch.SyntheticTarget.SyntheticTarget import SyntheticTarget
from SPSearch.SPGame import DVProposal, TransProposal

class CoordinateDescent(object):
    def __init__(self, game, scene_reconstructions_path, settings):
        self.target = game.target  # type: SyntheticTarget
        self.game = game
        self.scene_reconstructions_path = scene_reconstructions_path

        self.settings = settings

    def reconstruct_scene(self, logger, return_best_loss_per_value=False):

        decision_variables_list = self.game.decision_var_list

        # Made to work with synthetic target. Need to change to work with SPGame, for example translation offset

        # for every decision variable, select random value and set key

        best_prop_seq = []
        for dv in decision_variables_list:
            value = random.choice(dv.values)

            # print(dv.name, ',', value.value)

            new_proposal = DVProposal(
                    dv.name + '_',
                    value,
                    prop_type=NodesTypes.OTHERNODE)
            best_prop_seq.append(new_proposal)
        trans_proposal = TransProposal('OBJ_Translation',
                                       torch.nn.Parameter(
                                           torch.tensor([[0., 0., 0.]],
                                                        requires_grad=False,
                                                        device=self.target.device)),
                                       prop_type=NodesTypes.OTHERNODE)
        best_prop_seq.append(trans_proposal)
        best_loss_per_value = [[np.inf for _ in dv.values] for dv in decision_variables_list]

        best_loss = np.inf
        # for every iteration of coordinate descent, loop through decision variables and and try all values

        curr_step = 0
        start_time = time.time()
        num_loops = self.settings.num_loops
        for i in range(num_loops):
            new_best_solution_found = False
            for dv_ind, dv in enumerate(decision_variables_list):
                print(dv.name, ', curr_step:, ', curr_step)

                tmp_prop_seq = copy.deepcopy(best_prop_seq)

                # try all values of decision variable and evaluate loss
                for val_ind, value in enumerate(dv.values):
                    tmp_prop_seq[dv_ind].decision_value = value

                    loss = self.game.calc_loss_from_proposals(tmp_prop_seq)
                    if loss < best_loss_per_value[dv_ind][val_ind]:
                        best_loss_per_value[dv_ind][val_ind] = loss.item()

                    curr_step += 1

                    with torch.no_grad():
                        if loss < best_loss:
                            new_best_solution_found = True
                            best_loss = loss.item()
                            best_prop_seq = copy.deepcopy(tmp_prop_seq)

            if new_best_solution_found:
                if self.settings.refinement.use_refinement:
                    best_prop_seq, best_loss = self.optimize(
                        prop_seq=best_prop_seq,
                        optimize_threshold=5.0)
                with torch.no_grad():

                    input_dict, rotation_matrix, translation_offset = self.game.parse_prop_seq(
                        prop_seq=best_prop_seq)

                    print('-' * 80)
                    print("New best parameters")
                    print(input_dict)
                    print('rotation', matrix_to_axis_angle(rotation_matrix[:, :3, :3]))
                    print('translation_offset', translation_offset)

                    self.target.log_iter_from_input_dict(input_dict,
                                                         rotation_matrix, translation_offset, curr_step,
                                                         file_prefix='best_')
                    self.target.log_iter_from_input_dict(input_dict,
                                                         rotation_matrix, translation_offset, 0,
                                                         file_prefix='0best_')

                    curr_runtime = time.time() - start_time

                    print('New best lest: %f' % best_loss)
                    print('Current runtime in minutes: %f' % np.round(curr_runtime / 60.0, decimals=3))

                    logger.log_solution_dict(input_dict, matrix_to_axis_angle(
                        rotation_matrix[:, :3, :3]),
                                             translation_offset, curr_step, file_prefix='best_')

                    logger.log_solution_dict(input_dict,
                                             matrix_to_axis_angle(rotation_matrix[:, :3, :3]),
                                             translation_offset, 0, file_prefix='0best_')

                    meta_dict = {
                        'best_loss': float(best_loss),
                        'best_time': curr_runtime,
                        'full_time': curr_runtime
                    }
                    with open(os.path.join(self.target.log_path, 'best_{0}_meta.json.json'.format(curr_step)),
                              'w') as f:
                        json.dump(meta_dict, f)

        # assert False
        if return_best_loss_per_value:
            return best_loss, best_loss_per_value
        else:
            return best_loss

    def optimize(self, prop_seq, optimize_threshold):
        best_loss = np.inf

        parameters_to_train = []
        for prop in prop_seq:
            prop_params_list = prop.get_params_list()
            for prop in prop_params_list:
                if any(prop is train_param for train_param in parameters_to_train):
                    continue
                parameters_to_train.append(prop)

        assert parameters_to_train

        optimizer = torch.optim.Adam(parameters_to_train, lr=self.settings.refinement.optimizer_lr)

        # best_values = []
        best_props = prop_seq
        for iter in range(self.settings.refinement.optimize_steps):
            optimizer.zero_grad()

            curr_loss = self.game.calc_loss_from_proposals(prop_seq)

            if curr_loss < best_loss:
                best_loss = curr_loss

                best_props = copy.deepcopy(prop_seq)

            if curr_loss > best_loss + 0.005:
                break

            if curr_loss > optimize_threshold:
                break

            curr_loss.backward()
            optimizer.step()

            with torch.no_grad():
                for dv_ind, dv in enumerate(self.game.decision_var_list):
                    if isinstance(prop_seq[dv_ind], DVProposal):
                        if prop_seq[dv_ind].get_decision_value().get_value().requires_grad:
                            if dv.normalized_values:
                                prop_seq[dv_ind].get_decision_value().value.clamp_(
                                    min=-1,
                                    max=1)
                            else:
                                prop_seq[dv_ind].get_decision_value().value.clamp_(
                                    min=dv.valid_range[0],
                                    max=dv.valid_range[1])

        with (torch.no_grad()):
            for p_ind, p in enumerate(prop_seq):
                if isinstance(p, DVProposal):
                    if p.get_decision_value().get_value().requires_grad:
                        p.decision_value.value[:] = \
                            best_props[p_ind].get_decision_value().value.clone().detach()

                elif isinstance(p, TransProposal):
                    if p.translation_offset.requires_grad:
                        p.translation_offset[:] = \
                            best_props[p_ind].get_translation_offset().clone().detach()

        return best_props, best_loss




