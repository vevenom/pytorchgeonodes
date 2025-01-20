import torch
import numpy as np
import time
import json
import os

from pytorch3d.transforms import matrix_to_axis_angle

from SPSearch.SPSearchLogger import SPSearchLogger

class GeneticLogger(SPSearchLogger):
    def __init__(self, game, target, settings):
        super().__init__(game, target)

        self.best_loss = np.inf
        self.start_time = time.time()

        self.settings = settings

    def print_time(self):
        curr_runtime = time.time() - self.start_time
        print('Current runtime in minutes: %f' % np.round(curr_runtime / 60.0, decimals=3))

    def log_iteration(self, best_individual, best_loss, current_population, curr_gen):

        # Logging
        with torch.no_grad():
            best_prop_seq = best_individual.prop_seq
            input_dict, rotation_matrix, translation_offset = self.game.parse_prop_seq(
                prop_seq=best_prop_seq)

            # print('-' * 80)
            # print("New best parameters")
            # print(input_dict)
            # print('rotation', matrix_to_axis_angle(rotation_matrix[:, :3, :3]))
            # print('translation_offset', translation_offset)

            self.game.target.log_iter_from_input_dict(input_dict,
                                                      rotation_matrix, translation_offset, curr_gen,
                                                      file_prefix='best_')
            self.game.target.log_iter_from_input_dict(input_dict,
                                                      rotation_matrix, translation_offset, 0,
                                                      file_prefix='0best_')

            curr_runtime = time.time() - self.start_time

            print('Current runtime in minutes: %f' % np.round(curr_runtime / 60.0, decimals=3))

            self.log_solution_dict(input_dict, matrix_to_axis_angle(
                rotation_matrix[:, :3, :3]),
                                     translation_offset, curr_gen, file_prefix='best_')

            self.log_solution_dict(input_dict,
                                     matrix_to_axis_angle(rotation_matrix[:, :3, :3]),
                                     translation_offset, 0, file_prefix='0best_')

            meta_dict = {
                'best_loss': float(best_loss),
                'best_time': curr_runtime,
                'full_time': curr_runtime
            }
            with open(os.path.join(
                    self.game.target.log_path, 'best_{0}_meta.json'.format(curr_gen)),
                    'w') as f:
                json.dump(meta_dict, f)

            with open(os.path.join(
                    self.game.target.log_path, '0best_meta.json'),
                    'w') as f:
                json.dump(meta_dict, f)

            # self.log_population(current_population, 'best_{0}_population.jpg'.format(curr_gen))

    def log_population(self, current_population, filename):
        num_col = 4
        num_row = len(current_population) // 4
        import matplotlib.pyplot as plt

        resolution = 5
        f, axs = plt.subplots(num_row, num_col, figsize=(num_col * resolution, num_row * resolution))
        for individual_ind, logged_individual in enumerate(current_population):
            input_dict, rotation_matrix, translation_offset = self.game.parse_prop_seq(
                prop_seq=logged_individual.prop_seq)

            obj_mesh = self.game.target.calculate_mesh_from_input_dict(input_dict,
                                                             rotation_matrix)
            rendered_individual = self.game.target.render_image(
                obj_mesh, scene_frame_ind=0, transparent_overlay=True)

            curr_row = individual_ind // num_col
            curr_col = individual_ind % num_col
            # plt.subplot(num_row, num_col, individual_ind + 1, figsize=(3, 3))
            # print(axs)
            # print(axs[individual_ind])
            axs[curr_row][curr_col].imshow(rendered_individual)
            axs[curr_row][curr_col].axis('off')

        # turn off axis

        # plt.show()
        pop_vis_path = os.path.join(self.game.target.log_path, filename)
        plt.savefig(pop_vis_path, bbox_inches='tight', pad_inches=0)
        plt.close()





