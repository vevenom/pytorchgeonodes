from abc import abstractmethod
import os
import shutil

class Target(object):
    def __init__(self, log_path):
        super().__init__()

        self.log_path = log_path

        if os.path.exists(log_path):
            shutil.rmtree(log_path)
        os.makedirs(log_path)

    @abstractmethod
    def calculate_cost_from_input_dict(self, input_dict, rotation_matrix, translation_offset=None):
        raise NotImplementedError

    @abstractmethod
    def log_iter_from_input_dict(self, input_dict,
                                 rotation_matrix, translation_offset=None, iter_num=0, file_prefix=''):
        raise NotImplementedError

