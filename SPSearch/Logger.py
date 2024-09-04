# This file was adjusted from MonteScene package (Accessed 2024-27-08):
# https://github.com/vevenom/MonteScene

from abc import ABC, abstractmethod
from typing import List

from SPSearch.Proposal.Prop import Proposal


class Logger(ABC):
    """
    Abstract class representing Logger. Serves as a base class for task specific loggers

    Attributes:
          game: ProposalGame instance

    """
    def __init__(self, game):
        """

        :param game: ProposalGame instance
        :type game: ProposalGame
        """

        self.game = game

    def print_to_log(self, print_str):
        print(print_str)

    @abstractmethod
    def reset_logger(self):
        """
        Reset logging variables

        :return:
        """

        # TODO Reset variables that should be tracked
        raise NotImplementedError()

    @abstractmethod
    def export_solution(self, best_props_list):
        """
        Export final solution.

        :param best_props_list: List of best proposals
        :type best_props_list:  List[Proposal]
        :return:
        """

        # TODO Export solution. Should be called from log_final
        raise NotImplementedError()