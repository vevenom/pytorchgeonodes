# This file was adjusted from MonteScene package (Accessed 2024-27-08):
# https://github.com/vevenom/MonteScene

from abc import ABC, abstractmethod
import torch
from ordered_set import OrderedSet
from typing import List

from SPSearch.constants.constants import *
class ProposalGame(ABC):
    """
    Abstract class representing Game. It serves as a base class for implementing other Games.

        Attributes:
            prop_pool_set: set of all proposals in the Game
            pool_curr: remaining pool of proposals
            prop_seq: currently selected pool of proposals
    """
    def __init__(self, args_list):

        self.pool_curr = OrderedSet([])
        self.prop_seq = []

        self.initialize_game(*args_list)
        self.prop_pool_set = self.generate_proposals()
        self.restart_game()

    def __str__(self):
        return "Game()"

    @abstractmethod
    def initialize_game(self, **args):
        """
        Initialize game-specific attributes.

        :return:
        """

        # TODO Initialize game attributes
        raise NotImplementedError()

    @abstractmethod
    def generate_proposals(self):
        """
        Generate proposals for the game

        :return: a set of generated proposals
        :rtype: OrderedSet
        """

        # TODO Generate Proposals. Use Proposal class as the Base class and implement your own task-specific Proposals
        # TODO Handle neighbors and incompatible set generation
        raise NotImplementedError()

    def restart_game(self):
        """
        Restart game

        :return:
        """
        self.pool_curr = self.prop_pool_set.copy()
        self.prop_seq = []

    def set_state(self, pool_curr, prop_seq):
        """

        :param pool_curr: a set of remaining proposals
        :type pool_curr: OrderedSet
        :param prop_seq: a set of already selected proposals
        :type prop_seq: List

        :return:
        """
        self.pool_curr = pool_curr
        self.prop_seq = prop_seq

    def get_state(self):
        """
        Return current state of the game.

        :return: available pool, selected proposals
        :rtype: OrderedSet, List
        """
        return self.pool_curr, self.prop_seq

    def step(self, prop):
        """
        Take a single step in the game.

        :param prop:
        :type prop: Proposal
        :return:
        """
        assert prop in self.pool_curr or prop.type in NodesTypes.SPECIAL_NODES_LIST, \
            "Prop %s is not in current pool %s" % (prop.id, str([p.id for p in self.pool_curr]))

        candidate_pool = self.pool_curr
        compatible_props = candidate_pool - prop.incompatible_proposals_set

        if prop.type not in NodesTypes.SPECIAL_NODES_LIST:
            self.prop_seq.append(prop)

        self.pool_curr = compatible_props

    @abstractmethod
    def calc_score_from_proposals(self, prop_seq=None):
        """
        Calculate score from proposals

        :param prop_seq: Sequence of proposals. If None, uses self.prop_seq instead
        :type prop_seq: List[Proposal]

        :return: score
        :rtype: torch.Tensor
        """

        # TODO Implement score calculation
        # TODO Handle prop_seq is None
        # if prop_seq is None:
        #     prop_seq = self.prop_seq
        # TODO Handle empty prop_seq


        raise NotImplementedError()

    @abstractmethod
    def calc_loss_from_proposals(self, prop_seq=None):
        """
        Calculate loss from proposals

        :param prop_seq: Sequence of proposals. If None, uses self.prop_seq instead
        :type prop_seq: List[Proposal]

        :return: loss
        :rtype: torch.Tensor
        """

        # TODO Implement loss calculation. Required only if optimization is used
        # TODO Handle prop_seq is None
        # if prop_seq is None:
        #     prop_seq = self.prop_seq
        # TODO Handle empty prop_seq

        raise NotImplementedError()

    @abstractmethod
    def convert_loss_to_score(self, loss):
        """
        Convert loss to score

        :param loss: loss value
        :type loss: torch.Tensor

        :return: score
        :rtype: torch.Tensor
        """

        # TODO Convert loss to score

        raise NotImplementedError()

    def refine_proposals(self, props_optimizer, prop_seq=None):
        """
        Refine proposals using optimizer

        :param prop_seq:
        :param props_optimizer:
        :return:
        """

        raise NotImplementedError()

