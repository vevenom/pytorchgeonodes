# This file was adjusted from MonteScene package (Accessed 2024-27-08):
# https://github.com/vevenom/MonteScene

from abc import ABC
from ordered_set import OrderedSet
from SPSearch.constants.constants import NodesTypes

class Proposal(ABC):
    """
    Class representing proposal. Serves as a base class for task specific proposals.

    Attributes:
        id: Unique string identifier
        type: proposal type
        neighbouring_proposals_set: Set of neighbouring proposals that have preference over other proposals
        incompatible_proposals_set: Set of incompatible proposals

    """
    # TODO Extend your task specific proposal class as a child of Proposal and implement additional functionalities for
    # TODO the child class

    def __init__(self, prop_id, prop_type=NodesTypes.OTHERNODE):
        """

        :param prop_id: unique proposal id
        :type prop_id: str
        :param prop_type: proposal type
        :type prop_type: int
        """

        self.id = str(prop_id)
        self.type = prop_type

        self.neighbouring_proposals_set = OrderedSet()
        self.incompatible_proposals_set = OrderedSet([self])

    def append_neighbour_prop(self, prop):
        """
        Append proposal to the set of neighbours

        :param prop: proposal to be added
        :type prop: Proposal
        :return:
        """
        self.neighbouring_proposals_set = self.neighbouring_proposals_set | OrderedSet([prop])

    def append_incompatible_prop(self, prop):
        """
        Append proposal to the set of incompatible proposals

        :param prop: proposal to be added
        :type prop: Proposal
        :return:
        """
        self.incompatible_proposals_set = self.incompatible_proposals_set | OrderedSet([prop])

    def get_score(self, mode):
        """
        Get proposal score

        :param mode: score mode
        :return: proposal score
        :rtype: float
        """
        return None

    def get_visits(self):
        """
        Get proposal visits

        :return: proposal visits
        :rtype: int
        """
        return None

    def update(self, score, n_visits):
        """
        Update proposal score.

        This method is a stub and should be implemented in the child class if needed.

        Args:
            score:
            n_visits:

        Returns:

        """
        pass
