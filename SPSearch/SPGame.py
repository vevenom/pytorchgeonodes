import torch
from SPSearch.ProposalGame import ProposalGame
from SPSearch.Proposal import Proposal
from SPSearch.constants.constants import NodesTypes

from SPSearch.Target import Target
from SPSearch.DecisionVariable import DecisionVariable, RotationDecisionVariable
from PytorchGeoNodes.GeometryNodes import GeometryNodes


class DVProposal(Proposal):
    """
    Proposal class that contains a decision value

    Attributes:
        decision_value: decision value
    """

    def __init__(self, id, decision_value, prop_type):
        super().__init__(id, prop_type)

        self.decision_value = decision_value

    def get_decision_value(self):
        return self.decision_value

    def get_score(self, mode):
        return self.decision_value.get_score(mode)

    def get_visits(self):
        return self.decision_value.get_visits()

    def update(self, score, n_visits):
        self.decision_value.update(score, n_visits)

    def get_params_list(self):
        return [self.decision_value.get_value()] if self.decision_value.get_value().requires_grad else []


class TransProposal(Proposal):
    """
    Proposal class that contains translation offset

    Attributes:
        translation_offset: translation offset
    """

    def __init__(self, id, translation_offset, prop_type):
        super().__init__(id, prop_type)

        self.translation_offset = translation_offset

    def get_translation_offset(self):
        return self.translation_offset

    def get_params_list(self):
        return [self.translation_offset] if self.translation_offset.requires_grad else []

class SPGame(ProposalGame):
    def __init__(self, args_list):
        self.geometry_nodes = None  # type: GeometryNodes
        self.decision_var_list = None  # type: [DecisionVariable]
        self.target = None  # type: Target

        super().__init__(args_list)

        assert self.decision_var_list is not None

    def initialize_game(self, decision_var_list: [DecisionVariable], target: Target):
        """
        Initialize game-specific attributes.

        :return:
        """

        self.decision_var_list = decision_var_list
        self.target = target

    def generate_proposals(self):
        pass

    def restart_game(self):
        self.pool_curr = [] # This game does not use a pool
        self.prop_seq = []

    def set_state(self, pool_curr, prop_seq):
        self.prop_seq = prop_seq

    def step(self, prop):
        """
        Take a single step in the game.

        :param prop:
        :type prop: Proposal
        :return:
        """

        if prop.type not in NodesTypes.SPECIAL_NODES_LIST:
            self.prop_seq.append(prop)
        elif prop.type == NodesTypes.ENDNODE:
            # prop is AccumulatorProp

            self.prop_seq = prop.get_proposals()

    def calc_score_from_proposals(self, prop_seq=None, props_optimizer=None):
        """
        Calculate score from proposals

        :param prop_seq: Sequence of proposals. If None, uses self.prop_seq instead
        :type prop_seq: List[Proposal]
        :param props_optimizer: Optimizer. Enables optimization during score calculation. If None, optimization
        step is not performed
        :type props_optimizer: PropsOptimizer

        :return: score
        :rtype: torch.Tensor
        """

        loss = self.calc_loss_from_proposals(prop_seq)
        score = self.convert_loss_to_score(loss)

        return score

    def parse_prop_seq(self, prop_seq):
        """
        Parse prop_seq into input_params_dict and translation_offset

        :param prop_seq: sequence of proposals
        :type prop_seq: List[Proposal]
        :return: input_params_dict, translation_offset
        :rtype: dict, torch.nn.Parameter
        """

        if prop_seq is None:
            prop_seq = self.prop_seq
        assert len(prop_seq)

        input_params_dict = {}

        rotation_matrix = None  # type: torch.Tensor
        for dv_ind, dv in enumerate(self.decision_var_list):
            if isinstance(dv, RotationDecisionVariable):
                value = prop_seq[dv_ind].get_decision_value()

                if dv.add_value:
                    rotation_matrix = torch.bmm(rotation_matrix, dv.convert_value(value))
                else:
                    rotation_matrix = dv.get_rotation_matrix_from_y_angle(dv.convert_value(value))
            else:
                value = prop_seq[dv_ind].get_decision_value()
                if dv.add_value:
                    input_params_dict[dv.name] += dv.convert_value(value)
                else:
                    input_params_dict[dv.name] = dv.convert_value(value)

        assert isinstance(prop_seq[len(self.decision_var_list)], TransProposal)
        translation_offset = prop_seq[len(self.decision_var_list)].get_translation_offset()

        return input_params_dict, rotation_matrix, translation_offset

    def calc_loss_from_proposals(self, prop_seq=None):

        input_params_dict, rotation_matrix, translation_offset = self.parse_prop_seq(prop_seq)

        loss = self.target.calculate_cost_from_input_dict(input_params_dict,
                                                          rotation_matrix=rotation_matrix,
                                                          translation_offset=translation_offset)

        return loss

    @torch.no_grad()
    def convert_loss_to_score(self, loss):
        loss = loss.detach().cpu().numpy()
        return -loss