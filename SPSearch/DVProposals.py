import torch

class DVProposal(torch.nn.Module):
    """
    Proposal class that contains a decision value

    Attributes:
        decision_value: decision value
    """

    def __init__(self, id, decision_value, prop_type):
        super().__init__()

        self.decision_value = decision_value

    def get_decision_value(self):
        return self.decision_value

    def get_params_list(self):
        return [self.decision_value.get_value()] if self.decision_value.get_value().requires_grad else []

class TransProposal(torch.nn.Module):
    def __init__(self, id, translation_offset, prop_type):
        super().__init__()

        self.translation_offset = translation_offset

    def get_translation_offset(self):
        return self.translation_offset

    def get_params_list(self):
        return [self.translation_offset] if self.translation_offset.requires_grad else []