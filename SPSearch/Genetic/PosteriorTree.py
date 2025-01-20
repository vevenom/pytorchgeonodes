from copy import deepcopy
from typing import List
import numpy as np
import copy
import random
import torch

from SPSearch.DecisionVariable import DecisionVariable
from SPSearch.DecisionVariable import DecisionValue

class PosteriorNode(object):
    def __init__(self, decision_value: DecisionValue, parent, prior=1.0):
        self.decision_value = decision_value
        # None if root

        self.parent = parent # type: PosteriorNode
        self.children = [] # type: List[PosteriorNode]
        # None if root

        self.score = -np.inf
        self.prior = prior

        # self.score = 0.
        self.visits = 0

        self.tree_depth = 0 if parent is None else parent.tree_depth + 1

    def update(self, value, score):

        # max
        if score > self.score:
            self.score = score
            # if self.decision_value.value.dtype == torch.float32:
            #     print('check')
            #     print(self.decision_value.value)
            #     print(value.value)
            #     print(self.decision_value.value - value.value)
            #     assert (self.decision_value.value - value.value).sum() == 0

            self.decision_value = deepcopy(value)


        # avg
        # self.score += score.item()

        self.visits += 1


    def get_score(self, eps=1e-4):

        # max
        return self.score

        # avg
        # avg_score = self.score / (self.visits + eps)
        # return avg_score

    def append_child(self, child):
        self.children.append(child)

    def get_children_distribution(self, max_norm, min_norm,
                                  exploit_lambda = 100.0, explore_lambda=0.0, eps=1e-4):
        # children_distribution = np.zeros(len(self.children))

        # # Based on score only
        # exploit_terms = []
        # for c_ind, child in enumerate(self.children):
        #     # if child.score == -np.inf:
        #     #     exploit_terms.append(1.0)
        #     # else:
        #     #     exploit_terms.append(child.score)
        #     exploit_terms.append(child.score)
        # exploit_terms = np.array(exploit_terms)
        # exploit_terms[exploit_terms == -np.inf] = np.clip(exploit_terms.max(), a_min=-10000.0, a_max=0.0)
        #
        # # exploit_terms /= np.sum(exploit_terms + eps)
        # exploit_terms = (exploit_terms - exploit_terms.min()) / (exploit_terms.max() - exploit_terms.min() + eps)
        # exploit_terms = np.clip(exploit_terms, a_min=0.1, a_max=None)
        #
        # # Softmax
        # children_distribution = np.exp(exploit_terms) / np.sum(np.exp(exploit_terms))
        #
        # return children_distribution

        # Based on exploration

        children_distribution = np.zeros(len(self.children))
        exploit_terms = []
        total_visits = 0
        for c_ind, child in enumerate(self.children):
            if child.score == -np.inf:
                exploit_terms.append(max_norm)
            else:
                exploit_terms.append(child.score)
            total_visits += child.visits

        if total_visits == 0:
            return np.ones(len(self.children)) / len(self.children)

        exploit_terms = np.array(exploit_terms)
        # exploit_terms /= np.sum(exploit_terms + eps)

        # exploit_terms = (exploit_terms - min_norm) / (max_norm - min_norm)

        # exploit_terms = (exploit_terms - exploit_terms.min()) / (exploit_terms.max() - exploit_terms.min() + 1e-3)

        for c_ind, child in enumerate(self.children):
            exploit_term = exploit_terms[c_ind]
            if not child.visits:
                explore_term = child.prior
            else:
                # explore_term = np.sqrt(2 * np.log(self.visits + eps) / (child.visits + eps))
                # explore_term = np.sqrt(2 * np.log(1 - child.prior + eps) / (child.visits + eps))
                # explore_term = np.sqrt(2 * np.log(2 - child.prior + eps) / (child.visits + eps))

                # explore_term = child.prior / (child.visits + eps)
                explore_term = (1. + child.prior) * np.sqrt(2 * np.log(total_visits + eps) / (child.visits + eps))
                # explore_term = (1. + child.prior) / (child.visits + eps)

            c_score = exploit_lambda * exploit_term + explore_lambda * explore_term
            children_distribution[c_ind] = c_score + eps

        # children_distribution = children_distribution

        # Softmax
        children_distribution = (np.exp(children_distribution) / np.sum(np.exp(children_distribution)) +
                                 1. / children_distribution.shape[0])
        # children_distribution = np.clip(children_distribution, a_min=0.10, a_max=None)

        return children_distribution

    def get_children_scores(self):

        score_terms = []
        for c_ind, child in enumerate(self.children):
            score_terms.append(child.score)
        score_terms = np.array(score_terms)

        return score_terms

    def get_children_visits(self):
        visits_list = []
        for c_ind, child in enumerate(self.children):
            visits_list.append(child.visits)
        visits_list = np.array(visits_list)

        return visits_list


class PosteriorTree(object):
    def __init__(self, decision_variables: List[DecisionVariable], tree_settings,
                 prior_distributions=None):
        #NOTE: Genetic algorithm with posterior tree distribution converges faster but is more prone to local minima.
        # Use with caution!

        if prior_distributions is None:
            prior_distributions = []
        self.prior_distributions = prior_distributions

        self.decision_variables = decision_variables

        self.root = PosteriorNode(None, None)
        self.curr_node_ptr = self.root
        self.nodes = [self.root]

        self.min_score = np.inf
        self.max_score = -np.inf

        self.shared_children_nodes_dict = None
        if tree_settings.share_children_nodes:
            self.shared_children_nodes_dict = {}
            for dv in self.decision_variables:
                self.shared_children_nodes_dict[dv.name] = []

        self.init_children(self.root)


    def generate_tree(self):
        curr_nodes = [self.root]
        for dv in self.decision_variables:
            next_nodes = []
            for cn in curr_nodes:
                for dv_value in dv.values:
                    new_node = PosteriorNode(copy.deepcopy(dv_value), cn)

                    next_nodes.append(new_node)
                    self.nodes.append(new_node)
            curr_nodes = next_nodes

    def init_children(self, node):
        dv =  self.decision_variables[node.tree_depth]

        assert not node.children

        if self.shared_children_nodes_dict is not None:
            if self.shared_children_nodes_dict[dv.name]:
                node.children.extend(self.shared_children_nodes_dict[dv.name])
            else:
                for dvv_ind, dv_value in enumerate(dv.values):
                    if self.prior_distributions and self.prior_distributions[node.tree_depth] is not None:
                        new_node = PosteriorNode(copy.deepcopy(dv_value), node,
                                                 self.prior_distributions[node.tree_depth][dvv_ind].item())
                    else:
                        new_node = PosteriorNode(copy.deepcopy(dv_value), node)
                    node.append_child(new_node)
                    self.nodes.append(new_node)
                    self.shared_children_nodes_dict[dv.name].append(new_node)
        else:
            for dvv_ind, dv_value in enumerate(dv.values):
                if self.prior_distributions and self.prior_distributions[node.tree_depth] is not None:
                    new_node = PosteriorNode(copy.deepcopy(dv_value), node,
                                             self.prior_distributions[node.tree_depth][dvv_ind].item())
                else:
                    new_node = PosteriorNode(copy.deepcopy(dv_value), node)
                node.append_child(new_node)
                self.nodes.append(new_node)



    def update_nodes(self, score, decision_values):

        if score < self.min_score:
            self.min_score = score
        if score > self.max_score:
            self.max_score = score

        curr_node = self.root
        curr_tree_depth = 0
        while curr_tree_depth < len(self.decision_variables):
            curr_value = decision_values[curr_tree_depth]

            if not curr_node.children:
                self.init_children(curr_node)

            min_dist = np.inf
            best_cn_match = None
            for cn_ind, cn in enumerate(curr_node.children):
                cn_value = cn.decision_value
                dist = (cn_value.value.to(torch.float32) - curr_value.value.to(torch.float32)) ** 2
                if dist < min_dist:
                    min_dist = dist
                    best_cn_match = cn

            best_cn_match.update(curr_value, score)
            # if min_dist > 0:
            #     print(self.decision_variables[curr_tree_depth].name)
            #     print(best_cn_match.decision_value.value.to(torch.float32))
            #     print(curr_value.value.to(torch.float32))
            #     print(min_dist)
            #     input()

            curr_node = best_cn_match
            curr_tree_depth += 1


    def select_next_node(self):

        if not self.curr_node_ptr.children:
            self.init_children(self.curr_node_ptr)

        children_node_distr = self.curr_node_ptr.get_children_distribution(
            max_norm=self.max_score, min_norm=self.min_score)

        selected_cn = random.choices(self.curr_node_ptr.children, weights=children_node_distr)[0]

        self.curr_node_ptr = selected_cn

        return selected_cn

    def select_value(self, dec_value1, eps=1e-4):

        if not self.curr_node_ptr.children:
            self.init_children(self.curr_node_ptr)

        min_dist1 = np.inf
        best_cn_match1 = None
        best_cn_match1_ind = 0
        for cn_ind, cn in enumerate(self.curr_node_ptr.children):
            cn_value = cn.decision_value
            dist1 = (cn_value.value.to(torch.float32) - dec_value1.value.to(torch.float32)) ** 2
            if dist1 < min_dist1:
                min_dist1 = dist1
                best_cn_match1 = cn

        selected_cn = best_cn_match1
        self.curr_node_ptr = selected_cn

        return selected_cn.decision_value

    def select_between_two_values(self, dec_value1, dec_value2, eps=1e-4):
        if not self.curr_node_ptr.children:
            self.init_children(self.curr_node_ptr)
        children_node_distr = self.curr_node_ptr.get_children_distribution(max_norm=self.max_score,
                                                                           min_norm=self.min_score)

        min_dist1 = np.inf
        best_cn_match1 = None
        best_cn_match1_ind = 0
        min_dist2 = np.inf
        best_cn_match2 = None
        best_cn_match2_ind = 0
        for cn_ind, cn in enumerate(self.curr_node_ptr.children):
            cn_value = cn.decision_value
            dist1 = (cn_value.value.to(torch.float32) - dec_value1.value.to(torch.float32)) ** 2
            if dist1 < min_dist1:
                min_dist1 = dist1
                best_cn_match1 = cn

            dist2 = (cn_value.value.to(torch.float32) - dec_value2.value.to(torch.float32)) ** 2
            if dist2 < min_dist2:
                min_dist2 = dist2
                best_cn_match2 = cn

        if best_cn_match1 == best_cn_match2:
            selected_cn = best_cn_match1
        else:
            children_node_distr = children_node_distr[[best_cn_match1_ind, best_cn_match2_ind]]
            children_node_distr /= (children_node_distr.sum() + 1e-4)

            selected_cn = random.choices([best_cn_match1, best_cn_match2], weights=children_node_distr)[0]

        self.curr_node_ptr = selected_cn

        return selected_cn.decision_value

    def reset_current_node_ptr(self):
        self.curr_node_ptr = self.root

    def print_posterior(self):
        curr_node = self.root
        curr_tree_depth = 0
        while curr_tree_depth < len(self.decision_variables):
            if not curr_node.children:
                self.init_children(curr_node)

            children_node_distr = curr_node.get_children_distribution(max_norm=self.max_score,
                                                                      min_norm=self.min_score)
            children_node_scores = curr_node.get_children_scores()
            children_node_visits = curr_node.get_children_visits()

            print("Node name:", self.decision_variables[curr_tree_depth].name)
            print("  --Distribution: ", np.round(children_node_distr, decimals=2))
            print("  --Scores: ", np.round(children_node_scores, decimals=2))
            print("  --Visits: ", children_node_visits)
            # input()

            curr_node_ind = np.argmax(children_node_scores)
            curr_node = curr_node.children[curr_node_ind]
            curr_tree_depth += 1

    def get_distributions(self):
        curr_node = self.root
        curr_tree_depth = 0

        dv_distributions = {}
        while curr_tree_depth < len(self.decision_variables):
            if not curr_node.children:
                self.init_children(curr_node)

            children_node_distr = curr_node.get_children_distribution(max_norm=self.max_score,
                                                                      min_norm=self.min_score)
            children_node_scores = curr_node.get_children_scores()

            dv_distributions[self.decision_variables[curr_tree_depth].name] = children_node_distr

            curr_node_ind = np.argmax(children_node_scores)
            curr_node = curr_node.children[curr_node_ind]
            curr_tree_depth += 1

        return dv_distributions