# This file was adjusted from MonteScene package (Accessed 2024-27-08):
# https://github.com/vevenom/MonteScene

from torch.optim import Adam


class PropsOptimizer():
    """
    Class defining proposals optimizer.

    Attributes:
        id: Identifier string of the optimizer
        optimizer: optimizer instance
        optimize_steps: Number of steps to be performed every time the optimizer is called
    """
    def __init__(self, game, settings):
        """

        :param game: Game instance
        :type game: Game.Game
        :param settings: settings containing rules for building the tree
        """
        super().__init__()
        self.id = str([prop.id for prop in game.prop_seq])

        parameters_to_train = []
        for prop in game.prop_seq:
            prop_params_list = prop.get_params_list()
            for prop in prop_params_list:
                if any(prop is train_param for train_param in parameters_to_train):
                    continue
                parameters_to_train.append(prop)

        if parameters_to_train:
            self.optimizer = Adam(parameters_to_train, lr=settings.mcts.refinement.optimizer_lr)

        self.optimize_steps = settings.mcts.refinement.optimize_steps
        self.total_optimize_steps = 0

    def set_optimize_steps(self, n):
        """
        Set number of optimization steps

        :param n: number of optimziation steps
        :type n: int
        :return:
        """
        self.optimize_steps = n

    def optimize(self, loss_fun):
        assert False
        loss = 0.
        for iter in range(self.optimize_steps):
            self.optimizer.zero_grad()

            loss = loss_fun()

            loss.backward()
            self.optimizer.step()

            self.total_optimize_steps += 1

        return loss

