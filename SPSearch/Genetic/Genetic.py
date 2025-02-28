import torch
import numpy as np
import random
import time
import copy

from SPSearch.constants.constants import NodesTypes
from SPSearch.DVProposals import DVProposal, TransProposal


class Individual:
    def __init__(self, id, prop_seq):
        self.id = id
        self.prop_seq = prop_seq
        self.fitness = 0
        self.offsprings_n = 0

    @staticmethod
    def create_random_individual(individual_id, decision_variables_list,
                                 optimize_translation,
                                 settings,
                                 device):
        prop_seq = []
        for dv in decision_variables_list:
            value = random.choice(dv.values)

            # print(dv.name, ',', value.value)

            new_proposal = DVProposal(
                dv.name + '_',
                value,
                prop_type=NodesTypes.OTHERNODE)
            prop_seq.append(new_proposal)

        translation = torch.tensor([[0., 0., 0.]],
                         device=device)
        if optimize_translation:
            translation = torch.randn_like(translation) * settings.init_mutation_noise

        translation = torch.nn.Parameter(
            translation,
            requires_grad=optimize_translation)
        trans_proposal = TransProposal('OBJ_Translation',
                                       translation,
                                       prop_type=NodesTypes.OTHERNODE)
        prop_seq.append(trans_proposal)

        return Individual(individual_id, prop_seq)

    def __lt__(self, other):
        return self.fitness < other.fitness

    def get_prop(self, prop_ind):
        return self.prop_seq[prop_ind]

    def set_prop(self, prop_ind, prop):
        self.prop_seq[prop_ind] = prop

    def evaluate(self, game):
        self.fitness = - game.calc_loss_from_proposals(self.prop_seq).item()

    def fitness_to_loss(self):
        return -self.fitness


class Genetic(object):
    def __init__(self, game, scene_reconstructions_path, settings):
        self.game = game

        self.scene_reconstructions_path = scene_reconstructions_path
        self.settings = settings

        self.total_population = 0

        self.mutate_value_fn = self.mutate_random_value
        self.create_individual_fn = Individual.create_random_individual

    def print_population(self, population):
        for i in range(len(population)):
            print('Individual: ', population[i].id, 'Fitness: ', population[i].fitness)

    def get_device(self):
        return self.game.target.device

    # function to initialize the population
    def initialize_population(self, population_size):
        population = []

        for i in range(population_size):
            individual_id = self.generate_individual_id()
            individual = self.create_individual_fn(
                individual_id,
                self.game.decision_var_list,
                self.game.target.optimize_translation,
                self.settings,
                self.get_device())
            population.append(individual)

        return population

    def generate_individual_id(self):
        individual_id = str(self.total_population).zfill(6)
        self.total_population += 1
        return individual_id

    def evaluate_population(self, population):
        best_fitness = -np.inf
        print('\n')
        for i in range(len(population)):
            population[i].evaluate(self.game)
            best_fitness = max(population[i].fitness, best_fitness)
            print('Evaluate Individual: {} \ {}, Best Fitness: {}'.format(i, len(population) - 1,
                                                                 best_fitness),
                                                                 end='\r')
        return population

    # function to select the best individuals from the population
    def selection(self, population, n_to_select):
        population = [individual for individual in population if
                      individual.offsprings_n < self.settings.max_offsprings_per_parent]

        if self.settings.select_criterion == 'keep_n_best':
            population = sorted(population, key=lambda x: x.fitness, reverse=True)
            return population[:n_to_select]
        elif self.settings.select_criterion == 'avg':
            population_fs = sum([individual.fitness for individual in population]) / len(population)
            population = [individual for individual in population if individual.fitness >= population_fs]

            return population


    # function to perform crossover
    def crossover(self, parents, num_offsprings):
        offsprings = []
        decision_var_list = self.game.decision_var_list

        for i in range(num_offsprings):

            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

            parent1.offsprings_n += 1
            parent2.offsprings_n += 1

            offspring = copy.deepcopy(parent2)
            individual_id = self.generate_individual_id()
            offspring.id = individual_id
            offspring.offsprings_n = 0

            for dv_ind, dv in enumerate(decision_var_list):
                prop = offspring.get_prop(dv_ind)
                take_from_parent1 = random.random() < 0.5
                if take_from_parent1:
                    prop = parent1.get_prop(dv_ind)
                    prop = copy.deepcopy(prop)
                offspring.set_prop(dv_ind, prop)


            offsprings.append(offspring)

        return offsprings

    # function to perform mutation

    @staticmethod
    def mutate_random_value(dv):
        values = dv.get_values()
        return random.choice(values)

    def mutation(self, offsprings, mutation_rate_offspring, mutation_rate_param, mutation_noise,
                 translation_mutation_noise):

        for i in range(len(offsprings)):

            if random.random() < mutation_rate_offspring:
                decision_var_list = self.game.decision_var_list
                for dv_ind, dv in enumerate(decision_var_list):
                    prop = offsprings[i].get_prop(dv_ind)
                    if random.random() < mutation_rate_param:
                        prop = copy.deepcopy(prop)
                        value = self.mutate_value_fn(dv)
                        prop.decision_value = value
                        offsprings[i].set_prop(dv_ind, prop)
                    if mutation_noise > 0 and prop.decision_value.value.dtype == torch.float32:
                        prop = copy.deepcopy(prop)
                        prop.decision_value.value += \
                            torch.randn_like(prop.decision_value.value) * mutation_noise
                        offsprings[i].set_prop(dv_ind, prop)

                        if dv.normalized_values:
                            prop.decision_value.value.clamp_(
                                min=-1,
                                max=1)
                        else:
                            prop.decision_value.value.clamp_(
                                min=dv.valid_range[0],
                                max=dv.valid_range[1])

                if self.game.target.optimize_translation and random.random() < mutation_rate_param:
                    translation_prop = offsprings[i].prop_seq[-1]
                    translation_prop.translation_offset[:] += (
                            torch.randn_like(translation_prop.translation_offset) * translation_mutation_noise)

        return offsprings

    def optimize_population(self, population, lr, do_full_optimization=False):
        print('Optimizing population: ')

        # Optimize % of best individuals
        population = self.selection(population, int(len(population) * self.settings.refinement.refinement_probability))

        for i in range(len(population)):

            if do_full_optimization:
                best_prop_seq, best_loss = self.optimize(lr=lr,
                                                         prop_seq=population[i].prop_seq,
                                                         final_optimization=True)
                with torch.no_grad():
                    population[i].prop_seq = best_prop_seq
                    population[i].evaluate(self.game)

            else:
                print('Optimizing Individual: {} \ {}'.format(i, len(population) - 1),
                      end='\r')
                best_prop_seq, best_loss = self.optimize(lr=lr,
                                                         prop_seq=population[i].prop_seq)

                with torch.no_grad():
                    population[i].prop_seq = best_prop_seq
                    population[i].evaluate(self.game)

        return population

    def optimize(self, lr, prop_seq, final_optimization=False):
        best_loss = np.inf
        eps_range = 1e-2

        parameters_to_train = []
        for prop in prop_seq:
            prop_params_list = prop.get_params_list()
            for prop in prop_params_list:
                if any(prop is train_param for train_param in parameters_to_train):
                    continue
                parameters_to_train.append(prop)

        assert parameters_to_train

        if self.settings.refinement.optimizer == 'Adam':
            optimizer = torch.optim.Adam(parameters_to_train, lr=lr)
        elif self.settings.refinement.optimizer == 'SGD':
            optimizer = torch.optim.SGD(parameters_to_train, lr=lr)
        else:
            raise ValueError('Unknown optimizer: %s' % self.settings.refinement.optimizer)

        # best_values = []
        best_props = prop_seq

        optimizer_steps = self.settings.refinement.optimize_steps
        if final_optimization:
            optimizer_steps = self.settings.refinement.final_optimization_steps

        use_refinement_break = self.settings.refinement.use_refinement_break and not final_optimization

        for optim_iter in range(optimizer_steps):
            optimizer.zero_grad()

            curr_loss = self.game.calc_loss_from_proposals(prop_seq)

            if curr_loss > best_loss and use_refinement_break:
                break

            curr_loss.backward()
            optimizer.step()

            with torch.no_grad():
                for dv_ind, dv in enumerate(self.game.decision_var_list):
                    if isinstance(prop_seq[dv_ind], DVProposal):
                        if prop_seq[dv_ind].get_decision_value().get_value().requires_grad:
                            if dv.normalized_values:
                                prop_seq[dv_ind].get_decision_value().value.clamp_(
                                    min=-1 + eps_range,
                                    max=1 + eps_range)
                            else:
                                prop_seq[dv_ind].get_decision_value().value.clamp_(
                                    min=dv.valid_range[0] + eps_range,
                                    max=dv.valid_range[1] + eps_range)

            if curr_loss < best_loss:
                best_loss = curr_loss

                best_props = copy.deepcopy(prop_seq)

        # Set best values to proposals
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

    def reconstruct_scene(self, logger):
        init_population_size = self.settings.init_population_size
        population_size = self.settings.population_size
        incoming_population_size = self.settings.incoming_population_size
        num_generations = self.settings.num_generations
        # num_parents = self.settings.num_parents
        num_offsprings = self.settings.num_offsprings
        init_mutation_rate_offspring = self.settings.init_mutation_rate_offspring
        init_mutation_rate_param = self.settings.init_mutation_rate_param
        final_mutation_rate_offspring = self.settings.final_mutation_rate_offspring
        final_mutation_rate_param = self.settings.final_mutation_rate_param
        init_mutation_noise = self.settings.init_mutation_noise
        final_mutation_noise = self.settings.final_mutation_noise
        translation_mutation_noise = self.settings.translation_mutation_noise

        best_loss = np.inf
        best_individual = None
        start_time = time.time()

        # Initialize population
        with torch.no_grad():
            population = self.initialize_population(init_population_size)
            population = self.evaluate_population(population)

        for curr_gen in range(num_generations):
            start_gen_time = time.time()

            with torch.no_grad():
                mutation_rate_offspring = (final_mutation_rate_offspring +
                                           (init_mutation_rate_offspring - final_mutation_rate_offspring) *
                                           (curr_gen / num_generations))
                mutation_rate_param = (final_mutation_rate_param +
                                       (init_mutation_rate_param - final_mutation_rate_param) *
                                       (curr_gen / num_generations))
                mutation_noise = (final_mutation_noise +
                                       (init_mutation_noise - final_mutation_noise) *
                                       (curr_gen / num_generations))

                offsprings = self.crossover(population, num_offsprings)
                offsprings = self.mutation(offsprings,
                                           mutation_rate_offspring,
                                           mutation_rate_param,
                                           mutation_noise,
                                           translation_mutation_noise)

                if (self.settings.add_random_individuals_every and
                        curr_gen % self.settings.add_random_individuals_every == 0):
                    incoming_population = self.initialize_population(incoming_population_size)
                    incoming_population = self.evaluate_population(incoming_population)

                    offsprings = offsprings + incoming_population

                offsprings = self.evaluate_population(offsprings)

            if (self.settings.refine_every_n_generations > 0 and
                    ((curr_gen % self.settings.refine_every_n_generations == 0 and curr_gen > 0) or
                     curr_gen == num_generations - 1)):
                # Optimize
                # linearly decay learning rate between optimizer_lr and final_optimizer_lr based on current generation
                decay = (curr_gen / num_generations)
                lr = self.settings.refinement.optimizer_lr + \
                    (self.settings.refinement.final_optimizer_lr - self.settings.refinement.optimizer_lr) * decay

                do_full_refinement = (self.settings.refinement.full_refinement_frequency and
                                      curr_gen % self.settings.refinement.full_refinement_frequency == 0)
                self.optimize_population(offsprings, lr, do_full_optimization=do_full_refinement)


            population_offsprings = population + offsprings
            population = self.selection(population_offsprings, population_size)

            end_gen_time = time.time()
            gen_time = end_gen_time - start_gen_time
            print('Generation: ', curr_gen, 'Fitness: ', population[0].fitness, 'Gen. time: ', gen_time)
            best_gen_individual = population[0]

            with torch.no_grad():
                if best_gen_individual.fitness_to_loss() < best_loss:
                    print('Previous best loss: %f' % best_loss)
                    best_loss = best_gen_individual.fitness_to_loss()
                    best_individual = best_gen_individual
                    print('New best loss: %f' % best_loss)

                    logger.log_iteration(best_gen_individual, best_loss, population_offsprings, curr_gen)

        if num_generations == 0:

            best_individual = self.selection(population, 1)[0]

            best_loss = best_individual.fitness_to_loss()
            print('New best loss: %f' % best_loss)

            logger.log_iteration(best_individual, best_loss, None, 0)

        if (num_generations == 0 or curr_gen == num_generations - 1) and \
                self.settings.refinement.final_optimization_steps > 0:
            print('Final optimization for individual with fitness: ', best_individual.fitness)
            best_ind_prop_seq, best_ind_loss = self.optimize(lr=self.settings.refinement.final_optimizer_lr,
                                                             prop_seq=best_individual.prop_seq,
                                                             final_optimization=True)
            best_individual.prop_seq = best_ind_prop_seq
            best_individual.fitness = -best_ind_loss.item()
            final_optim_time = time.time() - start_time

            print('Final Optimization Fitness: ', best_individual.fitness, 'Optimization time: ', final_optim_time)

            if best_ind_loss < best_loss:
                print('Previous best loss: %f' % best_loss)
                best_loss = best_individual.fitness_to_loss()
                best_individual = best_individual
                print('New best loss: %f' % best_loss)

                logger.log_iteration(best_individual, best_loss, None, num_generations + 1)

        logger.print_time()

        return best_individual

