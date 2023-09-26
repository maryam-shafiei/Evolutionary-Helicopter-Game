import copy
import math
import random
from statistics import mean

from player import Player
import numpy as np
from config import CONFIG


class Evolution():

    def __init__(self, mode):
        self.mode = mode

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutate(self, child):

        # TODO
        # child: an object of class `Player`
        p_m = 0.3
        for param in child.nn.params:
            if p_m > np.random.uniform(0, 1):
                noise = np.random.normal(0, 1, child.nn.params[param].shape)
                child.nn.params[param] += noise
        return child

    def generate_new_population(self, num_players, prev_players=None):

        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:

            # TODO
            # num_players example: 150
            # prev_players: an array of `Player` objects
            # parents = self.roulette_wheel(prev_players, num_players)
            # TODO (additional): a selection method other than `fitness proportionate`
            parents = self.q_tournament(prev_players, num_players, 5)
            new_players = copy.deepcopy(parents)
            # TODO (additional): implementing crossover
            new_players = self.cross_over(new_players, num_players)
            children = [self.mutate(x) for x in copy.deepcopy(new_players)]
            return children

    def next_population_selection(self, players, num_players):

        # TODO
        # num_players example: 100
        # players: an array of `Player` objects
        #next_population = sorted(players, key=lambda x: x.fitness, reverse=True)[:num_players]
        # TODO (additional): a selection method other than `top-k`
        next_population = self.roulette_wheel(players, num_players)
        # TODO (additional): plotting
        self.save_fitness(players)
        return next_population

    def roulette_wheel(self, players, num_players):
        total_fitness = sum([player.fitness for player in players])
        prob_player = []
        for player in players:
            prob_player.append(player.fitness / total_fitness)
        return list(np.random.choice(a=players, size=num_players, replace=False, p=prob_player))

    def q_tournament(self, players, num_players, q):
        parents = []
        for i in range(num_players):
            q_selection = np.random.choice(a=players, size=q)
            best = max(q_selection, key=lambda x: x.fitness)
            parents.append(best)
        return parents

    def cross_over(self, players, num_players):
        children = []
        index = 0
        p_c = 0.7
        iteration_num = math.floor(num_players / 2)
        for _ in range(iteration_num):
            p = np.random.uniform(0, 1)
            if p < p_c:
                child1 = Player(self.mode)
                child2 = Player(self.mode)
                parent1 = players[index]
                parent2 = players[index + 1]
                cross_over_point_1 = math.floor(parent1.nn.layer_sizes[1] / 2)
                cross_over_point_2 = math.floor(parent1.nn.layer_sizes[2] / 2)
                child1.nn.params['w1'] = np.concatenate((parent1.nn.params['w1'][:cross_over_point_1], parent2.nn.params['w1'][cross_over_point_1:]), axis=0)
                child2.nn.params['w1'] = np.concatenate((parent2.nn.params['w1'][:cross_over_point_1], parent1.nn.params['w1'][cross_over_point_1:]), axis=0)

                child1.nn.params['w2'] = np.concatenate((parent1.nn.params['w2'][:cross_over_point_2], parent2.nn.params['w2'][cross_over_point_2:]), axis=0)
                child2.nn.params['w2'] = np.concatenate((parent2.nn.params['w2'][:cross_over_point_2], parent1.nn.params['w2'][cross_over_point_2:]), axis=0)

                child1.nn.params['b1'] = np.concatenate((parent1.nn.params['b1'][:cross_over_point_1], parent2.nn.params['b1'][cross_over_point_1:]), axis=0)
                child2.nn.params['b1'] = np.concatenate((parent2.nn.params['b1'][:cross_over_point_1], parent1.nn.params['b1'][cross_over_point_1:]), axis=0)

                child1.nn.params['b2'] = np.concatenate((parent1.nn.params['b2'][:cross_over_point_2], parent2.nn.params['b2'][cross_over_point_2:]), axis=0)
                child2.nn.params['b2'] = np.concatenate((parent2.nn.params['b2'][:cross_over_point_2], parent1.nn.params['b2'][cross_over_point_2:]), axis=0)

                children.append(child1)
                children.append(child2)
            else:
                children.append(players[index])
                children.append(players[index + 1])
            index += 2
        if num_players % 2 != 0:
            children.append(players[-1])
        return children


    def save_fitness(self, players):
        max_fitness_player = max(players, key=lambda x: x.fitness)
        max_fitness = max_fitness_player.fitness
        min_fitness_player = min(players, key=lambda x: x.fitness)
        min_fitness = min_fitness_player.fitness
        avg_fitness = mean([player.fitness for player in players])
        self.write_in_file(f"max_fitness_{self.mode}.csv", str(max_fitness))
        self.write_in_file(f"min_fitness_{self.mode}.csv", str(min_fitness))
        self.write_in_file(f"avg_fitness_{self.mode}.csv", str(avg_fitness))

    def write_in_file(self, filename, value):
        f = open('data/'+filename, "a")
        f.write(value)
        f.write('\n')
        f.close()