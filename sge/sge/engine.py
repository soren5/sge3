from __future__ import annotations
from os import ctermid
import os
import random
from re import search, sub
import re
import sys
from types import new_class

from numpy.lib.function_base import append, copy
import sge.grammar as grammar
import sge.logger as logger
from datetime import datetime
import math
from sge.operators.recombination import crossover
from sge.operators.mutation import mutate
from sge.operators.selection import tournament
from sge.parameters import (
    params,
    set_parameters,
    load_parameters
)
creation_attempts = 0
print_assistant = 1
cwd_path = os.getcwd()

def retry_counter():
    global creation_attempts, print_assistant
    if creation_attempts % print_assistant == 0:
        #print("Retry number:", creation_attempts)
        print_assistant*=2
    creation_attempts += 1

def tournament_search_space(population, tsize=3):
    pool = random.sample(population, tsize)
    pool.sort(key=lambda i: i.average_distance_from_average)
    return copy.deepcopy(pool[-1])
        
def generate_random_individual():
    genotype = [[] for key in grammar.get_non_terminals()]
    tree_depth = grammar.recursive_individual_creation(genotype, grammar.start_rule()[0], 0)
    return {'genotype': genotype, 'fitness': None, 'tree_depth' : tree_depth}

def make_initial_population(search_space: SearchSpace):
    for i in range(params['POPSIZE']):
        while True:
            individual = generate_random_individual()
            if search_space.check_individual(individual):
                break
        yield individual

def evaluate(ind, eval_func):
    mapping_values = [0 for i in ind['genotype']]
    phen, tree_depth = grammar.mapping(ind['genotype'], mapping_values)
    quality, other_info = eval_func.evaluate(phen)
    ind['phenotype'] = phen
    ind['fitness'] = quality
    ind['other_info'] = other_info
    ind['mapping_values'] = mapping_values
    ind['tree_depth'] = tree_depth

def setup(parameters_file_path = None):
    if parameters_file_path is not None:
        load_parameters(file_name=parameters_file_path)
    set_parameters(sys.argv[1:])
    if params['SEED'] is None:
        params['SEED'] = int(datetime.now().microsecond)
    logger.prepare_dumps()
    random.seed(params['SEED'])
    grammar.set_path(params['GRAMMAR'])
    grammar.read_grammar()
    grammar.set_max_tree_depth(params['MAX_TREE_DEPTH'])
    grammar.set_min_init_tree_depth(params['MIN_TREE_DEPTH'])

class SearchSpace:
    def __init__(self, search_dictionary:dict, restriction_dictionaries:list, parent:SearchSpace, depth: int, population: list) -> None:
        self.search_dictionary = search_dictionary
        self.restriction_dictionaries = restriction_dictionaries
        self.parent = parent #Parent SubSpace
        self.depth = depth
        self.population = population
        self.children = []
        self.total_fitness = 0
        self.total_evaluations = 0
        self.total_distance_from_average = 0
        self.average_distance_from_average = 0
    
    def check_individual(self, individual):
        if self.search_dictionary != None: #Is there a large circle?
            #Then check if the individual is inside it
            for individual_rule_choices, exploit_dictionary_rule in zip(individual['genotype'], self.search_dictionary):
                choice_tracker = ''
                for rule_choice in individual_rule_choices:
                    if choice_tracker in exploit_dictionary_rule and not (rule_choice in exploit_dictionary_rule[choice_tracker]):
                        retry_counter()
                        return False
                    choice_tracker += str(rule_choice)
        if self.restriction_dictionaries == None: #Are there small circles?
            return True #We get here if there are no small circles and the individual is inside the large circle (or there is no large circle)
        else:
            for exclusion_dictionary in self.restriction_dictionaries:
                for individual_rule_choices, explore_dictionary_rule in zip(individual['genotype'], exclusion_dictionary):
                    choice_tracker = ''
                    exclusion_condition_is_met = False
                    for rule_choice in individual_rule_choices:
                        if choice_tracker in explore_dictionary_rule and rule_choice in explore_dictionary_rule[choice_tracker]:
                            exclusion_condition_is_met = True #The individual is outside this smaller circle
                            break
                        choice_tracker += str(rule_choice)
                    if exclusion_condition_is_met:
                        break
                if not exclusion_condition_is_met:
                    retry_counter()
                    return False #The individual is inside one of the smaller circles
            return True #We have checked all the smaller circles and the individual is inside none of them

    def initialize_population(self, population_generation_function):
        self.population = list(population_generation_function(self))
        
    def check_propagate_fitness(self, fitness):
        #Use unique evaluations
        #Use number of rejections
        self.total_evaluations += 1
        self.total_fitness += fitness
        self.total_distance_from_average += abs(self.total_fitness / self.total_evaluations - fitness)
        self.average_distance_from_average = self.total_distance_from_average / self.total_evaluations
        if self.parent != None:
            self.parent.check_propagate_fitness(fitness)

def create_exploit_dictionary(population):
    exploit_dictionary = [{},{},{},{}]
    for individual in population:
        for individual_rule_choices, exploit_dictionary_rule in zip(individual['genotype'], exploit_dictionary):
            choice_tracker = ''
            for rule_choice in individual_rule_choices:
                if choice_tracker in exploit_dictionary_rule:
                    exploit_dictionary_rule[choice_tracker].add(rule_choice)
                else:
                    exploit_dictionary_rule[choice_tracker] = {rule_choice}
                choice_tracker += str(rule_choice)
    return exploit_dictionary

def create_explore_dictionary(exploit_dictionary, grammar_options_list):
    explore_dictionary = [{},{},{},{}]
    for exploit_dictionary_rule, explore_dictionary_rule, rule_options in zip(exploit_dictionary, explore_dictionary, grammar_options_list):
        for choice_tracker in exploit_dictionary_rule:
            if exploit_dictionary_rule[choice_tracker] != rule_options:
                explore_dictionary_rule[choice_tracker] = rule_options.copy()
                for production in exploit_dictionary_rule[choice_tracker]:
                    explore_dictionary_rule[choice_tracker].discard(production)
    return explore_dictionary

def evolutionary_algorithm(evaluation_function=None, parameters_file=None):
    global creation_attempts


    generations_for_reset_options = [5,10,25,50]
    experiments_dict = ['keep_best', 'subspace_search', 'default', 'random_restart']

    for generations_for_reset in generations_for_reset_options:
        for experiment in experiments_dict:
            with open(os.path.join(cwd_path, 'results_quartic_polynomial/' , "{}_{}.csv".format(experiment, generations_for_reset)), 'a') as f:
                print("Seed, Generations, Evaluations, Unique Evaluations, Iterations, Rejections, Fitness", file=f)
            print("Seed, Generations, Evaluations, Unique Evaluations, Iterations, Rejections, Fitness")
            for _ in range(30):
                creation_attempts = 0
                setup(parameters_file_path=parameters_file)
                random.seed(_)
                grammar_options_list = [
                    {0},
                    {0, 1, 2},
                    {0, 1, 2},
                    {0, 1},
                ]
                history = {}
                current_search_space = SearchSpace(None, None, None, 0, None)
                search_space_list = [current_search_space]
                population = list(make_initial_population(current_search_space))
                current_search_space.population = population
                it = 0
                unique_evals = 0
                evals = 0
                perfect_solution_found = False
                while not perfect_solution_found:
                    for ix in range(len(population)):
                        string_genotype = str(population[ix]['genotype'])
                        evals += 1
                        if string_genotype in history:
                            population[ix] = history[string_genotype]
                        if population[ix]['fitness'] is None:
                            evaluate(population[ix], evaluation_function)
                            current_search_space.check_propagate_fitness(population[ix]['fitness'])
                            unique_evals += 1
                            if population[ix]['fitness'] == 0:
                                with open(os.path.join(cwd_path, 'results_quartic_polynomial/' , "{}_{}.csv".format(experiment, generations_for_reset)), 'a') as f:
                                    line = "{}, {}, {}, {}, {}, {}, {}".format(_, generations_for_reset, evals, unique_evals, it,  creation_attempts, 0)
                                    print(line, file=f)
                                print(line)
                                perfect_solution_found = True
                            history[string_genotype] = population[ix]
                        #print(population[ix]['genotype'], population[ix]['fitness'])
                    population.sort(key=lambda x: x['fitness'])

                    logger.evolution_progress(it, population)
                    new_population = population[:params['ELITISM']]
                    while len(new_population) < params['POPSIZE']:
                        random_value = random.random()
                        attempt_counter = 0
                        while True:
                            if  random_value < params['PROB_CROSSOVER']:
                                p1 = tournament(population, params['TSIZE'])
                                p2 = tournament(population, params['TSIZE'])
                                ni = crossover(p1, p2)

                            else:
                                ni = tournament(population, params['TSIZE'])
                            ni = mutate(ni, params['PROB_MUTATION'])
                            if current_search_space.check_individual(ni):
                                break
                        new_population.append(ni)
                    best_individual = population[0]
                    population = new_population
                    current_search_space.population = population
                    it += 1
                    if it > 1000:
                        perfect_solution_found = True
                        with open(os.path.join(cwd_path, 'results_quartic_polynomial/' , "{}_{}.csv".format(experiment, generations_for_reset)), 'a') as f:
                            line = "{}, {}, {}, {}, {}, {}".format(_, generations_for_reset, evals, unique_evals, it,  best_individual['fitness'])
                            print(line, file=f)
                        print(line)

                        
                    if it % generations_for_reset == 0:
                        if experiment in ["random_restart", "keep_best"]:
                            population = list(make_initial_population(current_search_space))
                            if experiment == "keep_best":
                                population[0] = best_individual
                        if experiment == "subspace_search":
                            exploit_dictionary = create_exploit_dictionary(population)
                            explore_dictionary = create_explore_dictionary(exploit_dictionary, grammar_options_list)
                            
                            other_search_space = SearchSpace(exploit_dictionary, None, current_search_space, current_search_space.depth + 1, []) 
                            other_search_space.initialize_population(make_initial_population)
                            #for i in population:
                            #    other_search_space.check_propagate_fitness(i['fitness'])

                            if current_search_space.restriction_dictionaries != None:
                                restriction_dictionaries = [x for x in current_search_space.restriction_dictionaries]
                                restriction_dictionaries.append(explore_dictionary)
                            else:
                                restriction_dictionaries = [explore_dictionary]
                            new_search_space = SearchSpace(current_search_space.search_dictionary, restriction_dictionaries, current_search_space, current_search_space.depth + 1, [])
                            new_search_space.initialize_population(make_initial_population)

                            search_space_list.append(new_search_space)
                            search_space_list.append(other_search_space)
                            current_search_space = new_search_space
                            population = current_search_space.population

