from __future__ import annotations
from os import ctermid
import random
import sys
import sge.grammar as grammar
import sge.logger as logger
from datetime import datetime
from tqdm import tqdm
from sge.operators.recombination import crossover
from sge.operators.mutation import mutate
from sge.operators.selection import tournament
from sge.parameters import (
    params,
    set_parameters,
    load_parameters
)
class SearchSpace:
    def __init__(self, restriction_dictionary:dict, parent:SearchSpace, type:str) -> None:
        self.restriction_dictionary = restriction_dictionary
        self.parent = None #Parent SubSpace
        self.children = []
        self.type = type
    
    def check_individual(self, individual):
        if self.type == "exploit":
            for individual_rule_choices, exploit_dictionary_rule in zip(individual['genotype'], self.restriction_dictionary):
                choice_tracker = ''
                for rule_choice in individual_rule_choices:
                    if choice_tracker in exploit_dictionary_rule and not (rule_choice in exploit_dictionary_rule[choice_tracker]):
                        return False
                    choice_tracker += str(rule_choice)
            return True
        elif self.type == "explore":
            for individual_rule_choices, explore_dictionary_rule in zip(individual['genotype'], self.restriction_dictionary):
                choice_tracker = ''
                for rule_choice in individual_rule_choices:
                    if choice_tracker in explore_dictionary_rule and rule_choice in explore_dictionary_rule[choice_tracker]:
                        return True
                    choice_tracker += str(rule_choice)
            return False 
        else:
            raise AssertionError("Invalid SearchSpace type.")

def generate_random_individual():
    genotype = [[] for key in grammar.get_non_terminals()]
    tree_depth = grammar.recursive_individual_creation(genotype, grammar.start_rule()[0], 0)
    return {'genotype': genotype, 'fitness': None, 'tree_depth' : tree_depth}


def make_initial_population():
    for i in range(params['POPSIZE']):
        yield generate_random_individual()

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
    setup(parameters_file_path=parameters_file)
    grammar_options_list = [
        {0},
        {0, 1, 2},
        {0, 1, 2},
        {0, 1},
    ]
    population = list(make_initial_population())
    it = 0
    while it <= params['GENERATIONS']:
        for i in tqdm(population):
            if i['fitness'] is None:
                evaluate(i, evaluation_function)
            print(i['genotype'], i['fitness'])
        population.sort(key=lambda x: x['fitness'])
        exploit_dictionary = create_exploit_dictionary(population)
        explore_dictionary = create_explore_dictionary(exploit_dictionary, grammar_options_list)
        explore_search_space = SearchSpace(explore_dictionary, None, 'explore')
        exploit_search_space = SearchSpace(exploit_dictionary, None, 'exploit')

        for individual in population:
            if explore_search_space.check_individual(individual) or not exploit_search_space.check_individual(individual):
                print("error")

        logger.evolution_progress(it, population)
        new_population = population[:params['ELITISM']]
        while len(new_population) < params['POPSIZE']:
            if random.random() < params['PROB_CROSSOVER']:
                p1 = tournament(population, params['TSIZE'])
                p2 = tournament(population, params['TSIZE'])
                ni = crossover(p1, p2)
            else:
                ni = tournament(population, params['TSIZE'])
            ni = mutate(ni, params['PROB_MUTATION'])
            new_population.append(ni)
        population = new_population
        it += 1

