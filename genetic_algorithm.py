'''
This is a script that will read in data and produce the optimal parameters of a
densly connected feed-forward neural network. These parameters are produced
through a genetic algoritm. These parameters include number of layers,
number of nodes in each layer, and activation function node pairs.
'''

from operator import itemgetter
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import keras
import sys


# Format the data structure as follows.
# Model Params: [[shapes (int)], [activation function keys (int)]]

# These are the activation functions that our network will choose from.
activation_function_map = {0:'softmax',
                           1:'relu',
                           2:'sigmoid',
                           3:'tanh'}



def import_clean_data():
    '''
    imports and cleans then returns data
    '''
    # Use methods with these prototypes to impoprt the data.
    #data = pd.read_csv(sys.argv[1], header = None, names = ['1','2','3','4','class'])

    # One-hot-encode categorical data by setting the columns to 'dummy' columns.
    #data = pd.get_dummies(data, columns = ['class'])

    # Split into Training & Testing Data
    training  = data.sample(frac = 4/5, axis = 0)
    testing   = data.drop(training.index)
    train_out = training[['class_Iris-setosa','class_Iris-versicolor','class_Iris-virginica']].values
    del training['class_Iris-setosa']
    del training['class_Iris-versicolor']
    del training['class_Iris-virginica']
    train_in = training.values
    del training
    test_out = testing[['class_Iris-setosa','class_Iris-versicolor','class_Iris-virginica']].values
    del testing['class_Iris-setosa']
    del testing['class_Iris-versicolor']
    del testing['class_Iris-virginica']
    test_in = testing.values
    del testing

    return train_in, train_out, test_in, test_out

### Start? ###
def train_nn(data, model_dict):
    '''
    model_dict-- a dict representation of a keras nn model
               - shape        = 1, 1, 1, 1, num_layers
               - dtype        = model, compile, fit, evaluate, layer*num_layers
               - model_dtype  = func, args, kwargs
               - compile_dtype= args, kwargs
               - fit_dtype    = args, kwargs
               - eval_dtype   = args, kwargs
               - layer_dtype  = func, args, kwargs
               - func         = function
               - args         = tuple of function args
               - kwargs       = dict of function kwargs

    data      -- a list of data pre-cleaned to work with model
    '''
    # load data and define model
    train_in, train_out, test_in, test_out = data
    mfunc, margs, mkwargs = model_dict['model']
    model = mfunc(*margs, **mkwargs)

    # add layers
    layers = pd.DataFrame(model_dict['layers']).to_dict('index')
    for key in layers:
        layer = layers[key]
        func   = layer['func']
        args   = layer['args']
        kwargs = layer['kwargs']
        #print(args, kwargs)
        model.add(func(*args, **kwargs))

    # train and score nn
    cargs, ckwargs = model_dict['compile']
    model.compile(*cargs, **ckwargs)

    fargs, fkwargs = model_dict['fit']
    model.fit(train_in, train_out, *fargs, **fkwargs)

    eargs, ekwargs = model_dict['evaluate']
    scores = model.evaluate(test_in, test_out, *eargs, **ekwargs)
    return scores

def add_layer(model_dict, func, *args, **kwargs):
    '''
    add layers to a model_dict
    '''
    model_dict['layers']['func'].append(func)
    model_dict['layers']['args'].append(args)
    model_dict['layers']['kwargs'].append(kwargs)
    return model_dict

def gen_model(mod, comp, fit, evalu):
    '''
    generate an initial model_dict
    '''
    layers     = {'func':[], 'args':[], 'kwargs':[]}
    model_dict = {'model':mod, 'compile':comp, 'fit':fit, 'evaluate':evalu, 'layers':layers}
    return model_dict

def generate_random_model_params():
    '''
    returns array with form [shapes (int), functions (int)]
    the number of shapes and functions = number of layers in netowrk
    '''
    num_layers = np.random.randint(10) + 1
    functions  = [np.random.randint(4) for _ in range(num_layers)]
    shapes     = [np.random.randint(100) + 1 for _ in range(num_layers)]
    return [shapes, functions]

def compile_model(model_params):
    '''
    returns a keras network model from model parameters
    '''
    # default, same for each network
    mod   = (keras.models.Sequential, (), {})
    comp  = ((), {'loss':'categorical_crossentropy', 'optimizer':'adam', 'metrics':['accuracy']})
    fit   = ((), {'epochs':10, 'batch_size':10, 'verbose':0})
    evalu = ((), {'verbose':0})
    model = gen_model(mod, comp, fit, evalu)


    # generate unique parts of network
    num_layers = len(model_params[0])

    for i in range(num_layers):
        if i == 0:
            # First Layer
            model = add_layer(model, keras.layers.Dense, model_params[0][0], input_dim = 4, activation=activation_function_map[model_params[1][0]])
        elif i == num_layers - 1:
            # Last Layer
            model = add_layer(model, keras.layers.Dense, 3, activation=activation_function_map[model_params[1][0]])
        else:
            # Other Layers
            model = add_layer(model, keras.layers.Dense, model_params[0][i], activation=activation_function_map[model_params[1][i]])

    return model

def avg_over_seeds(num_seeds, data, model):
    '''
    returns the average accuracy of a model over a range of random seeds
    '''
    average = 0
    for i in range(1,num_seeds):
        np.random.seed(i)
        average += train_nn(data,model)[1]
    average = average/(num_seeds-1)
    return average






### Phase 2 ###
def create_initial_generation(generation_size):
    '''
    returns initial set of random networks
    '''
    initial_gen = []
    for _ in range(generation_size):
        initial_gen.append(generate_random_model_params())
    return initial_gen

def create_next_generation(last_generation):
    '''
    mutates, cross breeds, and returns a new generation
    '''
    next_gen = []
    for params1, _ in last_generation:
        for params2, _ in last_generation:
            next_gen.append(mutate(cross_breed(params1, params2)))
    return next_gen

def run_generation(generation, data, num_seeds):
    '''
    runs a bunch of models and returns results
    '''
    results = []
    for model_params in generation:
        model = compile_model(model_params)
        results.append(avg_over_seeds(num_seeds, data, model))
    return results

def select_best(generation, results, num_networks):
    '''
    returns 10 best models in tuple form (model params, score)
    '''
    gen = zip(generation, results)
    gen = sorted(gen,key=itemgetter(1))
    return gen[-num_networks:]


def cross_breed(set1,set2):
    '''
    cross_over returns a new model parameter which has parameters randomly selected from two other models parameters.
    '''
    hybrid = [[],[]]
    length = np.random.choice((len(set1[0]),len(set2[0])))
    for i in range(0,length):
        if(i<len(set1[0]) and i<len(set2[0])):
            hybrid[0].append(np.random.choice((set1[0][i],set2[0][i])))
            hybrid[1].append(np.random.choice((set1[1][i],set2[1][i])))

        elif(i<len(set1[0]) and i>len(set2[0])):
            hybrid[0].append(set1[0][i])
            hybrid[1].append(set1[1][i])

        elif(i<len(set2[0])and i>len(set1[0])):
            hybrid[0].append(set2[0][i])
            hybrid[1].append(set2[1][i])

    return hybrid


def mutate(model_params):
    '''
    Given model parameters, changes one model parameter randomly and returns
    the new set of model parameters.
    '''
    rand_change = np.random.randint(0,2)
    rand_item = np.random.randint(0,len(model_params[rand_change]))

    num_layers = len(model_params[1])

    mutation_map = {0: np.random.randint(1,101) ,
                    1: np.random.randint(0,4)}

    new_choice = mutation_map[rand_change]

    model_params[rand_change][rand_item] = new_choice

    return model_params




def main():

    data = import_clean_data()

    num = 2

    initial_gen       = create_initial_generation(generation_size = num ** 2)
    first_run_results = run_generation(initial_gen, data, num_seeds = 2)
    best_networks     = select_best(initial_gen, first_run_results, num_networks = num)

    print("\nEvolving...")
    for i in tqdm(range(3)): # <- Amount of total runs
        next_gen         = create_next_generation(best_networks)
        next_run_results = run_generation(next_gen, data, num_seeds = 2)
        best_networks    = select_best(next_gen, next_run_results, num_networks = num)

    print("\n")

    for p,r in best_networks:
        for i in range(len(p[0])):
            print("Layer: ", i, " - Nodes: ", p[0][i], " Activation: ", activation_function_map[p[1][i]])
        print("Network Result: ", r, "\n")






if __name__ == "__main__":
    main()
